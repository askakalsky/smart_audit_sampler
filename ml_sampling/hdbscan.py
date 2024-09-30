import pandas as pd
import optuna
from typing import Tuple, List
import numpy as np
import logging
from sklearn.cluster import HDBSCAN
import datetime
from scoring_methods.silhouette import calculate_silhouette

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def hdbscan_sampling(
    population_original: pd.DataFrame,
    population: pd.DataFrame,
    sample_size: int,
    features: List[str],  # Используется только для документации
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
    """
    Anomaly sampling using the HDBSCAN clustering algorithm with hyperparameter tuning via Optuna.

    Parameters:
    -----------
    population_original : pd.DataFrame
        The original dataset to be processed and annotated with cluster information.
    population : pd.DataFrame
        A copy of the original dataset used for clustering and anomaly detection.
    sample_size : int
        The desired number of anomalies to be sampled.
    features : List[str]
        List of features used for clustering (used for documentation purposes only).
    random_seed : int
        Seed for random number generation to ensure reproducibility.

    Returns:
    --------
    Tuple containing:
        - population_original : pd.DataFrame
          The original dataset updated with cluster and sample annotations.
        - population : pd.DataFrame
          The processed dataset containing cluster and anomaly information.
        - sample_processed : pd.DataFrame
          Subset of the dataset containing the sampled anomalies.
        - method_description : str
          Description of the method used for sampling.
        - study : optuna.Study
          The Optuna study object with results of hyperparameter optimization.
    """
    try:
        # Make copies to avoid modifying the original DataFrames
        population_original = population_original.copy()
        population = population.copy()

        # Initialize random number generator for reproducibility
        rng = np.random.default_rng(random_seed)

        def choose_algorithm(n_samples: int, n_features: int) -> str:
            """
            Chooses the optimal algorithm for HDBSCAN based on data size and feature count.

            Args:
                n_samples (int): Number of samples in the dataset.
                n_features (int): Number of features in the dataset.

            Returns:
                str: The name of the algorithm to use ('kdtree' or 'balltree').
            """
            if n_features < 20 and n_samples < 10000:
                return 'kdtree'
            else:
                return 'balltree'

        chosen_algorithm = choose_algorithm(
            population.shape[0], population.shape[1])

        # Define the Optuna objective function for hyperparameter tuning
        def objective(trial):
            params = {
                'min_cluster_size': trial.suggest_int('min_cluster_size', 2, 20),
                'min_samples': trial.suggest_int('min_samples', 1, 20),
                'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', 0.0, 1.0),
                'alpha': trial.suggest_float('alpha', 0.0, 2.0),
                'metric': 'euclidean',
                'algorithm': chosen_algorithm,
                'cluster_selection_method': 'eom',
                'allow_single_cluster': False
            }
            clusterer = HDBSCAN(**params)
            return calculate_silhouette(clusterer, population)

        # Run Optuna optimization with random seed for reproducibility
        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
        study.optimize(objective, n_trials=100, show_progress_bar=True)
        best_params = study.best_params
        best_clusterer = HDBSCAN(**best_params)
        labels = best_clusterer.fit_predict(population)

        # Annotate clusters and anomalies
        population_original['cluster'] = labels
        population_original['is_anomaly'] = (labels == -1).astype(int)
        population_original['is_sample'] = 0
        population['cluster'] = labels
        population['is_anomaly'] = (labels == -1).astype(int)
        population['is_sample'] = 0
        num_clusters = max(population_original['cluster']) + 1

        # Identify anomalies
        anomalies = population_original[population_original['is_anomaly'] == 1]

        # Check if there are any anomalies, if not raise an error or warning
        if anomalies.empty:
            raise ValueError("У наборі даних не було виявлено аномалій.")

        # Sample anomalies based on the requested sample size
        if len(anomalies) > sample_size:
            sample_indices = rng.choice(
                anomalies.index, size=sample_size, replace=False)
            sample_processed = anomalies.loc[sample_indices]
            warning_message = ""
        else:
            sample_processed = anomalies
            warning_message = f"Warning: Only {len(anomalies)} anomalies found, less than the requested sample size of {sample_size}."

        # Mark the sampled data
        population_original.loc[sample_processed.index, 'is_sample'] = 1
        population.loc[sample_processed.index, 'is_sample'] = 1
        sample_processed['is_sample'] = 1

        # Total population size and the number of anomalies detected
        total_anomalies = len(anomalies)
        population_size = len(population_original)

        method_description = (
            f"Sampling using HDBSCAN with automatic hyperparameter tuning (Optuna).\n"
            f"{warning_message}\n"
            f"Number of sampled anomalies: {len(sample_processed)}.\n"
            f"Number of detected anomalies: {total_anomalies}.\n"
            f"Total population size: {population_size}.\n"
            f"Number of clusters: {num_clusters}.\n"
            f"Features: {features}.\n"
            f"Best parameters: {best_params}.\n"
            f"Sample creation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
            f"Number of trials: {len(study.trials)}.\n"
        )

        return population_original, population, sample_processed, method_description, study

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {ve}", None

    except KeyError as ke:
        logger.error(f"KeyError: {ke}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"KeyError: {ke}", None

    except Exception as e:
        logger.exception(f"Unexpected error in hdbscan_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}", None
