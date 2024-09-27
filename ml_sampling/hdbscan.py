import pandas as pd
import optuna
from typing import Tuple, List
import numpy as np
import logging
from hdbscan import HDBSCAN
import datetime
from scoring_methods.silhouette import calculate_silhouette_hdbscan

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def hdbscan_sampling(
    population_original: pd.DataFrame,
    population: pd.DataFrame,
    sample_size: int,
    features: List[str],
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
    """
    Performs anomaly sampling using the HDBSCAN clustering algorithm with hyperparameter tuning via Optuna.
    The function clusters the data, marks anomalies, and returns a sample of anomalies. 
    It also optimizes the clustering parameters to maximize the silhouette score.

    Args:
        population_original (pd.DataFrame): Original DataFrame to be processed and annotated with cluster information.
        population (pd.DataFrame): A copy of the original DataFrame used for clustering and anomaly detection.
        sample_size (int): The desired number of anomalies to sample.
        features (List[str]): The list of feature columns used for clustering.
        random_seed (int): Seed for random number generation to ensure reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
            - Updated original population DataFrame with cluster and sample annotations.
            - Processed population DataFrame with cluster and anomaly information.
            - DataFrame containing the sampled anomalies.
            - A description of the method used for sampling.
            - The Optuna study object that contains the results of hyperparameter optimization.
    """
    try:
        # Make copies to avoid modifying the original DataFrames
        population_original = population_original.copy()
        population = population.copy()

        # Initialize random number generator
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
            return calculate_silhouette_hdbscan(clusterer, population)

        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)
        best_params = study.best_params
        best_clusterer = HDBSCAN(**best_params)
        labels = best_clusterer.fit_predict(population)

        # Annotate clusters and anomalies
        population_original['cluster'] = labels
        population_original['is_anomaly'] = (labels == -1).astype(int)
        population_original['is_sample'] = False
        population['cluster'] = labels
        population['is_anomaly'] = (labels == -1).astype(int)
        population['is_sample'] = False

        # Identify anomalies
        anomalies = population_original[population_original['is_anomaly'] == 1]

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
        population_original.loc[sample_processed.index, 'is_sample'] = True
        population.loc[sample_processed.index, 'is_sample'] = True

        method_description = (
            f"Sampling using HDBSCAN with automatic hyperparameter tuning (Optuna).\n"
            f"{warning_message}\n"
            f"Number of sampled anomalies: {len(sample_processed)}.\n"
            f"Requested features: {features}.\n"
            f"Used features: {population.columns.tolist()}.\n"
            f"Best parameters: {best_params}.\n"
            f"Sample creation date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
        )

        return population_original, population, sample_processed, method_description, study

    except ValueError as e:
        logger.error(f"ValueError: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {e}", None
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}", None
