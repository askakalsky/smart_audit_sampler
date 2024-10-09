import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import optuna
from sklearn.metrics import calinski_harabasz_score
from typing import Tuple, List
import logging
import datetime
import numpy as np
import heapq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def kmeans_sampling(
    population_original: pd.DataFrame,
    population: pd.DataFrame,
    sample_size: int,
    features: List[str],
    random_seed: int,
    progress_callback=None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
    """
    Performs sampling using K-Means clustering with Optuna-based hyperparameter optimization.
    The clustering quality is evaluated using dynamically selected metrics. 
    The selection of samples is based on relative distances to centroids within each cluster.
    """
    try:
        # Make copies to avoid modifying the original DataFrames
        population_original = population_original.copy()
        population = population.copy()

        def objective(trial):
            params = {
                'n_clusters': trial.suggest_int('n_clusters', 2, 20),
                'init': 'k-means++',
                'n_init': trial.suggest_int('n_init', 5, 30),
                'max_iter': trial.suggest_int('max_iter', 100, 500)
            }

            # Fit KMeans with the current parameters
            kmeans = KMeans(random_state=random_seed, **params)
            kmeans.fit(population)

            # Get cluster labels
            labels = kmeans.labels_

            # Calculate Calinski-Harabasz score for the current clustering
            score = calinski_harabasz_score(population, labels)

            return score

        n_trials = 100

        def optuna_callback(study, trial):
            if progress_callback is not None:
                progress = int((len(study.trials) / n_trials) * 100)
                progress_callback(progress)

        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_seed))
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[optuna_callback])

        best_params = study.best_params
        best_kmeans = KMeans(random_state=random_seed, **best_params)
        best_kmeans.fit(population)

        clusters = best_kmeans.predict(population)
        _, distances = pairwise_distances_argmin_min(
            population, best_kmeans.cluster_centers_)

        population_original['distance_to_centroid'] = distances
        population_original['cluster'] = clusters
        population['distance_to_centroid'] = distances
        population['cluster'] = clusters

        # Calculate relative distances within each cluster
        population['relative_distance'] = 0.0
        for cluster in np.unique(clusters):
            cluster_distances = population.loc[population['cluster']
                                               == cluster, 'distance_to_centroid']
            std_dev = cluster_distances.std()
            if std_dev > 0:
                population.loc[population['cluster'] == cluster,
                               'relative_distance'] = cluster_distances / std_dev
            else:
                population.loc[population['cluster'] ==
                               cluster, 'relative_distance'] = 0.0

        # Use the relative distances to select samples
        if sample_size < len(population_original):
            largest_relative_distances_indices = heapq.nlargest(
                sample_size, range(len(population)), population['relative_distance'].values.take)
            sample_processed = population_original.iloc[largest_relative_distances_indices]
        else:
            logger.warning(
                f"Requested sample size {sample_size} exceeds available data size {len(population_original)}. Using full data.")
            sample_processed = population_original

        population_original['is_sample'] = 0
        population['is_sample'] = 0
        population_original.loc[sample_processed.index, 'is_sample'] = 1
        population.loc[sample_processed.index, 'is_sample'] = 1
        sample_processed = sample_processed.copy()
        sample_processed['is_sample'] = 1

        population_size = len(population_original)
        num_clusters = best_params['n_clusters']

        method_description = (
            f"K-Means sampling with hyperparameter optimization using Optuna.\n"
            f"Sample size: {sample_size}.\n"
            f"Total population size: {population_size}.\n"
            f"Number of clusters: {num_clusters}.\n"
            f"Features: {features}.\n"
            f"Best K-Means parameters: {best_params}.\n"
            f"Creation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
            f"Number of trials: {len(study.trials)}.\n"
        )

        if progress_callback is not None:
            progress_callback(100)

        return population_original, population, sample_processed, method_description, study

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {ve}", None

    except KeyError as ke:
        logger.error(f"KeyError: {ke}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"KeyError: {ke}", None

    except Exception as e:
        logger.exception(f"Unexpected error in kmeans_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}", None
