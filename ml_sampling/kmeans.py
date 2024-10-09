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
    progress_callback=None,
    anomalies_per_cluster: int = None  # Новый параметр
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
    """
    Performs sampling using K-Means clustering with Optuna-based hyperparameter optimization.
    The clustering quality is evaluated using dynamically selected metrics. 
    The selection of samples is based on relative distances to centroids within each cluster.
    Anomalies are selected uniformly from each cluster based on their relative distance.

    Parameters:
    - population_original: Original DataFrame before any processing.
    - population: DataFrame to perform clustering on.
    - sample_size: Total number of samples to select.
    - features: List of feature names to use for clustering.
    - random_seed: Seed for reproducibility.
    - progress_callback: Optional callback for reporting progress.
    - anomalies_per_cluster: Number of anomalies to select per cluster. If None, sample_size is distributed proportionally.

    Returns:
    - population_original: Original DataFrame with 'distance_to_centroid', 'cluster', and 'is_sample' columns.
    - population: Processed DataFrame with 'distance_to_centroid', 'cluster', 'relative_distance', and 'is_sample' columns.
    - sample_processed: DataFrame containing the selected samples.
    - method_description: Description of the sampling method.
    - study: Optuna study object.
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

        # Determine anomalies per cluster
        unique_clusters = np.unique(clusters)
        num_clusters = len(unique_clusters)

        if anomalies_per_cluster is not None:
            total_anomalies = anomalies_per_cluster * num_clusters
            if total_anomalies > sample_size:
                logger.warning(
                    f"Requested anomalies_per_cluster * num_clusters ({total_anomalies}) exceeds sample_size ({sample_size}). Adjusting anomalies_per_cluster.")
                anomalies_per_cluster = sample_size // num_clusters
        else:
            anomalies_per_cluster = sample_size // num_clusters
            if anomalies_per_cluster == 0:
                anomalies_per_cluster = 1  # At least one per cluster

        selected_indices = []
        for cluster in unique_clusters:
            cluster_data = population.loc[population['cluster'] == cluster]
            # Sort by relative_distance descending
            sorted_cluster = cluster_data.sort_values(
                by='relative_distance', ascending=False)
            # Select top anomalies_per_cluster indices
            top_indices = sorted_cluster.head(
                anomalies_per_cluster).index.tolist()
            selected_indices.extend(top_indices)

        # If sample_size is not perfectly divisible, fill the remaining
        remaining = sample_size - len(selected_indices)
        if remaining > 0:
            # Select remaining points from all clusters based on relative_distance
            remaining_indices = heapq.nlargest(
                remaining,
                population.index,
                population['relative_distance'].take)
            selected_indices.extend(remaining_indices)

        # Ensure we don't exceed the population size
        selected_indices = selected_indices[:sample_size]

        sample_processed = population_original.loc[selected_indices].copy()

        population_original['is_sample'] = 0
        population['is_sample'] = 0
        population_original.loc[selected_indices, 'is_sample'] = 1
        population.loc[selected_indices, 'is_sample'] = 1
        sample_processed['is_sample'] = 1

        population_size = len(population_original)
        best_num_clusters = best_params['n_clusters']

        method_description = (
            f"K-Means sampling with hyperparameter optimization using Optuna.\n"
            f"Sample size: {sample_size}.\n"
            f"Total population size: {population_size}.\n"
            f"Number of clusters: {best_num_clusters}.\n"
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
