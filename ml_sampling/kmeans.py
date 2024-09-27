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
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Performs sampling using K-Means clustering with Optuna-based hyperparameter optimization.
    The clustering quality is evaluated using dynamically selected metrics.

    Parameters:
    -----------
    population_original : pd.DataFrame
        Original dataset to apply clustering on.
    population : pd.DataFrame
        Preprocessed dataset with selected features for clustering.
    sample_size : int
        Number of samples to retrieve based on distance to cluster centroid.
    features : List[str]
        List of features used for clustering.
    random_seed : int
        Seed for random number generation.

    Returns:
    --------
    Tuple containing:
        - population_original : pd.DataFrame
          Original dataset with clustering results.
        - population : pd.DataFrame
          Preprocessed dataset with clustering results.
        - sample_processed : pd.DataFrame
          Subset of the dataset containing the selected samples.
        - method_description : str
          Description of the sampling process and parameters used.
    """
    try:
        population_original = population_original.copy()
        population = population.copy()

        def objective(trial):
            # Define hyperparameters for K-Means
            params = {
                'n_clusters': trial.suggest_int('n_clusters', 2, 20),
                'init': 'k-means++',
                'n_init': trial.suggest_int('n_init', 5, 30),
                'max_iter': trial.suggest_int('max_iter', 100, 500)
            }

            # Обучаем модель KMeans с текущими параметрами
            kmeans = KMeans(random_state=random_seed, **params)
            kmeans.fit(population)

            # Получаем метки кластеров
            labels = kmeans.labels_

            # Вычисляем Silhouette Score
            score = calinski_harabasz_score(population, labels)

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)

        # Best parameters after optimization
        best_params = study.best_params
        best_kmeans = KMeans(random_state=random_seed, **best_params)
        best_kmeans.fit(population)

        # Cluster predictions
        clusters = best_kmeans.predict(population)
        if not np.issubdtype(clusters.dtype, np.integer):
            clusters = clusters.astype(int)

        # Distance to centroids
        _, distances = pairwise_distances_argmin_min(
            population, best_kmeans.cluster_centers_)

        # Теперь добавляем столбцы к оригинальным DataFrame
        population_original['distance_to_centroid'] = population['distance_to_centroid'] = distances
        population_original['cluster'] = population['cluster'] = clusters

        # Оптимизированная выборка без полной сортировки (используем heapq.nlargest)
        if sample_size < len(population_original):
            largest_distances_indices = heapq.nlargest(
                sample_size, range(len(distances)), distances.take)
            sample_processed = population_original.iloc[largest_distances_indices]
        else:
            logger.warning(
                f"Requested sample size {sample_size} exceeds available data size {len(population_original)}. Using full data.")
            sample_processed = population_original

        # Обновляем разметку выборки
        population_original['is_sample'] = population['is_sample'] = False
        population_original.loc[sample_processed.index, 'is_sample'] = True
        population.loc[sample_processed.index, 'is_sample'] = True

        method_description = (
            f"K-Means sampling with hyperparameter optimization using Optuna.\n"
            f"Sample size: {sample_size}.\n"
            f"Features: {features}.\n"
            f"Best K-Means parameters: {best_params}.\n"
            f"Creation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
            f"Trials: {len(study.trials)}.\n"
        )

        return population_original, population, sample_processed, method_description, study

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {ve}"

    except KeyError as ke:
        logger.error(f"KeyError: {ke}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"KeyError: {ke}"

    except Exception as e:
        logger.exception(f"Unexpected error in kmeans_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}"
