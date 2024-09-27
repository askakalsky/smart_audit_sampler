import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def define_sample_sizes(n_samples: int) -> List[int]:
    """
    Defines sample sizes for silhouette score calculation based on data size.

    Parameters:
    -----------
    n_samples : int
        Number of samples in the dataset.

    Returns:
    --------
    List[int]
        List of sample sizes to use for subsampling.
    """
    if n_samples < 1000:
        return [n_samples]
    elif n_samples < 10000:
        sample_size = min(max(int(n_samples * 0.1), 1000), 5000)
        return [sample_size] * 3
    elif n_samples < 100000:
        sample_size = min(max(int(n_samples * 0.1), 1000), 5000)
        return [sample_size] * 3
    else:
        return [5000] * 3


def compute_silhouette_scores(clusterer, data: pd.DataFrame, sample_sizes: List[int], available_features: List[str] = None, random_seed: int = None) -> float:
    """
    Computes the silhouette scores for the given clusterer and data using subsampling.

    Parameters:
    -----------
    clusterer : object
        Clustering object that implements fit_predict or predict method.
    data : pd.DataFrame
        Data to be clustered.
    sample_sizes : List[int]
        List of sample sizes for subsampling.
    available_features : List[str], optional
        List of feature names to use for clustering. If None, uses all features.
    random_seed : int, optional
        Seed for random number generation.

    Returns:
    --------
    float
        Average silhouette score across subsamples.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(data)
    silhouette_scores = []

    for size in sample_sizes:
        try:
            sample_indices = rng.choice(n_samples, size=size, replace=False)
            sample = data.iloc[sample_indices]
            if available_features is not None:
                labels = clusterer.predict(data[available_features])
                score = silhouette_score(
                    sample[available_features], labels[sample_indices])
            else:
                labels = clusterer.fit_predict(sample)
                unique_labels = set(labels)
                unique_labels.discard(-1)
                if len(unique_labels) > 1:
                    score = silhouette_score(sample, labels)
                else:
                    score = float('-inf')
            silhouette_scores.append(score)
        except ValueError:
            silhouette_scores.append(float('-inf'))

    return np.mean(silhouette_scores)


def calculate_silhouette_hdbscan(clusterer: HDBSCAN, data: pd.DataFrame) -> float:
    """
    Calculates the silhouette score for a given HDBSCAN clusterer using subsampling.

    Parameters:
    -----------
    clusterer : HDBSCAN
        HDBSCAN clustering object.
    data : pd.DataFrame
        Data to be clustered.

    Returns:
    --------
    float
        Average silhouette score across multiple subsamples.
    """
    sample_sizes = define_sample_sizes(len(data))
    return compute_silhouette_scores(clusterer, data, sample_sizes)


def calculate_silhouette(clf, data: pd.DataFrame, available_features: List[str], random_seed: int) -> float:
    """
    Calculates the silhouette score for a given clustering model using subsampling.

    Parameters:
    -----------
    clf : object
        Clustering model that implements fit_predict or predict method.
    data : pd.DataFrame
        Data to be clustered.
    available_features : List[str]
        List of feature names to use for clustering.
    random_seed : int
        Seed for random number generation.

    Returns:
    --------
    float
        Average silhouette score across multiple subsamples.
    """
    sample_sizes = define_sample_sizes(len(data))
    return compute_silhouette_scores(clf, data, sample_sizes, available_features, random_seed)
