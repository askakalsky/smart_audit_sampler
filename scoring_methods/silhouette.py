import numpy as np
import pandas as pd
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


def compute_silhouette_scores(clusterer, data: pd.DataFrame, sample_sizes: List[int], random_seed: int = None) -> float:
    """
    Computes the silhouette scores for the given clusterer or anomaly detector and data using subsampling.

    Parameters:
    -----------
    clusterer : object
        Clustering or anomaly detection object that implements fit_predict or predict method.
    data : pd.DataFrame
        Data to be clustered or classified for outliers.
    sample_sizes : List[int]
        List of sample sizes for subsampling.
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

            # Check if clusterer has fit_predict or predict
            if hasattr(clusterer, 'fit_predict'):
                labels = clusterer.fit_predict(sample)
            elif hasattr(clusterer, 'predict'):
                labels = clusterer.predict(sample)
            else:
                raise ValueError(
                    f"Clusterer does not support 'fit_predict' or 'predict': {type(clusterer)}")

            unique_labels = set(labels)
            unique_labels.discard(-1)  # For HDBSCAN or outlier labels
            if len(unique_labels) > 1:
                score = silhouette_score(sample, labels)
            else:
                score = float('-inf')
            silhouette_scores.append(score)
        except ValueError:
            silhouette_scores.append(float('-inf'))

    return np.mean(silhouette_scores)


def calculate_silhouette(clusterer, data: pd.DataFrame, random_seed: int = None) -> float:
    """
    Calculates the silhouette score for a given clustering or anomaly detection model using subsampling.

    Parameters:
    -----------
    clusterer : object
        Clustering or anomaly detection model that implements fit_predict or predict method.
    data : pd.DataFrame
        Data to be clustered or classified for outliers.
    random_seed : int, optional
        Seed for random number generation.

    Returns:
    --------
    float
        Average silhouette score across multiple subsamples.
    """
    sample_sizes = define_sample_sizes(len(data))
    return compute_silhouette_scores(clusterer, data, sample_sizes, random_seed)
