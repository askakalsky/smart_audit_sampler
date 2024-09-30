import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Tuple
import logging
import datetime
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def lof_sampling(
    population_original: pd.DataFrame,
    population: pd.DataFrame,
    sample_size: int,
    features: List[str],
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Performs anomaly sampling using Local Outlier Factor (LOF) with default parameters.

    Parameters:
    -----------
    population_original : pd.DataFrame
        Original dataset to apply anomaly detection on.
    population : pd.DataFrame
        Preprocessed dataset with selected features for analysis.
    sample_size : int
        Number of samples to retrieve based on anomaly score.
    features : List[str]
        List of features used for anomaly detection (used for documentation only).
    random_seed : int
        Seed for random number generation.

    Returns:
    --------
    Tuple containing:
        - population_original : pd.DataFrame
          Original dataset with anomaly scores, labels, and sampling flags.
        - population : pd.DataFrame
          Preprocessed dataset with anomaly scores, labels, and sampling flags.
        - sample_processed : pd.DataFrame
          Subset of the dataset containing detected anomalies.
        - method_description : str
          Description of the sampling process and parameters used.
    """
    try:
        # Make copies to avoid modifying the original DataFrames
        population_original = population_original.copy()
        population = population.copy()

        # Initialize random number generator for reproducibility
        rng = np.random.default_rng(random_seed)

        # Create a Local Outlier Factor model with default parameters
        # Turn novelty=False to apply LOF to the entire dataset
        lof = LocalOutlierFactor(novelty=False)
        anomaly_predictions = lof.fit_predict(population)

        # Get anomaly scores (negative_outlier_factor_)
        anomaly_scores = -lof.negative_outlier_factor_

        # Add anomaly scores and labels to the DataFrames
        population_original['anomaly_score'] = population['anomaly_score'] = anomaly_scores
        population_original['is_anomaly'] = population['is_anomaly'] = (
            anomaly_predictions == -1).astype(int)
        population_original['is_sample'] = population['is_sample'] = 0

        # Sort the DataFrames based on anomaly scores (highest scores first, as they indicate more abnormal)
        population_original = population_original.sort_values(
            by='anomaly_score', ascending=True)
        population = population.sort_values(
            by='anomaly_score', ascending=True)

        # Select top `sample_size` most anomalous samples based on the score
        sample_processed = population_original.head(sample_size)

        # Mark the sampled data
        population_original.loc[sample_processed.index, 'is_sample'] = 1
        population.loc[sample_processed.index, 'is_sample'] = 1

        # Total population size and number of anomalies
        population_size = len(population_original)
        total_anomalies = (population_original['is_anomaly'] == 1).sum()

        # Create method description
        method_description = (
            f"Method: Local Outlier Factor (LOF) with default parameters.\n"
            f"Sample size: {sample_size}.\n"
            f"Total population size: {population_size}.\n"
            f"Number of detected anomalies: {total_anomalies}.\n"
            f"Used features: {features}.\n"
            f"Sample creation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
        )

        return population_original, population, sample_processed, method_description

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {ve}"

    except KeyError as ke:
        logger.error(f"KeyError: {ke}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"KeyError: {ke}"

    except Exception as e:
        logger.exception(f"Unexpected error in lof_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}"
