import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Tuple
import logging
import datetime

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def lof_sampling(
    population_original: pd.DataFrame,
    population: pd.DataFrame,
    sample_size: int,
    features: List[str],  # Used for documentation only
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

        # Create a Local Outlier Factor model with default parameters
        lof = LocalOutlierFactor(novelty=True)
        lof.fit(population)

        # Get anomaly scores and predictions
        anomaly_scores = lof.decision_function(population)
        anomaly_predictions = (lof.predict(population) == -1).astype(int)

        # Add anomaly scores and labels to the DataFrames
        population_original['anomaly_score'] = population['anomaly_score'] = anomaly_scores
        population_original['is_anomaly'] = population['is_anomaly'] = anomaly_predictions
        population_original['is_sample'] = population['is_sample'] = False

        # Sort the DataFrames based on anomaly scores
        population_original = population_original.sort_values(
            by='anomaly_score', ascending=True)
        population = population.sort_values(by='anomaly_score', ascending=True)

        # Select the most anomalous samples
        sample_processed = population_original.head(sample_size)
        population_original.loc[sample_processed.index, 'is_sample'] = True
        population.loc[sample_processed.index, 'is_sample'] = True

        # Create method description
        method_description = (
            f"Method: Local Outlier Factor (LOF) with default parameters.\n"
            f"Sample size: {sample_size}.\n"
            f"Used features: {features}.\n"
            f"Sample creation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
        )

        return population_original, population, sample_processed, method_description
    except Exception as e:
        logger.exception(f"Error in lof_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Error: {e}"
