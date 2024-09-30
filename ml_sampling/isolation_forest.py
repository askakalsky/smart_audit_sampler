import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List, Tuple
import logging
import datetime

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def isolation_forest_sampling(
    population_original: pd.DataFrame,
    population: pd.DataFrame,
    sample_size: int,
    features: List[str],
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Performs anomaly sampling using Isolation Forest without hyperparameter optimization.

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

        # Create an Isolation Forest with default hyperparameters and random seed for reproducibility
        clf = IsolationForest(random_state=random_seed)

        # Train the model on the entire dataset
        clf.fit(population)

        # Get anomaly scores and predictions
        anomaly_scores = clf.decision_function(population)
        anomaly_predictions = (clf.predict(population) == -1).astype(int)

        # Add anomaly scores and labels to both DataFrames
        population_original['anomaly_score'] = population['anomaly_score'] = anomaly_scores
        population_original['is_anomaly'] = population['is_anomaly'] = anomaly_predictions
        population_original['is_sample'] = population['is_sample'] = 0

        # Sort the DataFrames based on anomaly scores
        population_original = population_original.sort_values(
            by='anomaly_score', ascending=True)
        population = population.sort_values(by='anomaly_score', ascending=True)

        # Select the most anomalous samples
        sample_processed = population_original.head(sample_size)
        population_original.loc[sample_processed.index, 'is_sample'] = 1
        population.loc[sample_processed.index, 'is_sample'] = 1

        # Get the total number of anomalies detected (where 'is_anomaly' == 1)
        total_anomalies = population_original['is_anomaly'].sum()

        # Create method description
        method_description = (
            f"**SAMPLING**\n"
            f"Sampling based on Isolation Forest with default hyperparameters.\n"
            f"Sample size: {sample_size}.\n"
            f"Requested features: {features}.\n"
            f"Total population size: {len(population_original)}.\n"
            f"Number of anomalies detected: {total_anomalies}.\n"
            f"Sample creation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Applied hyperparameters: n_estimators={clf.n_estimators}, max_samples={clf.max_samples}, contamination={clf.contamination}.\n"
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
        logger.exception(f"Unexpected error in kmeans_sampling: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}"
