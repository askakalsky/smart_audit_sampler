import pandas as pd
import optuna
from sklearn.ensemble import IsolationForest
from typing import List, Tuple
from scoring_methods.silhouette import calculate_silhouette
from sklearn.exceptions import NotFittedError
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, optuna.Study]:
    """
    Isolation Forest-based anomaly sampling with hyperparameter optimization using Optuna.

    Parameters:
    -----------
    population_original : pd.DataFrame
        Original dataset to apply anomaly detection on.
    population : pd.DataFrame
        Preprocessed dataset with selected features for analysis.
    sample_size : int
        Number of samples to retrieve based on anomaly score.
    features : List[str]
        List of features used for anomaly detection.
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
        - study : optuna.Study
          Optuna study object containing hyperparameter optimization results.
    """
    try:
        # Make copies to avoid modifying the original DataFrames
        population_original = population_original.copy()
        population = population.copy()

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_samples': trial.suggest_categorical('max_samples', ['auto', 0.1, 0.3, 0.5, 0.7, 0.9]),
                'max_features': trial.suggest_int('max_features', 1, len(features)),
            }
            clf = IsolationForest(random_state=random_seed, **params)
            clf.fit(population[features])
            return calculate_silhouette(clf, population, features, random_seed)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=True)

        best_params = study.best_params
        best_clf = IsolationForest(random_state=random_seed, **best_params)
        best_clf.fit(population[features])

        population_original['anomaly_score'] = population['anomaly_score'] = best_clf.decision_function(
            population[features])
        population_original['is_anomaly'] = population['is_anomaly'] = (
            best_clf.predict(population[features]) == -1).astype(int)
        population_original['is_sample'] = population['is_sample'] = False

        population_original = population_original.sort_values(
            by='anomaly_score', ascending=True)
        population = population.sort_values(by='anomaly_score', ascending=True)

        sample_processed = population_original.head(sample_size)
        population_original.loc[sample_processed.index, 'is_sample'] = True
        population.loc[sample_processed.index, 'is_sample'] = True

        method_description = (
            f"Sampling based on Isolation Forest with Optuna hyperparameter tuning.\n"
            f"Sample size: {sample_size}.\n"
            f"Requested features: {features}.\n"
            f"Number of anomalies detected: {len(sample_processed)}.\n"
            f"Used parameters: {best_params}.\n"
            f"Sample creation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
        )

        return population_original, population, sample_processed, method_description, study
    except ValueError as e:
        logger.error(f"ValueError: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"ValueError: {e}", None
    except NotFittedError as e:
        logger.error(f"NotFittedError: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"NotFittedError: {e}", None
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {e}", None
