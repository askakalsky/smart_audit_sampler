from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import skew
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame,
                    numerical_columns: List[str],
                    categorical_columns: List[str],
                    sample_fraction: float = 1.0,
                    skew_threshold: float = 0.75,
                    random_seed: Optional[int] = None
                    ) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Preprocesses a given DataFrame by applying one-hot encoding to categorical columns 
    and scaling numerical columns. Log transformation is applied to numerical columns 
    with a skewness higher than the skew_threshold. Other numerical columns are standardized.

    Args:
        df (pd.DataFrame): Input DataFrame containing raw data.
        numerical_columns (List[str]): List of column names that contain numerical data to be scaled.
        categorical_columns (List[str]): List of column names that contain categorical data for one-hot encoding.
        sample_fraction (float, optional): Fraction of the dataset to use (default is 1.0, meaning no subsampling).
        skew_threshold (float, optional): Threshold for skewness to apply log transformation (default is 0.75).
        random_seed (Optional[int], optional): Random seed for reproducibility in sampling (default is None).

    Returns:
        pd.DataFrame: Processed DataFrame with scaled numerical columns and one-hot encoded categorical columns.
        str: Method description detailing each preprocessing step.

    Raises:
        ValueError: If the input DataFrame or column lists are invalid.
    """
    try:
        # Make a copy of the DataFrame to avoid modifying the original
        df_processed = df.copy()

        method_description = ""

        # Apply sampling if necessary
        if sample_fraction < 1.0:
            df_processed = df_processed.sample(
                frac=sample_fraction, random_state=random_seed)
            method_description += (
                f"Sampling method: Random sampling.\n"
                f"Total population size: {len(df)}.\n"
                f"Sample size: {len(df_processed)}.\n"
                f"Sampling fraction: {sample_fraction}.\n"
                f"Sampling date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"Random seed: {random_seed}.\n"
            )
            logger.info(f"Sampled {sample_fraction * 100}% of the data.")

        # Capture the initial set of columns before one-hot encoding
        initial_columns = set(df_processed.columns)

        # Apply one-hot encoding to categorical columns
        if categorical_columns:
            df_processed = pd.get_dummies(
                df_processed, columns=categorical_columns)
            method_description += (
                f"One-hot encoding applied to categorical columns: {categorical_columns}.\n"
                f"Generated columns: {list(set(df_processed.columns) - initial_columns)}.\n"
                f"Transformation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            )
            logger.info(
                f"Applied one-hot encoding to categorical columns: {categorical_columns}")

            # Ensure the one-hot encoded columns are integers (0, 1)
            for col in df_processed.columns:
                if df_processed[col].dtype == 'bool':
                    df_processed[col] = df_processed[col].astype(int)

        # Identify the new columns generated by one-hot encoding
        new_categorical_columns = list(
            set(df_processed.columns) - initial_columns)

        # Process numerical columns
        skewed_features = []
        normal_features = []

        for col in numerical_columns:
            # Calculate skewness of the column
            skewness = skew(df_processed[col].dropna())
            logger.info(f"Skewness for {col}: {skewness}")

            if abs(skewness) > skew_threshold:
                # Apply log transformation to highly skewed data
                df_processed[col] = df_processed[col].apply(
                    lambda x: np.log1p(x) if x > 0 else x)
                skewed_features.append(col)
                method_description += (
                    f"Log transformation applied to skewed column: {col}.\n"
                    f"Skewness: {skewness}.\n"
                    f"Transformation threshold: {skew_threshold}.\n"
                    f"Transformation date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
                )
            else:
                # Standardize normally distributed data
                normal_features.append(col)
                method_description += (
                    f"Column considered normally distributed: {col}.\n"
                    f"Skewness: {skewness}.\n"
                    f"No transformation needed.\n"
                )

        # Apply Min-Max scaling for log-transformed (skewed) columns
        if skewed_features:
            scaler = MinMaxScaler()
            df_processed[skewed_features] = scaler.fit_transform(
                df_processed[skewed_features])
            method_description += (
                f"Min-Max scaling applied to skewed features: {skewed_features}.\n"
                f"Scaler used: MinMaxScaler.\n"
                f"Scaling date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            )
            logger.info(
                f"Min-Max scaling applied to skewed features: {skewed_features}")

        # Apply Standard scaling for normal-distributed columns
        if normal_features:
            scaler = StandardScaler()
            df_processed[normal_features] = scaler.fit_transform(
                df_processed[normal_features])
            method_description += (
                f"Standard scaling applied to normal features: {normal_features}.\n"
                f"Scaler used: StandardScaler.\n"
                f"Scaling date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            )
            logger.info(
                f"Standard scaling applied to normally distributed features: {normal_features}")

        # Combine numerical and new categorical columns for the final list of features
        available_features = numerical_columns + new_categorical_columns

        # Filter the DataFrame to include only the available features
        df_processed = df_processed[available_features]

        # Log the method description for reference
        logger.info(f"Preprocessing method description:\n{method_description}")

        # Return the processed DataFrame and method description
        return df_processed, method_description

    except Exception as e:
        # Log the exception and raise it as a ValueError
        logger.exception(f"Error in preprocess_data: {e}")
        raise ValueError(f"Data preprocessing failed: {e}")
