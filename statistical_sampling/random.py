import pandas as pd
from typing import Optional, Tuple
import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def random_sampling(
    population: pd.DataFrame,
    sample_size: int,
    random_seed: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Perform random sampling on a DataFrame based on indices.

    Parameters:
        population (pd.DataFrame): DataFrame containing the population.
        sample_size (int): The number of records to select in the sample.
        random_seed (Optional[int]): Random seed for reproducibility. Default is None.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
            - The updated DataFrame with a new 'is_sample' column indicating the sample selection.
            - A DataFrame containing the selected sample.
            - A string describing the sampling method and details.

    Raises:
        ValueError: If the sample size is greater than the population size.
        TypeError: If the population is not a valid DataFrame.
    """
    try:
        # Check if population is a DataFrame
        if not isinstance(population, pd.DataFrame):
            raise TypeError(
                f"Expected population to be a pd.DataFrame, got {type(population)} instead.")

        # Check if sample_size is valid
        if sample_size > len(population):
            raise ValueError(
                "Sample size cannot be greater than the population size.")

        # Add 'is_sample' column to flag the sampled data
        population['is_sample'] = 0

        # Perform the random sampling using the provided random seed
        sample = population.sample(n=sample_size, random_state=random_seed)

        # Mark the sampled rows in the original population DataFrame
        population.loc[sample.index, 'is_sample'] = 1
        sample['is_sample'] = 1

        # Create the sampling method description
        method_description = (
            f"**SAMPLING**\n"
            f"Sampling method: Random Sampling.\n"
            f"Total population size: {len(population)}.\n"
            f"Sample size: {sample_size}.\n"
            f"Number of selected records: {len(sample)}.\n"
            f"Sampling date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
        )

        return population, sample, method_description

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return None, None, f"ValueError: {e}"
    except TypeError as e:
        logger.error(f"TypeError: {e}")
        return None, None, f"TypeError: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, None, f"Unexpected error: {e}"
