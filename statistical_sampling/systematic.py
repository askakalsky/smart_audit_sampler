import pandas as pd
import random
from typing import Optional, Tuple
import logging
import datetime

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def systematic_sampling(
    population: pd.DataFrame,
    sample_size: int,
    random_seed: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Perform systematic sampling from a population.

    Parameters:
        population (pd.DataFrame): DataFrame containing the population to sample from.
        sample_size (int): Number of records to select in the sample.
        random_seed (Optional[int], default=None): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, str]:
            - Updated population DataFrame with an 'is_sample' column.
            - Sample DataFrame containing the selected records.
            - Method description string summarizing the sampling procedure.

    Raises:
        ValueError: If the sample size is greater than the population size.
        TypeError: If the input data types are incorrect.
    """
    try:
        # Ensure population is a DataFrame
        if not isinstance(population, pd.DataFrame):
            raise TypeError(
                f"Expected population to be a pd.DataFrame, got {type(population)}.")

        # Check if the sample size is valid
        if sample_size > len(population):
            raise ValueError(
                "Sample size cannot be greater than the population size.")

        # Make a copy of the population to avoid modifying the original DataFrame
        population = population.copy()
        population['is_sample'] = 0

        # Calculate the sampling interval
        interval = len(population) // sample_size

        # Generate a random starting point within the first interval
        start = random.Random(random_seed).randint(0, interval - 1)

        # Generate systematic indices for sampling
        indices = range(start, len(population), interval)

        # Select the sample based on the systematic indices
        sample = population.iloc[list(indices)[:sample_size]]

        # Update 'is_sample' column in the population
        population.loc[sample.index, 'is_sample'] = 1
        sample['is_sample'] = 1

        # Create method description
        method_description = (
            f"Sampling method: Systematic sampling.\n"
            f"Total population size: {len(population)}.\n"
            f"Sample size: {sample_size}.\n"
            f"Sampling interval: {interval}.\n"
            f"Starting position: {start}.\n"
            f"Sampling date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
        )

        return population, sample, method_description

    except ValueError as e:
        # Log and return a ValueError if the sample size is invalid
        logger.error(f"ValueError: {e}")
        return None, None, f"ValueError: {e}"
    except TypeError as e:
        # Log and return a TypeError if the input data type is incorrect
        logger.error(f"TypeError: {e}")
        return None, None, f"TypeError: {e}"
    except Exception as e:
        # Log and return an error for any unexpected exceptions
        logger.error(f"Unexpected error: {e}")
        return None, None, f"Unexpected error: {e}"
