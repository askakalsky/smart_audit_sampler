import pandas as pd
from typing import Optional, Tuple
import logging
import datetime

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def stratified_sampling(
    population: pd.DataFrame,
    sample_size: int,
    strata_column: str,
    random_seed: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Perform stratified sampling from a population.

    Parameters:
        population (pd.DataFrame): DataFrame containing the population to sample from.
        sample_size (int): Number of records to select in the sample.
        strata_column (str): The column representing the strata to group by.
        random_seed (Optional[int], default=None): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, str]:
            - Updated population DataFrame with an 'is_sample' column.
            - Sample DataFrame containing the selected records.
            - Method description string summarizing the sampling procedure.

    Raises:
        KeyError: If the strata column is not found in the DataFrame.
        ValueError: If the sample size is invalid.
        TypeError: If the input data types are incorrect.
    """
    try:
        # Ensure population is a DataFrame
        if not isinstance(population, pd.DataFrame):
            raise TypeError(
                f"Expected population to be a pd.DataFrame, got {type(population)}.")

        # Ensure the strata column exists in the population
        if strata_column not in population.columns:
            raise KeyError(
                f"Column '{strata_column}' not found in population DataFrame.")

        # Make a copy of the population to avoid modifying the original DataFrame
        population = population.copy()
        population['is_sample'] = 0

        # Calculate counts and proportions for each stratum
        strata_counts = population[strata_column].value_counts()
        proportions = strata_counts / len(population)

        # Calculate initial sample sizes (non-rounded)
        initial_sample_sizes = proportions * sample_size

        # Extract integer parts
        integer_parts = initial_sample_sizes.astype(int)

        # Extract fractional parts
        fractional_parts = initial_sample_sizes - integer_parts

        # Compute the total sample size from integer parts
        current_total = integer_parts.sum()

        # Calculate the difference between desired and current total sample sizes
        difference = sample_size - current_total

        # Adjust sample sizes to match the desired sample_size
        if difference > 0:
            # Allocate extra samples to strata with largest fractional parts
            indices = fractional_parts.sort_values(ascending=False).index
            for i in range(difference):
                integer_parts[indices[i]] += 1
        elif difference < 0:
            # Reduce samples from strata with smallest fractional parts
            indices = fractional_parts.sort_values().index
            for i in range(-difference):
                # Ensure we don't assign negative sample sizes
                if integer_parts[indices[i]] > 0:
                    integer_parts[indices[i]] -= 1

        # Now sample from each stratum based on adjusted sample sizes
        sample = pd.DataFrame()
        strata_groups = population.groupby(strata_column)

        for stratum in integer_parts.index:
            stratum_sample_size = integer_parts[stratum]
            if stratum_sample_size > 0:
                stratum_group = strata_groups.get_group(stratum)
                stratum_sample = stratum_group.sample(
                    n=min(stratum_sample_size, len(stratum_group)),
                    random_state=random_seed
                )
                sample = pd.concat([sample, stratum_sample])

        # Update 'is_sample' column in the population
        population.loc[sample.index, 'is_sample'] = 1
        sample['is_sample'] = 1

        # Create method description
        method_description = (
            f"Sampling method: Stratified sampling.\n"
            f"Total population size: {len(population)}.\n"
            f"Sample size: {sample_size}.\n"
            f"Stratification column: {strata_column}.\n"
            f"Number of selected records: {len(sample)}.\n"
            f"Sampling date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
        )

        return population, sample, method_description

    except KeyError as e:
        # Log and return a KeyError if the strata column is missing
        logger.error(f"KeyError: {e}")
        return None, None, f"KeyError: {e}"
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
