import pandas as pd
import random
import uuid
from typing import Optional, Tuple
import logging
import datetime

# Setting up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def monetary_unit_sampling(
    population: pd.DataFrame,
    sample_size: int,
    value_column: str,
    threshold: Optional[float] = None,
    strata_column: Optional[str] = None,
    random_seed: int = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Perform Monetary Unit Sampling (MUS) on a population.

    Parameters:
    -----------
    population : pd.DataFrame
        DataFrame containing the population to sample from.
    sample_size : int
        Number of records to select in the sample.
    value_column : str
        The column representing the monetary value to base the sampling on.
    threshold : Optional[float], optional
        Minimum value in the `value_column` to include in the sample. Default is None.
    strata_column : Optional[str], optional
        Optional column for stratified sampling. Default is None.
    random_seed : Optional[int], optional
        Random seed for reproducibility. Default is None.

    Returns:
    --------
    Tuple containing:
        - population_updated : pd.DataFrame
          Updated population DataFrame with an "is_sample" column indicating sampled records.
        - sample : pd.DataFrame
          DataFrame containing the selected records.
        - method_description : str
          String summarizing the sampling procedure.

    Raises:
    -------
    KeyError:
        If required columns are missing in the DataFrame.
    ValueError:
        If no records remain after applying the threshold.
    TypeError:
        If the input data types are incorrect.
    """
    try:
        # Validate that the population is a DataFrame
        if not isinstance(population, pd.DataFrame):
            raise TypeError(
                f"Expected population to be a pd.DataFrame, got {type(population)} instead.")

        # Make a copy of the population to avoid modifying the original DataFrame
        population = population.copy()

        # Check if the specified value_column exists in the population DataFrame
        if value_column not in population.columns:
            raise KeyError(
                f"Column '{value_column}' not found in population DataFrame.")

        # Generate a unique identifier for each row in the population using uuid4
        # This ensures that each record has a distinct identifier
        population['unique_id'] = [uuid.uuid4()
                                   for _ in range(len(population))]
        # Initialize the 'is_sample' column to 0 for all rows (not part of the sample)
        population['is_sample'] = 0

        # Get the total size of the population and the sum of the value column
        total_population_size = len(population)
        total_population_value = population[value_column].sum()

        # Filter the population based on the threshold if provided
        # If no threshold is given, use the entire population
        if threshold is not None:
            population_filtered = population[population[value_column] >= threshold].copy(
            )
            post_filtered_population_size = len(population_filtered)
            if post_filtered_population_size == 0:
                raise ValueError(
                    "No records remain after applying the threshold.")
        else:
            population_filtered = population.copy()

        # Reset the index of the filtered population for consistent indexing
        population_filtered = population_filtered.reset_index(drop=True)

        def sample_stratum(group, stratum_sample_size, random_seed):
            """
            Sample a stratum based on the cumulative monetary value.

            Parameters:
            -----------
            group : pd.DataFrame
                The group or stratum of the population to sample from.
            stratum_sample_size : int
                The number of records to sample from this stratum.
            random_seed : Optional[int]
                Seed for reproducibility of random sampling.

            Returns:
            --------
            pd.DataFrame
                A DataFrame containing the sampled records from the stratum.
            """
            # If the stratum size is less than or equal to the required sample size, return the whole group
            if len(group) <= stratum_sample_size:
                return group

            # Calculate the cumulative value within the stratum for sampling based on monetary units
            group['CumulativeValue'] = group[value_column].cumsum()
            total_value = group['CumulativeValue'].iloc[-1]

            # If the total value is zero, perform random sampling
            if total_value == 0:
                return group.sample(n=stratum_sample_size, random_state=random_seed)

            # Define sampling intervals based on cumulative value
            interval = total_value / stratum_sample_size
            random_gen = random.Random(random_seed)
            # Generate random selection points within each interval
            selection_points = [
                random_gen.uniform(0, interval) + i * interval
                for i in range(stratum_sample_size)
            ]

            # Find the indices of the records where the cumulative value crosses the sampling points
            sample_indices = group['CumulativeValue'].searchsorted(
                selection_points)
            # Ensure uniqueness of the sample indices
            sample_indices = pd.Series(
                sample_indices).drop_duplicates().tolist()

            # If the number of unique indices is less than required, add more random points
            while len(sample_indices) < stratum_sample_size:
                new_point = random_gen.uniform(0, total_value)
                new_index = group['CumulativeValue'].searchsorted(new_point)
                if new_index not in sample_indices:
                    sample_indices.append(new_index)

            # Return the selected rows from the group
            return group.iloc[sample_indices[:stratum_sample_size]]

        # If strata_column is specified, perform stratified sampling
        if strata_column:
            # Check if the strata_column exists
            if strata_column not in population_filtered.columns:
                raise KeyError(
                    f"Column '{strata_column}' not found in population DataFrame.")

            # Group the population by strata and initialize an empty DataFrame for the sample
            strata = population_filtered.groupby(strata_column)
            sample = pd.DataFrame()

            # Calculate the size of each stratum based on the population proportion
            stratum_sizes = (strata.size() / len(population_filtered)
                             * sample_size).round().astype(int)
            # Adjust the sample size if rounding errors occur
            size_diff = sample_size - stratum_sizes.sum()
            largest_strata = stratum_sizes.nlargest(abs(size_diff)).index

            # Adjust the sizes of the largest strata to correct any rounding differences
            for stratum in largest_strata:
                if stratum_sizes[stratum] > 0:
                    stratum_sizes[stratum] += 1 if size_diff > 0 else -1
                    size_diff += -1 if size_diff > 0 else 1
                if size_diff == 0:
                    break

            # Sample each stratum individually
            for stratum, group in strata:
                stratum_sample_size = stratum_sizes[stratum]
                if stratum_sample_size == 0:
                    continue
                # Use the sample_stratum function to select records from each stratum
                stratum_sample = sample_stratum(
                    group, stratum_sample_size, random_seed)
                # Append the sampled records to the final sample DataFrame
                sample = pd.concat([sample, stratum_sample])
        else:
            # If no stratification is specified, sample from the entire filtered population
            sample = sample_stratum(
                population_filtered, sample_size, random_seed)

        # Drop the cumulative value column from the sample, if it exists
        sample = sample.drop(columns=['CumulativeValue'], errors='ignore')
        # Mark the sampled records in the 'is_sample' column
        sample['is_sample'] = 1

        # Identify the unique IDs of the sampled records
        sample_ids = sample['unique_id'].tolist()
        population.loc[population['unique_id'].isin(
            sample_ids), 'is_sample'] = 1

        # Drop the unique ID column from both the population and the sample to clean up
        population = population.drop('unique_id', axis=1)
        sample = sample.drop('unique_id', axis=1)

        # Start forming the method description
        method_description = (
            f"Sampling method: Monetary Unit Sampling.\n"
            f"Total population size: {total_population_size}.\n"
            f"Total population value: {total_population_value}.\n"
            f"Sample size: {sample_size}.\n"
            f"Value column: {value_column}.\n"
            f"Number of selected records: {len(sample)}.\n"
        )

        # Include post-filtered population size only if a threshold is applied
        if threshold is not None:
            method_description += (
                f"Threshold: {threshold}.\n"
                f"Post-filtered population size: {post_filtered_population_size}.\n"
            )

        # Include stratification details if applicable
        if strata_column:
            method_description += f"Sampling strata: {strata_column}.\n"

        # Include timestamp and random seed information in the method description
        method_description += (
            f"Sampling date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
            f"Random seed: {random_seed}.\n"
            f"Cumulative sampling method: {'Yes' if strata_column else 'No'}.\n"
        )

        # Return the updated population, sample, and method description
        return population, sample, method_description

    except KeyError as e:
        # Log and return an error if a KeyError occurs
        logger.error(f"KeyError: {e}")
        return None, None, f"KeyError: {e}"
    except ValueError as e:
        # Log and return an error if a ValueError occurs
        logger.error(f"ValueError: {e}")
        return None, None, f"ValueError: {e}"
    except TypeError as e:
        # Log and return an error if a TypeError occurs
        logger.error(f"TypeError: {e}")
        return None, None, f"TypeError: {e}"
    except Exception as e:
        # Log and return an error if an unexpected exception occurs
        logger.error(f"Unexpected error: {e}")
        return None, None, f"Unexpected error: {e}"
