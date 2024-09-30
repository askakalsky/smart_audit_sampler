# Script Description

This script performs **Random sampling** from a population based on the indices of records. Random sampling allows selecting records without considering any structural features of the population, ensuring equal chances for every record.

## How it works:

### Data Preparation:
- The script checks if the population is a valid DataFrame and adds a new column `is_sample`, where all records are initially marked as not selected (`0`).

### Random Sampling:
- Sampling is performed randomly from the entire population, using a random seed (if provided) to ensure reproducibility.
- If the requested sample size exceeds the available records in the population, the script raises an error.

### Forming the sample:
- The randomly selected records are marked in the `is_sample` column with a value of `1`, while all other records remain marked as `0`.
- The selected records are compiled into the final sample, which can be used for further analysis.

### Execution report:
- A summary report of the sampling is generated, including the total population size, sample size, number of selected records, random seed (if used), and the time the sampling was performed.

---

## Important points:

- **Equal chance for each record**: Random sampling ensures that each record in the population has an equal chance of being selected.
- **Reproducibility**: If a random seed is used, the sampling can be reproduced with the same results.
- **Flexibility**: This method of sampling is suitable for general random selection of records, regardless of their structure or characteristics.

---

## When deviations may occur:

- **Incorrect sample size**: If the requested sample size exceeds the available records in the population, an error is raised.
- **Small samples**: For small samples, randomness might lead to non-representativeness if the population has a complex structure.

---

## In conclusion:

- This script provides a simple method for performing random sampling while ensuring reproducibility through the use of a random seed.
