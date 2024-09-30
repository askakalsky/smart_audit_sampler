# Script Description

This script performs **Systematic sampling** from a population, where records are selected at regular intervals after determining a random starting point.

## How it works:

### Data Preparation:
- A copy of the original dataset is created to prevent modifications to the original data.
- A new column `is_sample` is added, where all records are initially marked as not selected (`0`).

### Determining the sampling interval:
- The sampling interval is calculated based on the number of records in the population and the required sample size.
- A random starting point within the first interval is generated, which determines the starting record for the sampling process.

### Selecting records:
- Based on the interval, records are selected from the population at regular intervals. The number of selected records equals the sample size.
- If the sample size exceeds the total number of available records, the script raises an error.

### Forming the sample:
- The selected records are combined into a final sample.
- In the `is_sample` column, records that were selected are marked (`1` for selected records).

### Execution report:
- The script generates a summary of the sampling process, including the total population size, sample size, sampling interval, starting position, and the time of sampling.

---

## Important points:

- **Proportionality of the sample**: Records are selected evenly at regular intervals, ensuring that each record has a chance to be included in the sample.
- **Random start**: The sampling starts at a random point, ensuring randomness and minimizing systematic bias.
- **Interval-based selection**: After determining the starting point, records are selected at fixed intervals until the required sample size is reached.

---

## When deviations may occur:

- **Insufficient number of records**: If the sample size exceeds the number of records in the population, an error will be raised.
- **Small population**: If the population size is small, the sampling interval may be too large, which could affect the representativeness of the sample.

---

## In conclusion:

- This script ensures systematic selection of records from the population using regular intervals to evenly distribute the records in the sample.
