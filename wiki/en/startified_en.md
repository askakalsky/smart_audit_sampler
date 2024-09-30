# Script Description

This script implements a **Stratified sampling** technique, where records are selected from a dataset based on their grouping within strata (or categories). The goal is to ensure that the sample reflects the proportions of these strata in the overall population.

## How it works:

### Data Preparation:
- A copy of the original dataset is created to prevent any modifications to the original data.
- A new column `is_sample` is added to the dataset, initially marking all records as not selected (`0`).

### Grouping by strata:
- The population is divided into strata based on the values in the specified column (e.g., category or type).
- Each group (stratum) is processed individually to ensure that the sample reflects the proportion of each group in the total population.

### Determining how many records to select from each stratum:
- For each group (stratum), the number of records to be selected is calculated based on the proportion of the stratum in the population. For example, if a stratum represents 20% of the population, 20% of the total sample size will be selected from that stratum.
- The calculated sample size is rounded, which may cause small deviations from the ideal proportions.

### Sampling from each stratum:
- Records are randomly selected from each stratum, ensuring that the number of selected records matches the calculated sample size for that group.
- If a stratum contains fewer records than the calculated sample size, all available records from that stratum are selected.

### Adjusting the sample size:
- If the total sample size is smaller than requested after initial selection, additional records are randomly sampled from the largest stratum until the sample size matches the requested number.
- The proportions are adjusted accordingly during this process.

### Finalizing the sample:
- The records selected from all strata are combined into a single final sample.
- The `is_sample` column in the original population is updated to mark which records have been selected (`1` for selected records).

### Report on execution:
- The script generates a summary of the sampling process, including information such as the total population size, sample size, the column used for stratification, and the timestamp of when the sampling was performed.

---

## Important points:

- **Proportionality of the sample**: The script ensures that each group (stratum) is represented in the sample in proportion to its share in the population.
- **Random selection within strata**: For each group, the script randomly selects records, ensuring that the sample reflects the original distribution.
- **Adjustment after rounding**: If the total number of records in the sample differs from the requested size due to rounding, additional records are selected from the largest group to ensure that the sample size is correct.

---

## When proportions may deviate:

- **Rounding**: Due to rounding, the number of records selected from each group may slightly differ from the ideal proportional distribution.
- **Adjustment**: If the total sample size is smaller or larger than requested, the largest group may provide additional records, which can affect the proportions.
- **Small groups**: If a group contains fewer records than the calculated sample size, all available records from that group are included, which may slightly alter the proportions.

---

## In conclusion:

- This script ensures that the sample reflects the distribution of records across different strata in the population. The sample size is adjusted as needed to match the requested size while maintaining the proportionality of the original data as much as possible.
