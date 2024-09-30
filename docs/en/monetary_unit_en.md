# Script Description

This script is designed to perform sampling from a dataset where each record has a certain monetary value. The sampling is done in such a way that more expensive records have a higher chance of being selected. This is called **Monetary Unit Sampling** — a method of sampling that is based on monetary units.

## How it works:

### Data Preparation:
- First, a copy of the original dataset is created so that the original data remains unchanged.
- Each record is assigned a unique identifier to track selected records.
- All records are marked as not selected for the sample, which is necessary for further marking of selected records.

### Filtering:
- If the sampling should be based only on records with significant value, a threshold is set. Records with lower value are excluded from consideration. This allows you to focus on records that meet a certain level of significance.

### Grouping (if applicable):
- If it is necessary to divide the data into groups by some criterion (e.g., categories or types), the data is divided into groups. This is useful when it is necessary to ensure the representativeness of each group in the sample.

### Determining how many records to select from each group:
- For each group, it is calculated how many records should be selected. This calculation is based on the group's proportion relative to the total number of records in the data. For example, if one group makes up 30% of all the data, then approximately 30% of the sample should be taken from this group.
- After the calculation, rounding occurs. This can cause small deviations from the ideal proportions since the sample can only consist of whole records.

### Adjusting the sample size:
- If, after rounding, the total number of selected records does not match the requested sample size, the script adjusts this amount by adding or removing records from the largest groups. This is done to ensure that the final sample size exactly matches what was specified.

### Selecting records within each group:
- For each group, records are selected based on their value. The higher the value of the record, the more likely it is to be included in the sample.
- The selection is done as follows: the value of all records is summed, and random points are generated within this sum to select the records. Those records whose cumulative values "cover" these points are included in the sample.
- If duplicate selections occur (e.g., random points land on the same record), new points are generated until the required sample size is reached.

### Forming the final sample:
- The records selected from each group are combined into one final sample.
- The selected records are marked in the original dataset, so it is easy to see which records were selected.

### Report on execution:
- At the end, a description of the sampling process is generated, which includes general information about the population, sampling parameters, and results. This description provides details on the number of records in the sample, the parameters used, as well as information on when and how the sampling was performed.

---

## Important points:

- **Proportionality of the sample**: The script tries to maintain the proportions between groups so that each group is represented in the sample according to its share in the data.
- **Considering the value of records**: Within each group, the selection of records depends on their value. This means that more expensive records have a higher chance of being selected.
- **Adjustment after rounding**: After determining the number of records for each group, adjustments may be made to ensure that the final sample size exactly matches the requested size.

---

## When proportions may deviate:

- **Rounding**: Due to rounding, the number of records for each group may slightly differ from the ideally proportional distribution.
- **Adjustment**: If the total number of records after rounding is less or greater than requested, adjustments are made, which may slightly change the proportions.
- **Filtering**: If filtering by value is applied, groups with many low-value records may be underrepresented in the sample since they are partially excluded.
- **Small groups**: If a group is too small, its records may either be fully included in the sample or excluded, which also affects the adherence to proportions.

---

## In conclusion:

- This script helps to create a sample that reflects the distribution of monetary values in the original data and can also take into account different categories or types of records if necessary.
