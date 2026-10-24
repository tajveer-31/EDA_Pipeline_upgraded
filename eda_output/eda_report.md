# EDA Report

## Key Findings

1. Dataset has 10 rows and 4 columns (2 numerical, 2 categorical).
2. Dataset is small (< 100 rows): statistics and correlations may be unreliable.
3. Column 'Age' has 1 missing values (10.0%) - moderate - impute (fill) carefully.
4. Column 'Salary' has 1 missing values (10.0%) - moderate - impute (fill) carefully.
5. 1 duplicate row(s) found - verify if they are real or errors.
6. 'Age' and 'Salary' are highly correlated (r = 0.99) - possible multicollinearity, you may keep only one.
7. Strongest numeric relationship with 'Salary' is 'Age' (r = 0.99).
