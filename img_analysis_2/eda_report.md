# EDA Report

## Key Findings

1. 'Rating' has 12 missing values (4.0%).
2. Dataset: 300 rows, 6 columns (4 numeric, 2 categorical).
3. 'Rating' has 1 outlier(s) (IQR rule).
4. 'Age' and 'Experience' are highly correlated (r=0.99).
5. 'Experience' and 'Salary' are highly correlated (r=0.96).
6. 'Age' and 'Salary' are highly correlated (r=0.95).
7. 'Salary' differs significantly across 'Department' groups (p=0.000); highest in 'IT', lowest in 'HR'.
8. 'Department' and 'Remote' are associated (chi-square p=0.024, Cramer's V=0.18).
9. 'Age' has high multicollinearity (VIF=49.7).
10. 'Experience' has high multicollinearity (VIF=60.1).
11. 'Salary' has high multicollinearity (VIF=12.0).
12. 2 principal component(s) capture 90% of the variance of 4 numeric columns.
