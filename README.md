# Reusable EDA Pipeline in Python

A small, reusable **Exploratory Data Analysis (EDA) class** built with pandas, NumPy, Matplotlib and Seaborn. Give it any DataFrame and it runs a standard first-look analysis: structure, data quality, summary statistics, distributions and correlations, in one chained call.

I built it while learning EDA, so I wouldn't have to rewrite the same checks for every new dataset.

## What it does

| Step | Method | What you get |
|---|---|---|
| 1 | `structural_audit()` | Number of rows and columns, data types, non-null counts |
| 2 | `quality_audit()` | Missing values (count and %, sorted highest first) and duplicate rows |
| 3 | `descriptive_stats()` | Summary statistics for numeric columns and frequency summaries for categorical columns |
| 4 | `plot_distributions()` | Histograms with KDE for numeric columns; bar charts of the top 10 categories for categorical columns |
| 5 | `plot_correlations()` | Correlation heatmap for numeric columns (upper triangle masked) |
| - | `run_all()` | Runs all five steps in order |

Every method returns `self`, so steps can be chained:

```python
pipeline.structural_audit().quality_audit().descriptive_stats()
```

## Design highlights

- **Object-oriented and reusable:** one `EDAPipeline` class works on any DataFrame.
- **Automatic column detection:** numeric and categorical columns are identified from their dtypes.
- **Safe by default:** it works on a copy of the data (`df.copy()`), so the original is never modified.
- **Method chaining:** the fluent interface keeps the workflow short and readable.
- **High-cardinality handling:** categorical plots show only the top 10 values so charts stay readable.
- **Clean layouts:** subplot grids adjust to the number of columns, and unused axes are removed.

## Getting started

### Requirements
```bash
pip install pandas numpy matplotlib seaborn
```

### Usage
```python
import pandas as pd
import numpy as np

# Load your own data
df = pd.read_csv("your_data.csv")

pipeline = EDAPipeline(df=df)
pipeline.run_all()
```

Run the notebook `Learning.ipynb` to see it in action. It includes a small demo dataset:

```python
data = pd.DataFrame({
    'Age': [25, 30, 47, 35, np.nan, 28, 52, 39, 41, 30],
    'Salary': [50000, 62000, 110000, 75000, 58000, 52000, np.nan, 82000, 91000, 62000],
    'Department': ['IT', 'HR', 'Exec', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'Exec', 'HR'],
    'Remote': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'No']
})
```

## Project structure

```
.
├── Learning.ipynb   # EDAPipeline class and demo
└── README.md
```

## Planned improvements

- Use the `target_col` parameter for target-based analysis (target distribution, feature vs. target plots)
- Outlier detection (IQR / Z-score)
- Skewness and kurtosis report
- Handle `category` and `string` dtypes as categorical columns
- Export a summary report to HTML or Markdown
- Package it as an installable module with unit tests

## Tech stack

Python · pandas · NumPy · Matplotlib · Seaborn · Jupyter Notebook

## Author

[Tajveer Singh](https://github.com/tajveer-31)
