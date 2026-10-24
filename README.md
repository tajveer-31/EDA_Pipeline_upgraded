# Reusable EDA Pipeline

A small, reusable Python toolkit that automates **Exploratory Data Analysis (EDA)** on any pandas DataFrame.
Pass in a dataset and get a data audit, statistics, plots, statistical tests and a plain-English findings report, with only a few lines of code.

The project has **two pipelines** that work independently:

| Pipeline | File | Class | Best for |
|---|---|---|---|
| **Basic pipeline** | `eda_basic_pipeline.py` | `EDAPipeline` | Auditing a new dataset: structure, data quality, outliers, target analysis |
| **Multi-level analysis pipeline** | `eda_multi_analysis_pipeline.py` | `EDAToolkit` | Deeper analysis: univariate, bivariate and multivariate |


## Why this project

Every data project starts with the same checks: *How big is the data? Is anything missing? Are there duplicates or outliers? Which columns are related?*
Writing this code again for every dataset wastes time, so this project packages the routine into reusable classes.

- **Reusable**: works on any DataFrame; nothing is hard-coded to one dataset.
- **Modular**: every step is its own method, so you can run all of them or only the ones you need.
- **Readable**: simple code, with comments explaining *why* each step is done.
- **Self-documenting**: findings are collected automatically and written to a report.

---

## Project structure

```
.
├── eda_basic_pipeline.py            # Pipeline 1: EDAPipeline (audit-focused)
├── eda_multi_analysis_pipeline.py   # Pipeline 2: EDAToolkit (uni / bi / multivariate)
├── Learning_Basic.ipynb             # notebook for Pipeline 1
├── Learning_multi.ipynb             # notebook for Pipeline 2
---

## Installation

Requires **Python 3.9+**.

```bash
pip install pandas numpy matplotlib seaborn
pip install scipy        # optional: enables statistical tests (ANOVA, chi-square) in EDAToolkit
```

`scipy` is optional. Without it the pipeline still runs; only the p-values are skipped.

---

## Quick start

**Pipeline 1: basic audit**
```python
from eda_basic_pipeline import EDAPipeline

pipeline = EDAPipeline(df, target_col="Salary", save_dir="eda_output")
pipeline.run_all()
```

**Pipeline 2: univariate + bivariate + multivariate**
```python
from eda_multi_analysis_pipeline import EDAToolkit

eda = EDAToolkit(df, target_col="Salary", save_dir="eda_toolkit_output")
eda.run_all()
```

**Run only the steps you need** (methods can be chained):
```python
EDAPipeline(df).structural_audit().quality_audit().outlier_analysis()
EDAToolkit(df).univariate().bivariate()
```

The demo notebooks `Learning_Basic.ipynb` and `Learning_multi.ipynb` show both pipelines running end to end.

---

## Pipeline 1: Basic pipeline (`EDAPipeline`)

Runs seven steps in order (`run_all()`):

| # | Method | What it does |
|---|---|---|
| 1 | `structural_audit()` | Shape, column types (numeric / categorical), memory usage, data types and first rows |
| 2 | `quality_audit()` | Missing values (count and %), duplicate rows, constant columns, high-cardinality columns |
| 3 | `descriptive_stats()` | Summary statistics, skewness, kurtosis, category percentages, imbalance detection |
| 4 | `outlier_analysis()` | Outliers per numeric column using the **IQR rule** |
| 5 | `plot_missing()`, `plot_distributions()`, `plot_correlations()` | Missing-value map, histograms + boxplots, category bar charts, correlation heatmap with highly correlated pairs flagged |
| 6 | `target_analysis()` | If `target_col` is set: correlation of features with the target, and the target's average per category |
| 7 | `generate_report()` | Prints all insights in plain English and saves them to `eda_report.md` |

---

## Pipeline 2: Multi-level analysis (`EDAToolkit`)

Organised by how many columns are examined at once (`run_all()` runs everything):

### 0. `overview()`
Shape, column types, missing values and duplicates. Always the first look at the data.

### 1. `univariate()`: one column at a time
- Numeric: summary statistics, skewness, kurtosis, outlier count; histogram with density curve and boxplot
- Categorical: percentage of each category, imbalance warning; count charts

### 2. `bivariate()`: two columns together
| Pair type | Methods used |
|---|---|
| Numeric vs numeric | Pearson and Spearman correlation, scatter plots with a trend line for the strongest pairs |
| Category vs numeric | Group averages, boxplots, **ANOVA** p-value (are the groups really different?) |
| Category vs category | Crosstab heatmaps, **chi-square** test and **Cramér's V** (strength of association) |

### 3. `multivariate()`: many columns together
- **Correlation heatmap** of all numeric columns
- **VIF (Variance Inflation Factor)**: detects multicollinearity (VIF > 5 means a column is largely explained by the others)
- **PCA** (implemented with NumPy): explained-variance table, scree plot and a 2D projection
- **Pairplot** of the main numeric columns, coloured by a small category column
- **Pivot heatmap**: average of a numeric column across two categories

### `report()`
Prints all collected findings and saves them as `eda_report.md`.

---

## Parameters

### `EDAPipeline(df, target_col, save_dir, high_corr_threshold, high_cardinality_threshold)`

| Parameter | Default | Meaning |
|---|---|---|
| `df` | required | DataFrame to analyse (a copy is made; your original is never modified) |
| `target_col` | `None` | Column to focus on (enables `target_analysis`) |
| `save_dir` | `None` | Folder to save plots and the report |
| `high_corr_threshold` | `0.8` | \|correlation\| above this is flagged as highly correlated |
| `high_cardinality_threshold` | `50` | Categorical columns with more unique values than this are flagged |

### `EDAToolkit(df, target_col, save_dir, max_categories, corr_threshold, max_pairplot_cols)`

| Parameter | Default | Meaning |
|---|---|---|
| `df` | required | DataFrame to analyse (copied internally) |
| `target_col` | `None` | Column used to focus comparisons and colour plots |
| `save_dir` | `None` | Folder to save plots and the report |
| `max_categories` | `15` | Categorical columns with more unique values are skipped (likely IDs or free text) |
| `corr_threshold` | `0.8` | \|correlation\| above this is flagged |
| `max_pairplot_cols` | `5` | Limits the pairplot to this many numeric columns (it gets slow with more) |



## Sample findings

Running `EDAToolkit` on the 300-row synthetic employee dataset from `Learning_multi.ipynb` produces findings such as:

```
1. 'Rating' has 12 missing values (4.0%).
2. 'Rating' has 1 outlier(s) (IQR rule).
3. 'Age' and 'Experience' are highly correlated (r=0.99).
4. 'Experience' and 'Salary' are highly correlated (r=0.96).
5. 'Salary' differs significantly across 'Department' groups (p=0.000); highest in 'IT', lowest in 'HR'.
6. 'Department' and 'Remote' are associated (chi-square p=0.024, Cramer's V=0.18).
7. 'Age' has high multicollinearity (VIF=49.7).
8. 2 principal component(s) capture 90% of the variance of 4 numeric columns.
```

> The dataset in the demo notebooks is **synthetic**, created only to show the pipeline. Findings describe that sample, not real-world data.

Running `EDAPipeline` on the 10-row sample dataset from `Learning_Basic.ipynb` flags the missing values in `Age` and `Salary`, the duplicate row, and the near-perfect Age/Salary correlation.

---

## Design decisions

- **Method chaining**: every step returns `self`, so steps combine in one line.
- **Non-destructive**: the DataFrame is copied on creation; the original is never changed.
- **Plain-English insights**: each step stores findings in `self.findings`, so the report is generated automatically.
- **Fail early**: an empty DataFrame or a `target_col` that does not exist raises a clear error.
- **Safe on messy data**: high-cardinality columns (IDs, names) are skipped or flagged instead of producing unreadable plots.
- **Minimal dependencies**: PCA and VIF are implemented with NumPy; `scipy` is optional.

---

## Limitations

- **Small samples**: statistics and correlations are unreliable on very small datasets (the basic pipeline warns below 100 rows).
- **Linear methods**: Pearson correlation, VIF and PCA capture linear relationships only.
- **No data cleaning**: the pipelines only *report* problems (missing values, duplicates, outliers); they do not fix them.
- **Rows with missing values** are dropped for correlation, VIF and PCA calculations.
- **Very wide datasets** (many columns) produce many plots; use `max_categories` and `max_pairplot_cols` to keep output manageable.

---

## Possible improvements

- Automatic data-cleaning suggestions or imputation helpers
- Export the full report as HTML or PDF with embedded plots
- Datetime column analysis (trends, seasonality)
- Mutual information for non-linear relationships
- Automated tests (`pytest`) and packaging as a pip-installable library

---

## Tech stack

Python · pandas · NumPy · Matplotlib · Seaborn  · Jupyter