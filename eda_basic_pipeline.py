"""
eda_pipeline.py
================
A reusable Exploratory Data Analysis (EDA) pipeline.

Give it ANY pandas DataFrame and it will:
    1. Audit the structure   (shape, data types, memory)
    2. Audit the quality     (missing values, duplicates, constant columns, high-cardinality columns)
    3. Describe the data     (summary stats, skewness, kurtosis, category counts)
    4. Detect outliers       (IQR method)
    5. Plot                  (histograms, boxplots, category bars, missing-value map, correlation heatmap)
    6. Analyse the target    (optional: relation of every feature with a target column)
    7. Write a findings report (plain-English insights, saved as a Markdown file)

Basic usage
-----------
    from eda_pipeline import EDAPipeline
    pipeline = EDAPipeline(df, target_col="Salary")
    pipeline.run_all()                      # run everything
    pipeline.structural_audit().quality_audit()   # or run steps one by one (method chaining)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAPipeline:
    """Performs Exploratory Data Analysis (EDA) on a given dataset."""

    def __init__(self, df: pd.DataFrame, target_col: str = None, save_dir: str = None,
                 high_corr_threshold: float = 0.8, high_cardinality_threshold: int = 50):
        """
        Parameters
        ----------
        df : the DataFrame to analyse (a copy is made, so your original is never changed)
        target_col : optional, the column you want to predict / focus on
        save_dir : optional, folder where plots and the report will be saved
        high_corr_threshold : two features with |correlation| above this are flagged
        high_cardinality_threshold : categorical columns with more unique values than this are flagged
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("The DataFrame is empty - nothing to analyse")
        if target_col is not None and target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in the DataFrame columns")

        self.df = df.copy()
        self.target_col = target_col
        self.save_dir = save_dir
        self.high_corr_threshold = high_corr_threshold
        self.high_cardinality_threshold = high_cardinality_threshold

        # Column types. 'category', 'string' and 'bool' are also treated as categorical
        # (the old version only caught 'object').
        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(
            include=["object", "category", "string", "bool"]).columns.tolist()

        # Everything we discover is stored here, then used to write the final report
        self.findings = []

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Small helper functions (used internally by the steps below)
    # ------------------------------------------------------------------
    def _add_finding(self, text: str):
        """Store an insight so it appears in the final report."""
        self.findings.append(text)

    def _header(self, title: str):
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)

    def _finish_plot(self, filename: str):
        """Save the figure (if save_dir is set) and show it."""
        plt.tight_layout()
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Step 1: Structure
    # ------------------------------------------------------------------
    def structural_audit(self):
        """Shape, data types and memory usage of the dataset."""
        self._header("1. STRUCTURAL AUDIT")
        rows, cols = self.df.shape
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"Data shape      : {rows} rows, {cols} columns")
        print(f"Numerical cols  : {len(self.numerical_cols)} -> {self.numerical_cols}")
        print(f"Categorical cols: {len(self.categorical_cols)} -> {self.categorical_cols}")
        print(f"Memory usage    : {memory_mb:.4f} MB")
        if self.target_col:
            print(f"Target column   : {self.target_col}")
        print("\n--- Data Types & Non-Null Counts ---")
        self.df.info()
        print("\n--- First 5 rows ---")
        print(self.df.head())

        self._add_finding(f"Dataset has {rows} rows and {cols} columns "
                          f"({len(self.numerical_cols)} numerical, {len(self.categorical_cols)} categorical).")
        if rows < 100:
            self._add_finding("Dataset is small (< 100 rows): statistics and correlations may be unreliable.")
        return self

    # ------------------------------------------------------------------
    # Step 2: Quality
    # ------------------------------------------------------------------
    def quality_audit(self):
        """Missing values, duplicate rows, constant columns, high-cardinality columns."""
        self._header("2. DATA QUALITY AUDIT")

        # --- Missing values ---
        missing_count = self.df.isnull().sum()
        missing_pct = 100 * missing_count / len(self.df)
        missing_table = pd.concat([missing_count, missing_pct], axis=1,
                                  keys=["Missing Count", "Missing Percentage"])
        missing_table = missing_table[missing_table["Missing Count"] > 0] \
            .sort_values("Missing Percentage", ascending=False)

        print("Missing Data Profile:")
        if missing_table.empty:
            print("No missing values found.")
            self._add_finding("No missing values found.")
        else:
            print(missing_table)
            for col, row in missing_table.iterrows():
                level = "HIGH - consider dropping the column" if row["Missing Percentage"] > 40 else \
                        "moderate - impute (fill) carefully" if row["Missing Percentage"] > 5 else "low"
                self._add_finding(f"Column '{col}' has {int(row['Missing Count'])} missing values "
                                  f"({row['Missing Percentage']:.1f}%) - {level}.")

        # --- Duplicates ---
        duplicate_count = int(self.df.duplicated().sum())
        print(f"\nNumber of duplicate rows: {duplicate_count}")
        if duplicate_count > 0:
            print("Duplicate rows (all copies shown):")
            print(self.df[self.df.duplicated(keep=False)])
            self._add_finding(f"{duplicate_count} duplicate row(s) found - verify if they are real or errors.")

        # --- Constant columns (only one unique value = no information) ---
        constant_cols = [c for c in self.df.columns if self.df[c].nunique(dropna=True) <= 1]
        print(f"\nConstant columns: {constant_cols if constant_cols else 'None'}")
        if constant_cols:
            self._add_finding(f"Constant column(s) {constant_cols} carry no information - consider dropping.")

        # --- High cardinality (too many unique categories, e.g. IDs or names) ---
        high_card = [c for c in self.categorical_cols
                     if self.df[c].nunique() > self.high_cardinality_threshold]
        print(f"High-cardinality categorical columns (> {self.high_cardinality_threshold} unique): "
              f"{high_card if high_card else 'None'}")
        if high_card:
            self._add_finding(f"High-cardinality column(s) {high_card} - may be IDs or free text; "
                              f"avoid one-hot encoding them.")
        return self

    # ------------------------------------------------------------------
    # Step 3: Descriptive statistics
    # ------------------------------------------------------------------
    def descriptive_stats(self):
        """Summary stats for numerical and categorical columns, plus skewness and kurtosis."""
        self._header("3. DESCRIPTIVE STATISTICS")
        if self.numerical_cols:
            print("Numerical Features Summary:")
            summary = self.df[self.numerical_cols].describe().T
            summary["skewness"] = self.df[self.numerical_cols].skew()
            summary["kurtosis"] = self.df[self.numerical_cols].kurt()
            print(summary.round(2))

            # Skewness: 0 = symmetric. Beyond +-1 the distribution is strongly lopsided.
            for col in self.numerical_cols:
                skew = summary.loc[col, "skewness"]
                if pd.notna(skew) and abs(skew) > 1:
                    direction = "right (long tail of high values)" if skew > 0 else "left (long tail of low values)"
                    self._add_finding(f"'{col}' is strongly skewed {direction} (skew = {skew:.2f}) - "
                                      f"consider a log transform.")

        if self.categorical_cols:
            print("\nCategorical Features Summary:")
            print(self.df[self.categorical_cols].describe().T)
            print("\nValue counts (percentage):")
            for col in self.categorical_cols:
                if self.df[col].nunique() <= self.high_cardinality_threshold:
                    print(f"\n{col}:")
                    print((self.df[col].value_counts(normalize=True) * 100).round(1).astype(str) + " %")
                    top_share = self.df[col].value_counts(normalize=True).iloc[0]
                    if top_share > 0.8:
                        self._add_finding(f"'{col}' is imbalanced: one category makes up {top_share:.0%} of rows.")
        return self

    # ------------------------------------------------------------------
    # Step 4: Outliers
    # ------------------------------------------------------------------
    def outlier_analysis(self):
        """
        Detect outliers with the IQR rule:
            Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
            anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier.
        """
        self._header("4. OUTLIER ANALYSIS (IQR method)")
        if not self.numerical_cols:
            print("No numerical columns.")
            return self

        records = []
        for col in self.numerical_cols:
            data = self.df[col].dropna()
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((data < lower) | (data > upper)).sum())
            records.append({"Column": col, "Lower Bound": lower, "Upper Bound": upper,
                            "Outlier Count": n_out, "Outlier %": 100 * n_out / len(data)})
            if n_out > 0:
                self._add_finding(f"'{col}' has {n_out} outlier(s) outside [{lower:.1f}, {upper:.1f}].")

        self.outlier_table = pd.DataFrame(records).set_index("Column")
        print(self.outlier_table.round(2))
        return self

    # ------------------------------------------------------------------
    # Step 5: Plots
    # ------------------------------------------------------------------
    def plot_missing(self):
        """Heatmap showing WHERE the missing values are."""
        if self.df.isnull().sum().sum() == 0:
            return self
        self._header("5a. MISSING VALUE MAP")
        plt.figure(figsize=(10, 4))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title("Missing Values Map (yellow = missing)")
        self._finish_plot("missing_values.png")
        return self

    def plot_distributions(self):
        """Univariate plots: histograms + boxplots for numeric, bar plots for categorical."""
        self._header("5b. UNIVARIATE VISUALIZATIONS")

        if self.numerical_cols:
            n = len(self.numerical_cols)
            # squeeze=False always returns a 2D array - this fixes the crash the old version had
            # when there was only one column.
            fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 3 * n), squeeze=False)
            for i, col in enumerate(self.numerical_cols):
                sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i][0], color="teal")
                axes[i][0].set_title(f"Distribution of {col}", fontsize=10)
                sns.boxplot(x=self.df[col].dropna(), ax=axes[i][1], color="lightseagreen")
                axes[i][1].set_title(f"Boxplot of {col} (dots = outliers)", fontsize=10)
            self._finish_plot("numeric_distributions.png")

        if self.categorical_cols:
            n = len(self.categorical_cols)
            fig, axes = plt.subplots(nrows=(n + 1) // 2, ncols=2,
                                     figsize=(12, 3 * ((n + 1) // 2)), squeeze=False)
            axes = axes.flatten()
            for i, col in enumerate(self.categorical_cols):
                top_cats = self.df[col].value_counts().head(10)  # top 10 only, for high cardinality
                # hue=y and legend=False is the new seaborn way (removes the FutureWarning)
                sns.barplot(x=top_cats.values, y=top_cats.index.astype(str), hue=top_cats.index.astype(str),
                            palette="viridis", legend=False, ax=axes[i])
                axes[i].set_title(f"Top Categories: {col}", fontsize=10)
            for j in range(len(self.categorical_cols), len(axes)):
                fig.delaxes(axes[j])
            self._finish_plot("categorical_distributions.png")
        return self

    def plot_correlations(self):
        """Correlation heatmap + list of highly correlated feature pairs."""
        if len(self.numerical_cols) < 2:
            return self
        self._header("5c. CORRELATION ANALYSIS")
        corr_matrix = self.df[self.numerical_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # hide the repeated upper triangle

        plt.figure(figsize=(max(6, len(self.numerical_cols)), max(5, len(self.numerical_cols) * 0.8)))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
        plt.title("Numerical Correlation Matrix (Upper Triangle Masked)")
        self._finish_plot("correlation_matrix.png")

        # Find pairs above the threshold
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr_matrix.iloc[i, j]
                if pd.notna(r) and abs(r) >= self.high_corr_threshold:
                    pairs.append((cols[i], cols[j], r))
        if pairs:
            print(f"Highly correlated pairs (|r| >= {self.high_corr_threshold}):")
            for a, b, r in pairs:
                print(f"  {a} <-> {b}: {r:.2f}")
                self._add_finding(f"'{a}' and '{b}' are highly correlated (r = {r:.2f}) - "
                                  f"possible multicollinearity, you may keep only one.")
        else:
            print(f"No feature pairs with |r| >= {self.high_corr_threshold}.")
        return self

    # ------------------------------------------------------------------
    # Step 6: Target analysis (only runs if target_col was given)
    # ------------------------------------------------------------------
    def target_analysis(self):
        """How does every feature relate to the target column?"""
        if not self.target_col:
            return self
        self._header(f"6. TARGET ANALYSIS: {self.target_col}")
        target = self.target_col

        if target in self.numerical_cols:
            # Numeric target: correlation with other numeric features,
            # and average target per category for categorical features.
            others = [c for c in self.numerical_cols if c != target]
            if others:
                corr = self.df[others + [target]].corr()[target].drop(target).sort_values(
                    key=abs, ascending=False)
                print("Correlation with target:")
                print(corr.round(3))
                best = corr.index[0]
                self._add_finding(f"Strongest numeric relationship with '{target}' is '{best}' "
                                  f"(r = {corr[best]:.2f}).")
            for col in self.categorical_cols:
                if self.df[col].nunique() <= self.high_cardinality_threshold:
                    print(f"\nAverage {target} by {col}:")
                    print(self.df.groupby(col)[target].agg(["mean", "median", "count"]).round(1))
            if self.categorical_cols:
                n = len(self.categorical_cols)
                fig, axes = plt.subplots(nrows=(n + 1) // 2, ncols=2,
                                         figsize=(12, 3.5 * ((n + 1) // 2)), squeeze=False)
                axes = axes.flatten()
                for i, col in enumerate(self.categorical_cols):
                    sns.boxplot(data=self.df, x=col, y=target, hue=col, palette="Set2",
                                legend=False, ax=axes[i])
                    axes[i].set_title(f"{target} by {col}", fontsize=10)
                for j in range(n, len(axes)):
                    fig.delaxes(axes[j])
                self._finish_plot("target_vs_categories.png")
        else:
            # Categorical target: show class balance
            print("Class balance:")
            print((self.df[target].value_counts(normalize=True) * 100).round(1))
            top_share = self.df[target].value_counts(normalize=True).iloc[0]
            if top_share > 0.7:
                self._add_finding(f"Target '{target}' is imbalanced ({top_share:.0%} in one class).")
        return self

    # ------------------------------------------------------------------
    # Step 7: Report
    # ------------------------------------------------------------------
    def generate_report(self, filename: str = "eda_report.md"):
        """Print all insights in plain English and save them as a Markdown file."""
        self._header("7. KEY FINDINGS")
        if not self.findings:
            print("Run the other steps first - no findings yet.")
            return self
        for i, f in enumerate(self.findings, 1):
            print(f"{i}. {f}")

        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# EDA Report\n\n## Key Findings\n\n")
                for i, f in enumerate(self.findings, 1):
                    fh.write(f"{i}. {f}\n")
            print(f"\nReport saved to: {path}")
        return self

    # ------------------------------------------------------------------
    # Run everything
    # ------------------------------------------------------------------
    def run_all(self):
        """Execute the entire pipeline sequentially (method chaining)."""
        self.findings = []  # reset so running twice doesn't duplicate insights
        return (self.structural_audit()
                    .quality_audit()
                    .descriptive_stats()
                    .outlier_analysis()
                    .plot_missing()
                    .plot_distributions()
                    .plot_correlations()
                    .target_analysis()
                    .generate_report())
