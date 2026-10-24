"""
eda_toolkit.py
==============
A reusable EDA toolkit that covers all three levels of analysis:

    UNIVARIATE    -> look at ONE column at a time
                     (distribution, spread, skewness, outliers, category counts)
    BIVARIATE     -> look at TWO columns together
                     (numeric vs numeric, category vs numeric, category vs category)
    MULTIVARIATE  -> look at MANY columns together
                     (correlation matrix, multicollinearity/VIF, PCA, pairplot, pivot heatmap)

Basic usage
-----------
    from eda_toolkit import EDAToolkit
    eda = EDAToolkit(df, target_col="Salary", save_dir="eda_output")
    eda.run_all()                              # everything
    eda.univariate().bivariate()               # or pick the levels you want
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:                                   # scipy is optional: only used for statistical tests
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class EDAToolkit:
    """Reusable Exploratory Data Analysis: univariate, bivariate and multivariate."""

    def __init__(self, df: pd.DataFrame, target_col: str = None, save_dir: str = None,
                 max_categories: int = 15, corr_threshold: float = 0.8, max_pairplot_cols: int = 5):
        """
        df             : DataFrame to analyse (a copy is made, your original is safe)
        target_col     : optional column you care most about (used to colour / compare plots)
        save_dir       : optional folder; if given, every plot and the report are saved there
        max_categories : categorical columns with more unique values than this are skipped
                         in bivariate/multivariate steps (they are usually IDs or free text)
        corr_threshold : |correlation| above this is flagged as "highly correlated"
        max_pairplot_cols : the pairplot is limited to this many numeric columns (it gets slow)
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty pandas DataFrame")
        if target_col is not None and target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in columns")

        self.df = df.copy()
        self.target_col = target_col
        self.save_dir = save_dir
        self.max_categories = max_categories
        self.corr_threshold = corr_threshold
        self.max_pairplot_cols = max_pairplot_cols

        self.num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        all_cat = self.df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
        # keep only categorical columns with a sensible number of categories
        self.cat_cols = [c for c in all_cat if self.df[c].nunique() <= max_categories]
        self.skipped_cols = [c for c in all_cat if c not in self.cat_cols]

        self.findings = []             # plain-English insights, written to the report at the end
        sns.set_theme(style="whitegrid")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _note(self, text):
        self.findings.append(text)

    def _header(self, title):
        print("\n" + "=" * 70 + f"\n {title}\n" + "=" * 70)

    def _show(self, filename):
        plt.tight_layout()
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _grid(n, ncols=3, cell=(4.5, 3.5)):
        """Create a grid of n subplots and return (fig, flat axes list)."""
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
        axes = axes.flatten()
        for ax in axes[n:]:
            fig.delaxes(ax)
        return fig, axes[:n]

    # ==================================================================
    # 0. OVERVIEW
    # ==================================================================
    def overview(self):
        """Shape, types, missing values and duplicates - always look at this first."""
        self._header("0. DATA OVERVIEW")
        r, c = self.df.shape
        print(f"Shape: {r} rows x {c} columns")
        print(f"Numerical  ({len(self.num_cols)}): {self.num_cols}")
        print(f"Categorical({len(self.cat_cols)}): {self.cat_cols}")
        if self.skipped_cols:
            print(f"Skipped high-cardinality columns: {self.skipped_cols}")

        miss = self.df.isnull().sum()
        miss = miss[miss > 0]
        if miss.empty:
            print("\nMissing values: none")
        else:
            table = pd.DataFrame({"count": miss, "percent": (100 * miss / r).round(1)})
            print("\nMissing values:\n", table)
            for col, row in table.iterrows():
                self._note(f"'{col}' has {int(row['count'])} missing values ({row['percent']}%).")
        dups = int(self.df.duplicated().sum())
        print(f"\nDuplicate rows: {dups}")
        if dups:
            self._note(f"{dups} duplicate row(s) found.")
        self._note(f"Dataset: {r} rows, {c} columns ({len(self.num_cols)} numeric, {len(self.cat_cols)} categorical).")
        return self

    # ==================================================================
    # 1. UNIVARIATE  (one column at a time)
    # ==================================================================
    def univariate(self):
        """Describe and plot every column on its own."""
        self._header("1. UNIVARIATE ANALYSIS")

        # ---- numeric columns ----
        if self.num_cols:
            summary = self.df[self.num_cols].describe().T
            summary["skew"] = self.df[self.num_cols].skew()
            summary["kurtosis"] = self.df[self.num_cols].kurt()

            # IQR outlier rule: outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            q1, q3 = self.df[self.num_cols].quantile(0.25), self.df[self.num_cols].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.df[self.num_cols] < q1 - 1.5 * iqr) |
                        (self.df[self.num_cols] > q3 + 1.5 * iqr)).sum()
            summary["outliers"] = outliers
            print("Numeric summary:\n", summary.round(2))

            for col in self.num_cols:
                if abs(summary.loc[col, "skew"]) > 1:
                    side = "right" if summary.loc[col, "skew"] > 0 else "left"
                    self._note(f"'{col}' is strongly skewed to the {side} (skew={summary.loc[col, 'skew']:.2f}); "
                               f"consider a log transform.")
                if outliers[col] > 0:
                    self._note(f"'{col}' has {outliers[col]} outlier(s) (IQR rule).")

            # Histogram + KDE and boxplot for each numeric column
            fig, axes = self._grid(len(self.num_cols) * 2, ncols=4, cell=(3.6, 3))
            for i, col in enumerate(self.num_cols):
                sns.histplot(self.df[col].dropna(), kde=True, ax=axes[2 * i], color="teal")
                axes[2 * i].set_title(f"{col} - distribution", fontsize=9)
                sns.boxplot(x=self.df[col].dropna(), ax=axes[2 * i + 1], color="lightseagreen")
                axes[2 * i + 1].set_title(f"{col} - boxplot", fontsize=9)
            self._show("1_univariate_numeric.png")

        # ---- categorical columns ----
        if self.cat_cols:
            for col in self.cat_cols:
                pct = (self.df[col].value_counts(normalize=True) * 100).round(1)
                print(f"\n{col} (% of rows):\n{pct.to_string()}")
                if pct.iloc[0] > 80:
                    self._note(f"'{col}' is imbalanced: '{pct.index[0]}' is {pct.iloc[0]}% of rows.")
            fig, axes = self._grid(len(self.cat_cols), ncols=3)
            for ax, col in zip(axes, self.cat_cols):
                order = self.df[col].value_counts().index
                sns.countplot(data=self.df, y=col, order=order, hue=col, palette="viridis",
                              legend=False, ax=ax)
                ax.set_title(f"{col} - counts", fontsize=9)
            self._show("1_univariate_categorical.png")
        return self

    # ==================================================================
    # 2. BIVARIATE  (two columns together)
    # ==================================================================
    def bivariate(self, top_n_pairs: int = 6):
        """
        Three kinds of pairs:
            numeric  vs numeric      -> correlation + scatter plots
            category vs numeric      -> group means + boxplots (+ ANOVA / t-test)
            category vs category     -> crosstab heatmap (+ chi-square, Cramer's V)
        """
        self._header("2. BIVARIATE ANALYSIS")

        # ---- numeric vs numeric ----
        if len(self.num_cols) >= 2:
            print("--- Numeric vs Numeric ---")
            pairs = []
            for a, b in itertools.combinations(self.num_cols, 2):
                sub = self.df[[a, b]].dropna()
                if len(sub) > 2:
                    pairs.append((a, b, sub[a].corr(sub[b]), sub[a].corr(sub[b], method="spearman")))
            pair_df = pd.DataFrame(pairs, columns=["col_1", "col_2", "pearson", "spearman"])
            pair_df = pair_df.reindex(pair_df["pearson"].abs().sort_values(ascending=False).index)
            print(pair_df.round(3).head(15).to_string(index=False))
            print("(Pearson = straight-line relationship, Spearman = any rising/falling relationship)")

            for _, r in pair_df.iterrows():
                if abs(r["pearson"]) >= self.corr_threshold:
                    self._note(f"'{r['col_1']}' and '{r['col_2']}' are highly correlated (r={r['pearson']:.2f}).")

            top = pair_df.head(top_n_pairs)
            fig, axes = self._grid(len(top), ncols=3)
            for ax, (_, r) in zip(axes, top.iterrows()):
                hue = self.target_col if (self.target_col in self.cat_cols) else None
                sns.scatterplot(data=self.df, x=r["col_1"], y=r["col_2"], hue=hue, ax=ax, alpha=0.7)
                sns.regplot(data=self.df, x=r["col_1"], y=r["col_2"], scatter=False, ax=ax, color="red",
                            line_kws={"linewidth": 1})
                ax.set_title(f"r = {r['pearson']:.2f}", fontsize=9)
            self._show("2_bivariate_num_num.png")

        # ---- category vs numeric ----
        if self.cat_cols and self.num_cols:
            print("\n--- Category vs Numeric ---")
            combos = [(c, n) for c in self.cat_cols for n in self.num_cols]
            # if a numeric target exists, focus on it so the output stays readable
            if self.target_col in self.num_cols:
                combos = [(c, self.target_col) for c in self.cat_cols]
            rows = []
            for cat, num in combos:
                groups = [g[num].dropna().values for _, g in self.df.groupby(cat) if g[num].notna().sum() > 1]
                p = np.nan
                if HAS_SCIPY and len(groups) >= 2:
                    p = stats.f_oneway(*groups).pvalue          # ANOVA (2 groups = t-test equivalent)
                means = self.df.groupby(cat)[num].mean()
                rows.append({"category": cat, "numeric": num, "highest_group": means.idxmax(),
                             "lowest_group": means.idxmin(), "p_value": p})
                if pd.notna(p) and p < 0.05:
                    self._note(f"'{num}' differs significantly across '{cat}' groups (p={p:.3f}); "
                               f"highest in '{means.idxmax()}', lowest in '{means.idxmin()}'.")
            print(pd.DataFrame(rows).round(4).to_string(index=False))
            print("(p_value < 0.05 means the group averages are really different, not just chance)")

            plot_combos = combos[:9]
            fig, axes = self._grid(len(plot_combos), ncols=3)
            for ax, (cat, num) in zip(axes, plot_combos):
                sns.boxplot(data=self.df, x=cat, y=num, hue=cat, palette="Set2", legend=False, ax=ax)
                ax.set_title(f"{num} by {cat}", fontsize=9)
                ax.tick_params(axis="x", rotation=30)
            self._show("2_bivariate_cat_num.png")

        # ---- category vs category ----
        if len(self.cat_cols) >= 2:
            print("\n--- Category vs Category ---")
            combos = list(itertools.combinations(self.cat_cols, 2))[:6]
            rows = []
            fig, axes = self._grid(len(combos), ncols=3)
            for ax, (a, b) in zip(axes, combos):
                ct = pd.crosstab(self.df[a], self.df[b])
                sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_title(f"{a} vs {b}", fontsize=9)
                if HAS_SCIPY and ct.shape[0] > 1 and ct.shape[1] > 1:
                    chi2, p, _, _ = stats.chi2_contingency(ct)
                    cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
                    rows.append({"col_1": a, "col_2": b, "p_value": p, "cramers_v": cramers_v})
                    if p < 0.05:
                        self._note(f"'{a}' and '{b}' are associated (chi-square p={p:.3f}, Cramer's V={cramers_v:.2f}).")
            self._show("2_bivariate_cat_cat.png")
            if rows:
                print(pd.DataFrame(rows).round(4).to_string(index=False))
                print("(Cramer's V: 0 = no link, 1 = perfect link)")
        return self

    # ==================================================================
    # 3. MULTIVARIATE  (many columns together)
    # ==================================================================
    def multivariate(self):
        """Correlation matrix, multicollinearity (VIF), PCA, pairplot and pivot heatmap."""
        self._header("3. MULTIVARIATE ANALYSIS")
        data = self.df[self.num_cols].dropna()

        if len(self.num_cols) >= 2 and len(data) > len(self.num_cols):
            # ---- correlation heatmap ----
            corr = data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            size = max(5, len(self.num_cols) * 0.9)
            plt.figure(figsize=(size + 1, size))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
            plt.title("Correlation matrix")
            self._show("3_multivariate_correlation.png")

            # ---- VIF: how much each column can be predicted from all the others ----
            # VIF = diagonal of the inverse correlation matrix. VIF > 5 (or 10) = multicollinearity.
            try:
                vif = pd.Series(np.diag(np.linalg.inv(corr.values)), index=corr.columns, name="VIF")
                print("\nVariance Inflation Factor (VIF):\n", vif.round(2).to_string())
                print("(VIF > 5 suggests the column is largely explained by the other columns)")
                for col, v in vif.items():
                    if v > 5:
                        self._note(f"'{col}' has high multicollinearity (VIF={v:.1f}).")
            except np.linalg.LinAlgError:
                print("\nVIF skipped: correlation matrix is singular (some columns are perfectly correlated).")
                self._note("Some numeric columns are perfectly correlated (singular matrix).")

            # ---- PCA (done with plain numpy so no extra library is needed) ----
            z = (data - data.mean()) / data.std(ddof=0).replace(0, 1)
            _, s, vt = np.linalg.svd(z.values, full_matrices=False)
            explained = (s ** 2) / (s ** 2).sum()
            cum = np.cumsum(explained)
            print("\nPCA explained variance by component:")
            print(pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(explained))],
                                "explained_%": (explained * 100).round(1),
                                "cumulative_%": (cum * 100).round(1)}).to_string(index=False))
            n90 = int(np.searchsorted(cum, 0.90) + 1)
            self._note(f"{n90} principal component(s) capture 90% of the variance of {len(self.num_cols)} numeric columns.")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].bar(range(1, len(explained) + 1), explained * 100, color="teal")
            axes[0].plot(range(1, len(cum) + 1), cum * 100, "o-", color="red")
            axes[0].set_xlabel("Principal component"); axes[0].set_ylabel("% variance")
            axes[0].set_title("PCA scree plot (red = cumulative)")
            scores = z.values @ vt.T[:, :2]
            hue = self.df.loc[data.index, self.target_col] if self.target_col else None
            sc = axes[1].scatter(scores[:, 0], scores[:, 1], c=hue if (hue is not None and
                                 pd.api.types.is_numeric_dtype(hue)) else None, cmap="viridis", alpha=0.7)
            if hue is not None and pd.api.types.is_numeric_dtype(hue):
                plt.colorbar(sc, ax=axes[1], label=self.target_col)
            axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].set_title("Data projected on PC1 vs PC2")
            self._show("3_multivariate_pca.png")

            # ---- pairplot of the most relevant numeric columns ----
            cols = self.num_cols[: self.max_pairplot_cols]
            if self.target_col in self.num_cols and self.target_col not in cols:
                cols = cols[:-1] + [self.target_col]
            hue = next((c for c in self.cat_cols if self.df[c].nunique() <= 4), None)
            g = sns.pairplot(self.df[cols + ([hue] if hue else [])].dropna(), hue=hue,
                             diag_kind="kde", corner=True, plot_kws={"alpha": 0.6, "s": 25})
            g.fig.suptitle(f"Pairplot (colour = {hue})" if hue else "Pairplot", y=1.02)
            if self.save_dir:
                g.savefig(os.path.join(self.save_dir, "3_multivariate_pairplot.png"), dpi=130)
            plt.show()

        # ---- pivot heatmap: mean of a numeric column across TWO categories ----
        if len(self.cat_cols) >= 2 and self.num_cols:
            value = self.target_col if self.target_col in self.num_cols else self.num_cols[0]
            a, b = self.cat_cols[:2]
            pivot = self.df.pivot_table(values=value, index=a, columns=b, aggfunc="mean")
            plt.figure(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.title(f"Average {value} by {a} and {b}")
            self._show("3_multivariate_pivot.png")
        return self

    # ==================================================================
    # REPORT + RUN ALL
    # ==================================================================
    def report(self, filename="eda_report.md"):
        """Print all insights and save them as a Markdown file."""
        self._header("KEY FINDINGS")
        for i, f in enumerate(self.findings, 1):
            print(f"{i}. {f}")
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# EDA Report\n\n## Key Findings\n\n")
                fh.writelines(f"{i}. {f}\n" for i, f in enumerate(self.findings, 1))
            print(f"\nReport saved to {path}")
        return self

    def run_all(self):
        """Run the full pipeline: overview -> univariate -> bivariate -> multivariate -> report."""
        self.findings = []
        return self.overview().univariate().bivariate().multivariate().report()
