"""
3MuVis Results Analysis Pipeline
=================================
Standardized, replicable analysis script.
Iterates a directory of results CSVs, applies a dataset name mapping,
produces per-dataset PNG figures and a consolidated CSV of table results.

Usage:
    python result_report.py --results_dir /path/to/csvs --output_dir /path/to/output

Directory layout expected:
    results/
        tests_htad.csv
        tests_har70.csv
        ...

Output layout:
    result_output/
        summary_table.csv          ← all datasets × metrics (mean ± CI + Wilcoxon p)
        <dataset_code>/
            upset_selection.png
            violin_mcc.png
            violin_accuracy.png
            violin_weighted.png
            violin_times.png
            wilcoxon_table.png
            training_time_table.png
            selection_time_table.png
"""

import os
import re
import argparse
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import wilcoxon, skew


warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET NAME MAPPING
# Keys   → substring(s) that appear in the CSV filename  (case-insensitive)
# Values → publication-ready dataset name used in titles, CSV index, filenames
# ─────────────────────────────────────────────────────────────────────────────
DATASET_NAME_MAP: dict[str, str] = {
    "berkeley": "D1 MHAD",
    "digits": "D2 Multiple Features Digits",
    "transport": "D3 Transportation Mode Detection",
    "htad":    "D4 HTAD",
    "har70":   "D5 HAR70+",
    "attacks":  "D6 UNSWNB15",
    
}

METHOD_NAME_MAP: dict[str,str] = {
    "3 MuVis" : "3MuViS",
    "brute force": "Brute Force",
    "random forest mvs": "Random Forest MVS",
    "random choice": "Random Choice",
    "Randopt" : "Randopt"
}



# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = 0.05
CI_Z  = 1.96          # z* for 95 % confidence interval
N_DECIMALS = 3        # all displayed floats use this many decimal places
FMT = f"{{:.{N_DECIMALS}f}}"   # format,e.g. "{:.3f}"

METHOD_3MUVIS     = METHOD_NAME_MAP["3 MuVis"]
METHOD_BRUTEFORCE = METHOD_NAME_MAP["brute force"]
BASELINE_METHODS  = [METHOD_NAME_MAP["random forest mvs"],
                     METHOD_NAME_MAP["random choice"],
                     METHOD_NAME_MAP["Randopt"]]

METRIC_SUBSTRINGS  = {"precision", "average", "f1", "recall", "matthews", "accuracy"}
WEIGHTED_METRICS   = ["weighted avg_precision", "weighted avg_recall", "weighted avg_f1-score"]
TIME_METRICS       = ["Fit time", "Predict time"]

SEARCH_TIME_COLS = {
    METHOD_3MUVIS: "3MuViS search time",
    "Randopt":     "Randopt search time",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dataset_name(filename: str) -> str:
    """
    Map a CSV filename to a publication-ready dataset name using DATASET_NAME_MAP.
    Falls back to the stem of the filename if no key matches.
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    for key, name in DATASET_NAME_MAP.items():
        if key.lower() in stem:
            return name
    # Fallback: strip common suffixes and return the stem
    clean = re.sub(r"tests?_|_\d+f$|_results?", "", stem, flags=re.IGNORECASE)
    return clean.upper()

def resolve_method_name(name: str) -> str:
    return METHOD_NAME_MAP.get(name, name)


def fmt(value: float) -> str:
    """Format a float to N_DECIMALS decimal places."""
    return FMT.format(value)


def mean_ci(series: pd.Series) -> str:
    """Return 'mean ∓ CI' string (95 % CI of the mean, z-based)."""
    n   = len(series)
    m   = np.mean(series)
    ci  = CI_Z * (np.std(series, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return f"{fmt(m)} ∓ {fmt(ci)}"


def wilcoxon_pval(a: np.ndarray, b: np.ndarray,
                  alternative: str = "greater") -> float | str:
    """
    One-sided Wilcoxon signed-rank test (H1: a > b).
    Returns p-value or 'N/A' when the test cannot be computed
    (e.g. all differences are zero).
    """
    diff = a - b
    if np.all(diff == 0):
        return "N/A"
    try:
        _, p = wilcoxon(a, b, alternative=alternative)
        return p
    except Exception:
        return "N/A"


def pval_star(p) -> str:
    """Annotate a p-value with significance stars."""
    if isinstance(p, str):
        return p
    if p < 0.001:
        return f"{fmt(p)} ***"
    if p < 0.01:
        return f"{fmt(p)} **"
    if p < ALPHA:
        return f"{fmt(p)} *"
    return fmt(p)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Identify metric columns by substring membership."""
    return [c for c in df.columns
            if any(sub in c.lower() for sub in METRIC_SUBSTRINGS)]


def get_bf_value(results: pd.DataFrame, col: str):
    """Return brute-force scalar for a metric column, or None."""
    bf_rows = results.loc[results["Method"] == METHOD_BRUTEFORCE, col]
    if not bf_rows.empty:
        return bf_rows.values[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Violin plot (single metric)
# ─────────────────────────────────────────────────────────────────────────────

def _violin_ax(ax, data: pd.DataFrame, col: str, dataset_name: str,
               results: pd.DataFrame) -> None:
    """Draw a single violin panel on *ax*."""
    plot_data    = data.loc[data["Method"] != METHOD_BRUTEFORCE]
    mean_by_meth = plot_data.groupby("Method")[col].mean()
    methods      = plot_data["Method"].unique()
    palette      = dict(zip(methods, sns.color_palette("Set2", len(methods))))

    sns.violinplot(data=plot_data, y=col, hue="Method",
                   palette=palette, cut=1, ax=ax)
    
    for method in methods:
        color = palette[method]
        ax.axhline(y=mean_by_meth[method], color=color,linestyle=":", linewidth=1.2, alpha=0.7)

    patches = [Patch(color=palette[m],
                     label=f"{m} (mean={fmt(mean_by_meth[m])})")
               for m in methods if m in mean_by_meth]

    bf_val = get_bf_value(results, col)
    if bf_val is not None:
        ax.axhline(y=bf_val, color="red", linestyle="--", linewidth=2)
        patches.append(Patch(color="red",
                             label=f"Brute Force = {fmt(bf_val)}"))

    ax.legend(handles=patches, title="Method", loc="lower right", fontsize=8)
    ax.set_title(f"{col} — {dataset_name}", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_violin_single(results: pd.DataFrame, col: str,
                       dataset_name: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    _violin_ax(ax, results, col, dataset_name, results)
    col_dict = {"matthews_corrcoef": "Matthew´s Correlation Coefficients",
                "accuracy": "Accuracy"}
    plt.tight_layout()
    plt.title(f"{col_dict.get(col, '')} distribution for dataset {dataset_name[0:2]}:{dataset_name[2:]}  by Method")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [violin {col}] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Multi-panel violin (weighted metrics)
# ─────────────────────────────────────────────────────────────────────────────

def plot_violin_multi(results: pd.DataFrame, cols: list[str],
                      dataset_name: str, out_path: str,
                      sharey: bool = True) -> None:
    existing = [c for c in cols if c in results.columns]
    if not existing:
        print(f"  [violin_multi] none of {cols} found – skipping.")
        return

    fig, axes = plt.subplots(1, len(existing),
                             figsize=(6 * len(existing), 5),
                             sharey=sharey)
    if len(existing) == 1:
        axes = [axes]

    for ax, col in zip(axes, existing):
        _violin_ax(ax, results, col, dataset_name, results)

    plt.suptitle(f"{dataset_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [violin_multi] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Wilcoxon table (rendered as a PNG table)
# ─────────────────────────────────────────────────────────────────────────────

def compute_wilcoxon_table(results: pd.DataFrame,
                           metrics: list[str]) -> pd.DataFrame:
    """
    For each metric: test H1: 3MuVis > baseline (one-sided Wilcoxon).
    Returns a DataFrame with columns = baselines, rows = metrics.
    Cells contain 'p-value [stars]'.

    Additional diagnostics columns:
        skew_diff_<baseline>: skewness of the difference distribution,
        to aid in interpreting the Wilcoxon test's applicability.
    """
    muvis = results.loc[results["Method"] == METHOD_3MUVIS]
    rows  = []

    for col in metrics:
        if col not in results.columns:
            continue
        muvis_vals = muvis[col].to_numpy()
        row = {"Metric": col}

        for baseline in BASELINE_METHODS:
            bl_rows = results.loc[results["Method"] == baseline, col]
            if bl_rows.empty or len(bl_rows) != len(muvis_vals):
                row[baseline]                       = "N/A"
                row[f"skew_diff_{baseline}"]        = "N/A"
                continue

            bl_vals  = bl_rows.to_numpy()
            diff     = muvis_vals - bl_vals
            sk       = fmt(skew(diff))
            p        = wilcoxon_pval(muvis_vals, bl_vals, alternative="greater")
            row[baseline]                    = pval_star(p)
            row[f"skew_diff_{baseline}"]     = sk

        rows.append(row)

    return pd.DataFrame(rows).set_index("Metric")


def _render_table_png(df: pd.DataFrame, title: str, out_path: str,
                      col_width: float = 2.2) -> None:
    """Render a DataFrame as a styled PNG table."""
    n_rows, n_cols = df.shape
    fig_w = max(8, col_width * (n_cols + 1))
    fig_h = max(2, 0.4 * (n_rows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)

    tbl = ax.table(
        cellText  = df.values,
        rowLabels = df.index.tolist(),
        colLabels = df.columns.tolist(),
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # Highlight significant cells
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0 or col_idx == -1:
            cell.set_facecolor("#d0d0d0")
        else:
            text = cell.get_text().get_text()
            if "***" in text:
                cell.set_facecolor("#90ee90")
            elif "**" in text:
                cell.set_facecolor("#b8f0b8")
            elif "*" in text:
                cell.set_facecolor("#d8f8d8")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [table] saved → {out_path}")


def plot_wilcoxon_table(results: pd.DataFrame, metrics: list[str],
                        dataset_name: str, out_path: str) -> pd.DataFrame:
    wdf = compute_wilcoxon_table(results, metrics)
    # Show only p-value columns (not skew diagnostics) in the figure
    display_cols = [c for c in wdf.columns if not c.startswith("skew_diff_")]
    _render_table_png(
        wdf[display_cols],
        title    = f"Wilcoxon Tests (H₁: 3MuVis > baseline) — {dataset_name}\n"
                   f"* p<0.05  ** p<0.01  *** p<0.001",
        out_path = out_path,
    )
    return wdf


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Mean ± CI table (MCC + training/selection times)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_ci_table(results: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    For each method × column: compute mean ± 95%CI string.
    Returns DataFrame with methods as rows and cols as columns.
    """
    rows = []
    for method in results["Method"].unique():
        sub  = results.loc[results["Method"] == method]
        row  = {"Method": method}
        for col in cols:
            if col in sub.columns and not sub[col].isna().all():
                row[col] = mean_ci(sub[col].dropna())
            else:
                row[col] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def plot_mean_ci_table(results: pd.DataFrame, cols: list[str],
                       title_prefix: str, dataset_name: str,
                       out_path: str) -> pd.DataFrame:
    existing = [c for c in cols if c in results.columns]
    if not existing:
        print(f"  [mean_ci_table] none of {cols} found – skipping.")
        return pd.DataFrame()

    df = compute_mean_ci_table(results, existing)
    _render_table_png(
        df,
        title    = f"{title_prefix} — {dataset_name}  (mean ∓ 95% CI)",
        out_path = out_path,
        col_width= 2.8,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PER-DATASET ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset(csv_path: str, out_root: str,
                    summary_rows: list) -> None:
    """
    Full analysis for one results CSV.  Appends flat summary row(s)
    (one per method × metric) to *summary_rows* for later CSV export.
    """
    dataset_name = resolve_dataset_name(csv_path)

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name}  ({os.path.basename(csv_path)})")
    print(f"{'='*60}")

    results = pd.read_csv(csv_path)
    results["Method"] = results["Method"].map(resolve_method_name).fillna(results["Method"])
    out_dir = ensure_dir(os.path.join(out_root, dataset_name))

    # ── Identify metric / time columns ──────────────────────────────────
    metrics      = metric_columns(results)
    existing_wt  = [c for c in WEIGHTED_METRICS   if c in results.columns]
    existing_tm  = [c for c in TIME_METRICS        if c in results.columns]

    # Identify search-time columns that actually exist
    search_cols  = [v for v in SEARCH_TIME_COLS.values() if v in results.columns]


    # ── Figure 2a: MCC violin ────────────────────────────────────────────
    mcc_col = next((c for c in results.columns if "matthews" in c.lower()), None)
    if mcc_col:
        plot_violin_single(results, mcc_col, dataset_name,
                           os.path.join(out_dir, "violin_mcc.png"))

    # ── Figure 2b: Accuracy violin ───────────────────────────────────────
    acc_col = next((c for c in results.columns if "accuracy" in c.lower()), None)
    if acc_col:
        plot_violin_single(results, acc_col, dataset_name,
                           os.path.join(out_dir, "violin_accuracy.png"))

    # ── Figure 3: Weighted metrics multi-violin ──────────────────────────
    if existing_wt:
        plot_violin_multi(results, existing_wt, dataset_name,
                          os.path.join(out_dir, "violin_weighted.png"),
                          sharey=True)

    # ── Figure 4: Fit/Predict time multi-violin ──────────────────────────
    if existing_tm:
        plot_violin_multi(results, existing_tm, dataset_name,
                          os.path.join(out_dir, "violin_times.png"),
                          sharey=False)

    # ── Figure 5: Wilcoxon table (MCC + all metrics) ────────────────────
    wdf = plot_wilcoxon_table(results, metrics, dataset_name,
                              os.path.join(out_dir, "wilcoxon_table.png"))

    # ── Figure 6: Training time mean ± CI table ──────────────────────────
    training_df = plot_mean_ci_table(
        results, existing_tm,
        title_prefix = "Training & Prediction Time",
        dataset_name = dataset_name,
        out_path     = os.path.join(out_dir, "training_time_table.png"),
    )

    # ── Figure 7: Search / selection time mean ± CI table ────────────────
    search_df = plot_mean_ci_table(
        results, search_cols,
        title_prefix = "Search / Selection Time (seconds)",
        dataset_name = dataset_name,
        out_path     = os.path.join(out_dir, "selection_time_table.png"),
    )

    # ── Collect summary rows ─────────────────────────────────────────────
    for method in results["Method"].unique():
        sub = results.loc[results["Method"] == method]
        base_row = {
            "dataset": dataset_name,
            "method":  method,
        }
        # Mean ± CI for each metric
        for col in metrics:
            if col in sub.columns:
                base_row[f"mean_ci_{col}"] = mean_ci(sub[col].dropna())

        # Mean ± CI for time columns
        for col in existing_tm + search_cols:
            if col in sub.columns:
                base_row[f"mean_ci_{col}"] = mean_ci(sub[col].dropna())

        # Wilcoxon p-values (only for 3 MuVis row)
        if method == METHOD_3MUVIS and not wdf.empty:
            for baseline in BASELINE_METHODS:
                if baseline in wdf.columns:
                    for metric_name, pval_str in wdf[baseline].items():
                        base_row[f"wilcoxon_{metric_name}_vs_{baseline}"] = pval_str

        summary_rows.append(base_row)

    print(f"  [done] outputs in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3MuVis results analysis pipeline"
    )
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing results CSV files")
    parser.add_argument("--output_dir", required= True,
                        help="Root output directory (default: ./output)")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_root    = ensure_dir(args.output_dir)

    # ── Discover CSVs ────────────────────────────────────────────────────
    csv_files = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.lower().endswith(".csv")
    ])
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s): {[os.path.basename(f) for f in csv_files]}")

    # ── Per-dataset loop ─────────────────────────────────────────────────
    summary_rows: list[dict] = []
    for csv_path in csv_files:
        analyze_dataset(csv_path, out_root, summary_rows)

    # ── Write consolidated summary CSV ───────────────────────────────────
    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_root, "summary_table.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"  Summary CSV saved → {summary_path}")
        print(f"  Shape: {summary_df.shape}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
