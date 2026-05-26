"""
main.py
───────
Entry point for the 3MuViS experiment pipeline.

Usage:
    python main.py                          # uses config.yaml in same directory
    python main.py --config path/to/config.yaml
    python main.py --datasets htad har70    # run only specific datasets
    python main.py --no-brute-force         # skip exhaustive search

Output per dataset:
    results/tests_<dataset>_<k>f_<date>.csv
    summaries/summary_<dataset>_<k>f_<date>.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# Make sure sibling modules are importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_dataset
from experiment import run_brute_force, run_iteration
from muvis_core import MODEL_CATALOG


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Summary logger
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(
    dataset_name: str,
    dataset_cfg: dict,
    exp_cfg: dict,
    out_cfg: dict,
    run_brute: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name}")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────
    X, y, views, encoder = load_dataset(dataset_cfg)
    print(f"  X={X.shape}  classes={list(encoder.classes_)}  views={len(views)}")

    # ── Experiment parameters ──────────────────────────────────────────
    n_iterations  = exp_cfg["n_iterations"]
    initial_seed  = exp_cfg["initial_seed"]
    metric        = exp_cfg["metric"]
    k_folds_base  = exp_cfg["k_folds_base"]
    k_folds_meta  = exp_cfg["k_folds_meta"]
    test_size     = exp_cfg["test_size"]
    k             = k_folds_meta   # used in filenames

    today         = date.today().strftime("%Y-%m-%d")
    results_dir   = out_cfg["results_dir"]
    summaries_dir = out_cfg["summaries_dir"]
    os.makedirs(results_dir,   exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    path_results  = os.path.join(results_dir,
                                 f"tests_{dataset_name}_{k}f_{today}.csv")
    path_summary  = os.path.join(summaries_dir,
                                 f"summary_{dataset_name}_{k}f_{today}.txt")

    np.random.seed(initial_seed)
    seed = initial_seed

    # ── Iteration loop ─────────────────────────────────────────────────
    global_results: list[dict] = []
    t_iter_start = time.time()

    for i in range(n_iterations):
        seed += 1
        print(f"\n  --- Iteration {i+1}/{n_iterations}  (seed={seed}) ---")
        iteration_results = run_iteration(
            X, y, views, encoder,
            seed=seed,
            metric=metric,
            k_folds_base=k_folds_base,
            k_folds_meta=k_folds_meta,
            test_size=test_size,
        )
        global_results.extend(iteration_results)

        # Checkpoint: save CSV after every iteration so progress is not lost
        pd.DataFrame(global_results).to_csv(path_results, index=False)

    total_iter_time = time.time() - t_iter_start

    # ── Summary log ────────────────────────────────────────────────────
    summary_text = (
        f"\nDate: {date.today()}\n"
        f"Dataset: {dataset_name}\n"
        f"Metric optimised: {metric}\n"
        f"Experiment iterations: {n_iterations}\n"
        f"Total iteration time: {total_iter_time:.1f} seconds\n"
        f"K cross-validation folds: {k}\n"
        f"Initial seed: {initial_seed}\n"
        f"ML algorithms: {list(MODEL_CATALOG.keys())}\n"
    )
    write_summary(path_summary, summary_text)
    print(f"\n  [checkpoint] results → {path_results}")

    # ── Brute force ────────────────────────────────────────────────────
    if run_brute:
        from sklearn.metrics import get_scorer
        metric_func = get_scorer(metric)._score_func

        print(f"\n  --- Brute Force ---")
        t_brute_start = time.time()
        bf_result = run_brute_force(
            X, y,
            views=views,
            metric=metric,
            metric_func=metric_func,
            encoder=encoder,
            seed=initial_seed,
            k_folds=k_folds_meta,
        )
        brute_time = time.time() - t_brute_start
        global_results.append(bf_result)

        # Final save including brute force
        pd.DataFrame(global_results).to_csv(path_results, index=False)

        brute_summary = (
            f"----------------------------------------------\n"
            f"Brute force time: {brute_time:.1f} seconds\n"
            f"Seed: {initial_seed}\n"
        )
        write_summary(path_summary, brute_summary)

    print(f"\n  [done] {dataset_name} → {path_results}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="3MuViS experiment pipeline")
    parser.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml (default: config.yaml next to main.py)",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        help="Run only specific datasets by key (e.g. --datasets htad har70). "
             "Defaults to all datasets in config.",
    )
    parser.add_argument(
        "--no-brute-force", action="store_true",
        help="Skip brute-force search regardless of config setting.",
    )
    args = parser.parse_args()

    cfg           = load_config(args.config)
    exp_cfg       = cfg["experiment"]
    out_cfg       = cfg["output"]
    datasets_cfg  = cfg["datasets"]
    run_brute     = exp_cfg.get("run_brute_force", True) and not args.no_brute_force

    # Filter datasets if --datasets flag is provided
    keys = args.datasets if args.datasets else list(datasets_cfg.keys())
    missing = [k for k in keys if k not in datasets_cfg]
    if missing:
        print(f"[error] Unknown dataset keys: {missing}")
        print(f"        Available: {list(datasets_cfg.keys())}")
        sys.exit(1)

    print(f"Running datasets: {keys}")
    print(f"Brute force: {run_brute}")

    for dataset_name in keys:
        run_dataset(
            dataset_name   = dataset_name,
            dataset_cfg    = datasets_cfg[dataset_name],
            exp_cfg        = exp_cfg,
            out_cfg        = out_cfg,
            run_brute      = run_brute,
        )

    print(f"\n{'='*60}")
    print("  All datasets complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
