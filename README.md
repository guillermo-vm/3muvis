# 3MuViS — Three-View Multi-View Stacking

> **Heuristic optimization of multi-view stacking classifiers for human activity recognition**
> 
> *Manuscript under review — citation will be added upon publication*

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results Analysis](#results-analysis)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Contact](#contact)

---

## Overview

**3MuViS** is a heuristic algorithm that automates the selection of base learners and meta-learner in a multi-view stacking ensemble. Rather than exhaustively searching all possible classifier combinations — an NP-hard problem that grows as $12^{(n_\text{views}+1)}$ — 3MuViS decomposes the search into two sequential cross-validated stages:

1. **Base-learner search** — for each view independently, select the classifier that maximises Matthews Correlation Coefficient (MCC) under k-fold cross-validation.
2. **Meta-learner search** — given the selected base learners, select the meta-classifier that maximises MCC on a held-out meta-training split.

This repository contains the full experiment pipeline, the analysis and reporting scripts, and the HTAD dataset used in the study.

### Baselines

| Method | Description |
|---|---|
| **Brute Force** | Exhaustive search over all $12^{(n_\text{views}+1)}$ combinations (upper bound) |
| **RF-MVS** | Multi-view stacking with Random Forest at every level |
| **Random Choice** | Randomly sampled base and meta learners |
| **RandOpt** | Randomly sampled base learners with heuristic meta-learner search |

---

## Repository Structure

```
3muvis/
├── datasets/
│   └── HTAD.csv                  # HTAD dataset (included — see Datasets)
├── results/                      # Output CSVs written by main.py
├── summaries/                    # Experiment log files
├── src/
│   ├── main.py                   # Entry point — runs experiment loop
│   ├── experiment.py             # Per-method runners and iteration logic
│   ├── muvis_core.py             # 3MuViS algorithm and model catalog
│   ├── data_loader.py            # Dataset loading driven by config.yaml
│   ├── config.yaml               # All parameters and dataset definitions
│   └── analyze_3muvis.py         # Results analysis, figures and statistics
├── pyproject.toml
└── README.md
```

---

## Datasets

### Included

| Dataset | Instances | Features | Views | Classes | Source |
|---|---|---|---|---|---|
| **HTAD** | 1,386 | 52 | 2 (Audio MFCCs, Accelerometer) | 6 | Included in `datasets/` |

HTAD records audio and wrist-accelerometer signals while subjects perform home activities (sweeping, brushing teeth, washing hands, watching TV). Three users, user-balanced.

### Not Included

The following datasets were used in the study but are not distributed here due to intellectual property restrictions. They can be obtained from their original sources:

| Dataset | Instances | Features | Views | Classes | Source |
|---|---|---|---|---|---|
| **HAR70+** | 11,263 | 40 | 2 (Back acc., Thigh acc.) | 7 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/780/har70) |
| **Multiview Digits** | 2,000 | 649 | 2 (KL coefficients, Pixel avg.) | 10 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/72/multiple+features) |
| **Transportation Sensors** | 5,893 | 36 | 2 (Accelerometer, Gyroscope) | 5 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/426/transportation+mode+prediction+using+smartphones) |
| **Berkeley MHAD** | — | 154 | 3 (Kinect, Accelerometer, Microphone) | — | [Berkeley MHAD](http://tele-immersion.citris-uc.org/berkeley_mhad) |

Once downloaded, place each CSV in the `datasets/` directory and ensure the filenames and column names match the entries in `config.yaml`.

---

## Installation

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/<your-username>/3muvis.git
cd 3muvis
uv sync

## Usage

### Run experiments

```bash
# All datasets defined in config.yaml
python src/main.py

# Specific datasets only
python src/main.py --datasets htad

# Skip brute-force search (much faster, useful for development)
python src/main.py --datasets htad --no-brute-force

# Custom config path
python src/main.py --config path/to/config.yaml
```

Results are written to `results/tests_<dataset>_<k>f_<date>.csv` and checkpointed after every iteration so progress is not lost on interruption.

### Analyse results

```bash
python src/analyze_3muvis.py --results_dir results/ --output_dir figures/
```

This produces per-dataset subdirectories containing:

| File | Content |
|---|---|
| `upset_selection.png` | UpSet plot of learner combination frequencies |
| `violin_mcc.png` | MCC distribution by method |
| `violin_accuracy.png` | Accuracy distribution by method |
| `violin_weighted.png` | Weighted precision / recall / F1 distributions |
| `violin_times.png` | Training and prediction time distributions |
| `wilcoxon_table.png` | Wilcoxon signed-rank p-values (3MuViS vs each baseline) |
| `training_time_table.png` | Mean ± 95% CI for training and prediction time |
| `selection_time_table.png` | Mean ± 95% CI for search/selection time |
| `summary_table.csv` | All datasets × methods × metrics in one flat file |

---

## Configuration

All experiment parameters are controlled through `src/config.yaml`. No source code changes are needed to add datasets or adjust hyperparameters.

```yaml
experiment:
  n_iterations: 25
  initial_seed: 13
  metric: "matthews_corrcoef"
  k_folds_base: 5
  k_folds_meta: 5
  test_size: 0.2
  meta_split_size: 0.5
  run_brute_force: true
```

To add a new dataset:

```yaml
datasets:
  my_dataset:
    path: "datasets/my_dataset.csv"
    label_col: "label"
    drop_cols: ["id"]
    views:
      - [0, 20]    # view 0: columns 0–19
      - [20, 40]   # view 1: columns 20–39
    scale: true
```

---

## Reproducibility

All results in the paper can be reproduced with the following settings (already set as defaults in `config.yaml`):

| Parameter | Value |
|---|---|
| Initial seed | 13 (incremented per iteration) |
| Iterations | 25 |
| Train / test split | 80% / 20%, stratified |
| Meta-train split | 50% / 50% of training set |
| CV folds (base) | 5 |
| CV folds (meta) | 5 |
| Preprocessing | MinMaxScaler (fit on train only) |
| Random state (stochastic models) | 13 |

Statistical comparisons use one-sided Wilcoxon signed-rank tests (H₁: 3MuViS > baseline, α = 0.05) over the 25 paired iterations. Effect size is reported as mean ± 95% CI.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

The HTAD dataset is distributed under its original terms. Please cite the original dataset authors if you use it in your own work.

---

## Contact

For questions about the code or methodology, please open a GitHub Issue.

For questions about the paper, contact the corresponding author at:  
`<author-email@institution.edu>` *(to be updated upon publication)*

---

> *This README will be updated with the full citation, DOI, and published results once the manuscript is accepted.*
