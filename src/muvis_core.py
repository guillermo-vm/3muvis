"""
muvis_core.py
─────────────
3MuViS algorithm: base-learner search, meta-learner search, and model catalog.
All functions are pure (no global state) and fully compatible with
Python 3.12 / scikit-learn 1.6.x.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer_names
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from multiviewstacking import MultiViewStacking

# ─────────────────────────────────────────────────────────────────────────────
# Model catalog
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CATALOG: dict[str, type] = {
    "LinearDiscriminantAnalysis":    LinearDiscriminantAnalysis,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "KNeighborsClassifier":          KNeighborsClassifier,
    "NearestCentroid":               NearestCentroid,
    "SVC":                           SVC,
    "DecisionTreeClassifier":        DecisionTreeClassifier,
    "ExtraTreesClassifier":          ExtraTreesClassifier,
    "RandomForestClassifier":        RandomForestClassifier,
    "SGDClassifier":                 SGDClassifier,
    "GaussianNB":                    GaussianNB,
    "MultinomialNB":                 MultinomialNB,
    "BernoulliNB":                   BernoulliNB,
}

# Models that require a random_state for reproducibility
_RANDOM_STATE_MODELS: frozenset[str] = frozenset({
    "HistGradientBoostingClassifier",
    "SVC",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "RandomForestClassifier",
    "SGDClassifier",
})

RANDOM_STATE: int = 13


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model_instance(model_name: str, **kwargs):
    """
    Instantiate a model by name from MODEL_CATALOG.

    Enforces:
      - SVC always has probability=True  (needed for MultiViewStacking)
      - SGDClassifier always uses loss='log_loss'  (probabilistic output)
      - Models in _RANDOM_STATE_MODELS receive random_state=RANDOM_STATE
    """
    if model_name not in MODEL_CATALOG:
        raise ValueError(f"Model '{model_name}' is not in MODEL_CATALOG. "
                         f"Available: {list(MODEL_CATALOG.keys())}")

    if model_name == "SVC":
        kwargs.setdefault("probability", True)
    elif model_name == "SGDClassifier":
        kwargs.setdefault("loss", "log_loss")

    instance = MODEL_CATALOG[model_name](**kwargs)

    if model_name in _RANDOM_STATE_MODELS:
        instance.set_params(random_state=RANDOM_STATE)

    return instance


# ─────────────────────────────────────────────────────────────────────────────
# Base-learner search
# ─────────────────────────────────────────────────────────────────────────────

def find_baselearner(
    base_train_x: np.ndarray,
    base_train_y: np.ndarray,
    *,
    views: list[list[int]],
    k_folds: int = 5,
    metric: str = get_scorer_names()[0],
) -> tuple[list[tuple[list[int], str]], list[dict[str, float]]]:
    """
    For each view, evaluate every model in MODEL_CATALOG via k-fold CV
    and return the best-performing model per view.

    Parameters
    ----------
    base_train_x : feature matrix (n_samples, n_features)
    base_train_y : label vector   (n_samples,)
    views        : list of column-index lists, one per view
    k_folds      : number of CV folds
    metric       : sklearn scorer name to optimise

    Returns
    -------
    selected : list of (view_indices, best_model_name) tuples
    all_results : list of dicts {model_name: mean_cv_score} per view,
                  sorted descending — useful for logging / analysis
    """
    all_results: list[dict[str, float]] = []

    for view_idx, view in enumerate(views):
        X_view = base_train_x[:, view]
        scores: dict[str, float] = {}

        for model_name in MODEL_CATALOG:
            try:
                cv_scores = cross_val_score(
                    get_model_instance(model_name),
                    X_view, base_train_y,
                    cv=k_folds, scoring=metric,
                )
                scores[model_name] = float(cv_scores.mean())
            except Exception:
                print(f"model {model_name} incompatible with view {view_idx}")
                pass   # model incompatible with this view 

        scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
        all_results.append(scores)

        best_name  = next(iter(scores))
        best_score = scores[best_name]
        print(f"  [base view {view_idx}] {best_name}  {metric}={best_score:.4f}")

    selected = [
        (view, list(results.keys())[0])
        for view, results in zip(views, all_results)
    ]
    return selected, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Meta-learner search
# ─────────────────────────────────────────────────────────────────────────────

def find_metalearner(
    meta_train_x: np.ndarray,
    meta_train_y: np.ndarray,
    *,
    base_learner_names: list[str],
    views: list[list[int]],
    k_folds: int = 5,
    metric: str = get_scorer_names()[0],
) -> dict[str, float]:
    """
    Evaluate every model in MODEL_CATALOG as a meta-learner on top of the
    supplied base learners, using k-fold CV.

    Parameters
    ----------
    meta_train_x      : feature matrix for meta-learner training split
    meta_train_y      : labels for meta-learner training split
    base_learner_names: list of model names (one per view) selected by find_baselearner
    views             : list of column-index lists, one per view
    k_folds           : number of CV folds
    metric            : sklearn scorer name to optimise

    Returns
    -------
    dict {meta_model_name: mean_cv_score}, sorted descending
    """
    results_meta: dict[str, float] = {}

    for model_name in MODEL_CATALOG:
        try:
            mvs = MultiViewStacking(
                views_indices=views,
                first_level_learners=[get_model_instance(m) for m in base_learner_names],
                meta_learner=get_model_instance(model_name),
            )
            cv_scores = cross_val_score(
                mvs, meta_train_x, meta_train_y,
                cv=k_folds, scoring=metric,
            )
            results_meta[model_name] = float(cv_scores.mean())
        except Exception:
            pass

    return dict(sorted(results_meta.items(), key=lambda kv: kv[1], reverse=True))
