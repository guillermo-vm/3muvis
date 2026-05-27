"""
experiment.py
─────────────
Single-iteration experiment runner.
Executes all four methods (3MuViS, Random Choice, RF-MVS, RandOpt)
plus optionally brute force, and returns a list of result dicts
ready to be appended to a DataFrame.

Compatible with Python 3.12 / scikit-learn 1.6.x.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import classification_report, get_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from multiviewstacking import MultiViewStacking

from muvis_core import (
    MODEL_CATALOG,
    find_baselearner,
    find_metalearner,
    get_model_instance,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(
    method: str,
    y_test: np.ndarray,
    y_preds: np.ndarray,
    metric: str,
    metric_func,
    encoder,
    views: list[list[int]],
    base_learner_names: list[str],
    meta_learner_name: str,
    fit_time: float,
    predict_time: float,
    search_time: float = 0.0,
    randopt_search_time: float = 0.0,
    brute_force_search_time: float = 0.0
) -> dict:
    
    """Assemble the flat result dict that maps to one CSV row."""
    score      = metric_func(y_test, y_preds)
    clf_report = classification_report(
        y_test, y_preds,
        target_names=encoder.classes_,
        output_dict=True,
    )
    accuracy = clf_report.pop("accuracy")

    report: dict = {"Method": method}
    report[metric]              = score
    report["3MuViS search time"] = search_time
    report["Randopt search time"] = randopt_search_time
    report["Brute Force search time"] = brute_force_search_time

    for i, (view, bl) in enumerate(zip(views, base_learner_names)):
        report[f"view {i} base learner {view}"] = bl

    report["Meta learner"] = meta_learner_name
    report["accuracy"]     = accuracy
    report.update({
        f"{cat}_{m}": v
        for cat, metrics in clf_report.items()
        for m, v in metrics.items()
    })
    report["Fit time"]     = fit_time
    report["Predict time"] = predict_time
    return report


def _fit_predict(model, X_train, y_train, X_test):
    """Fit and predict, returning (y_preds, fit_time, predict_time)."""
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    y_preds = model.predict(X_test)
    predict_time = time.time() - t0

    return y_preds, fit_time, predict_time


# ─────────────────────────────────────────────────────────────────────────────
# Per-method runners
# ─────────────────────────────────────────────────────────────────────────────

def run_3muvis(
    X_train, y_train, X_test, y_test,
    views, metric, metric_func, encoder,
    k_folds_base, k_folds_meta, seed,

) -> dict:
    base_x, meta_x, base_y, meta_y = train_test_split(
        X_train, y_train, test_size=0.5, random_state=seed,
    )

    t0 = time.time()
    base_results, _ = find_baselearner(
        base_x, base_y,
        views=views, k_folds=k_folds_base, metric=metric,
    )

    base_names = [name for _, name in base_results]

    meta_results = find_metalearner(
        meta_x, meta_y,
        base_learner_names=base_names,
        views=views, k_folds=k_folds_meta, metric=metric,
    )
    search_time = time.time() - t0

    meta_name = next(iter(meta_results))
    model = MultiViewStacking(
        views_indices=views,
        first_level_learners=[get_model_instance(n) for n in base_names],
        meta_learner=get_model_instance(meta_name),
        k=k_folds_meta,
    )
    y_preds, fit_time, predict_time = _fit_predict(model, X_train, y_train, X_test)

    print(f"  [3MuViS] meta={meta_name}  base={base_names}  "
          f"search={search_time:.1f}s")

    return _build_report(
        method="3 MuVis",
        y_test=y_test, y_preds=y_preds,
        metric=metric, metric_func=metric_func,
        encoder=encoder, views=views,
        base_learner_names=base_names,
        meta_learner_name=meta_name,
        fit_time=fit_time, predict_time=predict_time,
        search_time=search_time,
    )


def run_random_choice(
    X_train, y_train, X_test, y_test,
    views, metric, metric_func, encoder, seed,
) -> dict:
    rng    = np.random.default_rng(seed)
    names  = list(MODEL_CATALOG.keys())
    n      = len(views) + 1
    chosen = rng.choice(names, size=n, replace=True)

    meta_name  = chosen[0]
    base_names = list(chosen[1:])

    
    model = MultiViewStacking(
        views_indices=views,
        first_level_learners=[get_model_instance(m) for m in base_names],
        meta_learner=get_model_instance(meta_name),
    )

    y_preds, fit_time, predict_time = _fit_predict(model, X_train, y_train, X_test)
    print(f"  [Random Choice] meta = {meta_name} base = {base_names}    ")

    return _build_report(
        method="random choice",
        y_test=y_test, y_preds=y_preds,
        metric=metric, metric_func=metric_func,
        encoder=encoder, views=views,
        base_learner_names=base_names,
        meta_learner_name=meta_name,
        fit_time=fit_time, predict_time=predict_time,
    )


def run_rf_mvs(
    X_train, y_train, X_test, y_test,
    views, metric, metric_func, encoder,
) -> dict:
    base_names = ["RandomForestClassifier"] * len(views)
    meta_name  = "RandomForestClassifier"

    model = MultiViewStacking(
        views_indices=views,
        first_level_learners=[get_model_instance(m) for m in base_names],
        meta_learner=get_model_instance(meta_name),
    )
    y_preds, fit_time, predict_time = _fit_predict(model, X_train, y_train, X_test)
    print(f"  [Random Forest MVS] meta = {meta_name} base = {base_names}    ")
    return _build_report(
        method="random forest mvs",
        y_test=y_test, y_preds=y_preds,
        metric=metric, metric_func=metric_func,
        encoder=encoder, views=views,
        base_learner_names=base_names,
        meta_learner_name=meta_name,
        fit_time=fit_time, predict_time=predict_time,
    )


def run_randopt(
    X_train, y_train, X_test, y_test,
    views, metric, metric_func, encoder,
    k_folds_meta, seed,
) -> dict:
    rng    = np.random.default_rng(seed)
    names  = list(MODEL_CATALOG.keys())
    chosen = rng.choice(names, size=len(views), replace=True)
    base_names = list(chosen)

    # Split for meta search (same ratio as 3MuViS)
    _, meta_x, _, meta_y = train_test_split(
        X_train, y_train, test_size=0.5, random_state=seed,
    )
    t0 = time.time()
    meta_results = find_metalearner(
        meta_x, meta_y,
        base_learner_names=base_names,
        views=views, k_folds=k_folds_meta, metric=metric,
    )
    randopt_search_time = time.time() - t0

    

    meta_name = next(iter(meta_results))

    print(f"  [RandOpt] meta = {meta_name} base = {base_names} search: {randopt_search_time}s")
    model = MultiViewStacking(
        views_indices=views,
        first_level_learners=[get_model_instance(m) for m in base_names],
        meta_learner=get_model_instance(meta_name),
        k=k_folds_meta,
    )
    y_preds, fit_time, predict_time = _fit_predict(model, X_train, y_train, X_test)

    return _build_report(
        method="Randopt",
        y_test=y_test, y_preds=y_preds,
        metric=metric, metric_func=metric_func,
        encoder=encoder, views=views,
        base_learner_names=base_names,
        meta_learner_name=meta_name,
        fit_time=fit_time, predict_time=predict_time,
        randopt_search_time=randopt_search_time,
    )


def run_brute_force(
    X: np.ndarray, y: np.ndarray,
    views, metric, metric_func, encoder,
    seed, k_folds,
) -> dict:
    """
    Exhaustive search over all model combinations.
    Uses a single fixed train/test split (seed=initial_seed).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    catalog_keys = list(MODEL_CATALOG.keys())
    n_models     = len(views) + 1
    n_alg        = len(catalog_keys)
    n_combs      = n_alg ** n_models

    best_score  = -np.inf
    best_base   = None
    best_meta   = None
    untrained   = 0

    print(f"  [brute force] {n_combs} combinations to evaluate...")
    t0 = time.time()

    for i in range(n_combs):
        print(f"\r  [brute force] {i} combinations evaluated", end="", flush=True)
        indices        = [int(i / (n_alg ** p) % n_alg) for p in range(n_models)]
        meta_name_i    = catalog_keys[indices[0]]
        base_names_i   = [catalog_keys[j] for j in indices[1:]]

        try:
            model_i = MultiViewStacking(
                views_indices=views,
                first_level_learners=[get_model_instance(m) for m in base_names_i],
                meta_learner=get_model_instance(meta_name_i),
                k=k_folds,
            )
            model_i.fit(X_train, y_train)
            score_i = metric_func(y_test, model_i.predict(X_test))

            if score_i > best_score:
                best_score  = score_i
                best_base   = base_names_i
                best_meta   = meta_name_i
        except Exception:
            untrained += 1

    brute_search_time = time.time() - t0
    print(f"  [brute force] done in {brute_search_time:.1f}s  "
          f"untrained={untrained}  best={best_meta}/{best_base}")

    # Train best combination
    best_model = MultiViewStacking(
        views_indices=views,
        first_level_learners=[get_model_instance(m) for m in best_base],
        meta_learner=get_model_instance(best_meta),
    )
    y_preds, fit_time, predict_time = _fit_predict(best_model, X_train, y_train, X_test)

    return _build_report(
        method="brute force",
        y_test=y_test, y_preds=y_preds,
        metric=metric, metric_func=metric_func,
        encoder=encoder, views=views,
        base_learner_names=best_base,
        meta_learner_name=best_meta,
        fit_time=fit_time, predict_time=predict_time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full single-iteration runner
# ─────────────────────────────────────────────────────────────────────────────

def run_iteration(
    X: np.ndarray,
    y: np.ndarray,
    views: list[list[int]],
    encoder,
    *,
    seed: int,
    metric: str,
    k_folds_base: int,
    k_folds_meta: int,
    test_size: float,
) -> list[dict]:
    """
    One full train/test iteration: splits, scales, runs all four methods,
    returns a list of result dicts (one per method).
    """
    metric_func = get_scorer(metric)._score_func

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = []

    results.append(run_3muvis(
        X_train, y_train, X_test, y_test,
        views=views, metric=metric, metric_func=metric_func,
        encoder=encoder, k_folds_base=k_folds_base,
        k_folds_meta=k_folds_meta, seed=seed,
    ))

    results.append(run_random_choice(
        X_train, y_train, X_test, y_test,
        views=views, metric=metric, metric_func=metric_func,
        encoder=encoder, seed=seed,
    ))

    results.append(run_rf_mvs(
        X_train, y_train, X_test, y_test,
        views=views, metric=metric, metric_func=metric_func,
        encoder=encoder,
    ))

    results.append(run_randopt(
        X_train, y_train, X_test, y_test,
        views=views, metric=metric, metric_func=metric_func,
        encoder=encoder, k_folds_meta=k_folds_meta, seed=seed,
    ))

    return results
