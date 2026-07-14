"""Reproduce the 25 TF-IDF and BERT-embedding pipelines with Optuna.

This standalone runner replaces the experiment notebooks with a script that
uses a locally generated feature matrix, the paper protocol, saved seeds, and
per-sample prediction outputs.

What this script does

  - Builds the temporal and LODO splits described in the paper.
  - For each (feature_set, classifier_type, split, fold), runs Optuna
    with 200 trials. Each trial trains the classifier with N random
    seeds (default 3 — same as the deep-model run) and reports the mean
    val F1, so Optuna picks configurations that are stable across seeds
    rather than lucky ones.
  - The final retrain uses the winning hyperparameters and trains N
    models with seeds 42..42+N-1. Test predictions are the ensemble of
    those N models (mean class probability, threshold 0.5).
  - Feature extraction is per-split to avoid leakage:
      * TF-IDF features: fit ``TfidfVectorizer`` on train texts, transform
        val and test; then ``SelectKBest(chi2, k=1000)`` fitted on train.
      * BERT embeddings: generated locally under
        ``data/generated/dataset_processed.parquet``; no fitted transform is
        applied because BERT is frozen.
  - SQLite + WAL + ``spawn`` workers for safe parallelism.

Feature/model matrix

  TF-IDF: LR, SVC, RF, GB, NB, DT, KNN, LGBM, XGB   (9 classifiers)
  BERT:   LR, SVC, RF,     DT, KNN, LGBM, XGB       (7 classifiers — no NB
                                                      because BERT
                                                      embeddings have
                                                      negative values; no
                                                      GB to keep runtime
                                                      reasonable)
  BERT + numerical metadata: LR, SVC, RF             (7 engagement/profile
                                                      counts)
  BERT + LLM features: LR, SVC, RF                    (3 polarity + 29 markers)
  BERT + all additional features: LR, SVC, RF         (7 numerical + 32 LLM)

Outputs (under ``results/rerun/``)

  predictions/{split}__{feature}-{model}__{fold}.csv
  best_params/{split}__{feature}-{model}__{fold}.json
  metrics.csv
  optuna.db

Usage

  python scripts/reexecute_baselines_with_optuna.py all
  python scripts/reexecute_baselines_with_optuna.py all --workers 4 --seeds-per-trial 3
  python scripts/reexecute_baselines_with_optuna.py tfidf --n-trials 20  # smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.embedding_utils import extract_tfidf_features, select_features  # noqa: E402
from src.utils.feature_scaling import (apply_numerical_scaler,  # noqa: E402
                                       fit_numerical_scaler)
from src.utils.sklearn_utils import get_classifier  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
DATA_PATH = "data/generated/dataset_processed.parquet"
OUT_DIR = Path("results/rerun")
PRED_DIR = OUT_DIR / "predictions"
PARAMS_DIR = OUT_DIR / "best_params"
METRICS_PATH = OUT_DIR / "metrics.csv"
STUDIES_DB_PATH = OUT_DIR / "optuna.db"
for d in (OUT_DIR, PRED_DIR, PARAMS_DIR):
    d.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

TFIDF_CLASSIFIERS = [
    "LogisticRegression", "SVC", "RandomForest", "GradientBoosting",
    "MultinomialNB", "DecisionTree", "KNeighbors", "LightGBM", "XGBoost",
]
BERT_CLASSIFIERS = [
    "LogisticRegression", "SVC", "RandomForest",
    "DecisionTree", "KNeighbors", "LightGBM", "XGBoost",
]
# Matched feature-block baselines, restricted to the three strongest
# BERT-embedding pipelines. Together they form a factorial comparison:
# text only, text + numerical metadata, text + LLM features, and text + both.
BERT_AUGMENTED_CLASSIFIERS = [
    "LogisticRegression", "SVC", "RandomForest",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seeds_tuple(n: int) -> tuple:
    return tuple(SEED + i for i in range(max(1, int(n))))


def compute_metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
    }
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                  zero_division=0)
    if len(f) > 1:
        out["f1_class0"] = float(f[0])
        out["f1_class1"] = float(f[1])
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["auc_roc"] = float("nan")
    return out


def save_predictions(name, split, fold, metadata, y_true, y_pred, y_proba):
    df = metadata.reset_index(drop=True).copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba
    safe = name.replace("/", "_")
    df.to_csv(PRED_DIR / f"{split}__{safe}__{fold}.csv", index=False)


def predictions_path(name, split, fold):
    safe = name.replace("/", "_")
    return PRED_DIR / f"{split}__{safe}__{fold}.csv"


def best_params_path(name, split, fold):
    safe = name.replace("/", "_")
    return PARAMS_DIR / f"{split}__{safe}__{fold}.json"


def append_metrics(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if METRICS_PATH.exists():
        existing = pd.read_csv(METRICS_PATH)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(METRICS_PATH, index=False)


# ---------------------------------------------------------------------------
# Split + feature extraction (per worker, per task)
# ---------------------------------------------------------------------------
def _build_split_indices(df, split, fold):
    """Return DataFrame slices (train_df, val_df, test_df) for a (split, fold)."""
    if split == "temporal":
        df = df.copy()
        df["created_at"] = pd.to_datetime(
            df["created_at"], errors="coerce", format="mixed", utc=True
        )
        sorted_df = df.sort_values(by=["created_at"])
        train_val_df, test_df = train_test_split(sorted_df, test_size=0.15, shuffle=False)
        train_df, val_df = train_test_split(train_val_df, test_size=0.15, shuffle=False)
    elif split == "lodo":
        train_val_df = df[df["categoria"] != fold]
        test_df = df[df["categoria"] == fold]
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15, random_state=SEED,
            stratify=train_val_df["label_encoded"],
        )
    else:
        raise ValueError(f"Unknown split: {split!r}")
    return train_df, val_df, test_df


def _build_features(feature_set, train_df, val_df, test_df):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test).

    Feature extraction is per-split so the test-set vocabulary or selector
    never leaks into training.
    """
    y_train = np.asarray(train_df["label_encoded"].tolist())
    y_val = np.asarray(val_df["label_encoded"].tolist())
    y_test = np.asarray(test_df["label_encoded"].tolist())

    if feature_set == "tfidf":
        Xt, Xv, Xte, _ = extract_tfidf_features(
            train_df["full_text"].tolist(),
            val_df["full_text"].tolist(),
            test_df["full_text"].tolist(),
            n_gram_range=(1, 3),
        )
        Xt, Xv, Xte, _ = select_features(Xt, y_train, Xv, Xte, k=1000)
        return Xt, Xv, Xte, y_train, y_val, y_test
    if feature_set == "bert":
        Xt = np.asarray(train_df["embedding"].tolist())
        Xv = np.asarray(val_df["embedding"].tolist())
        Xte = np.asarray(test_df["embedding"].tolist())
        return Xt, Xv, Xte, y_train, y_val, y_test
    if feature_set in ("bert_num", "bert_llm", "bert_meta"):
        # Numerical metadata is z-scored with a scaler fitted on the training
        # split only. The LLM-only representation never reads that block.
        scaler = None
        if feature_set != "bert_llm":
            scaler = fit_numerical_scaler(train_df)
        parts = []
        for frame in (train_df, val_df, test_df):
            embeddings = np.asarray(frame["embedding"].tolist(), dtype=np.float64)
            additional = np.asarray(frame["features"].tolist(), dtype=np.float64)
            if feature_set == "bert_llm":
                selected = additional[:, 7:]
            else:
                scaled = apply_numerical_scaler(frame, scaler)
                additional = np.asarray(scaled["features"].tolist(), dtype=np.float64)
                if feature_set == "bert_num":
                    selected = additional[:, :7]
                else:
                    selected = additional
            parts.append(np.hstack([embeddings, selected]))
        Xt, Xv, Xte = parts
        return Xt, Xv, Xte, y_train, y_val, y_test
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


# ---------------------------------------------------------------------------
# Objective + train/inference
# ---------------------------------------------------------------------------
def make_objective(classifier_type, X_train, y_train, X_val, y_val, seeds=(SEED,)):
    """Build an Optuna objective. Average val_f1 across the seeds tuple."""
    def objective(trial):
        scores = []
        for seed in seeds:
            try:
                model = get_classifier(trial, classifier_type, seed=seed)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                scores.append(float(f1_score(y_val, preds, zero_division=0)))
            except Exception as e:
                raise optuna.exceptions.TrialPruned(f"Trial failed: {e}")
        return float(np.mean(scores))
    return objective


def _build_with_params(classifier_type, params, seed):
    """Instantiate a classifier with explicit hyperparameters (post-search).

    Optuna's ``best_params`` is a flat dict; we need to feed those into the
    same constructor logic that ``get_classifier`` uses with a trial.
    Cheapest way: a fake "fixed-trial" that just returns each param.
    """
    fixed = optuna.trial.FixedTrial(params)
    return get_classifier(fixed, classifier_type, seed=seed)


def _ensemble_predict_proba(models, X):
    """Average predicted probabilities across an ensemble of fitted models."""
    sum_proba = None
    for m in models:
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)[:, 1]
        else:
            # Last-resort: decision_function → sigmoid, only for SVC w/o probability.
            scores = m.decision_function(X)
            p = 1.0 / (1.0 + np.exp(-scores))
        sum_proba = p if sum_proba is None else sum_proba + p
    proba = sum_proba / len(models)
    preds = (proba >= 0.5).astype(int)
    return preds, proba


def optimize_and_train(feature_set, classifier_type, split, fold,
                       X_train, y_train, X_val, y_val, X_test, y_test,
                       test_metadata, n_trials, force=False, seeds_per_trial=3):
    name = f"{feature_set}-{classifier_type}"
    pred_file = predictions_path(name, split, fold)
    if pred_file.exists() and not force:
        print(f"  {name:>40s}: SKIP (predictions cached at {pred_file.name})")
        return None

    seeds = _seeds_tuple(seeds_per_trial)
    study_name = f"{split}__{name}__{fold}"
    if seeds_per_trial > 1:
        study_name = f"{study_name}__seeds{seeds_per_trial}"
    storage = f"sqlite:///{STUDIES_DB_PATH}"
    sampler = TPESampler(seed=SEED)

    study = optuna.create_study(
        study_name=study_name, storage=storage, sampler=sampler,
        direction="maximize", load_if_exists=True,
    )

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, n_trials - completed)
    t0 = time.time()
    if remaining > 0:
        study.optimize(
            make_objective(classifier_type, X_train, y_train, X_val, y_val, seeds=seeds),
            n_trials=remaining,
        )
    best_params = study.best_params

    # Final retrain — one model per seed, ensemble at the end.
    models = []
    for seed in seeds:
        m = _build_with_params(classifier_type, best_params, seed)
        m.fit(X_train, y_train)
        models.append(m)
    y_pred, y_proba = _ensemble_predict_proba(models, X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    save_predictions(name, split, fold, test_metadata, y_test, y_pred, y_proba)
    with best_params_path(name, split, fold).open("w") as f:
        json.dump({"split": split, "feature_set": feature_set,
                   "model": classifier_type, "fold": fold,
                   "n_trials_completed": completed + remaining,
                   "seeds_per_trial": int(seeds_per_trial),
                   "seeds": list(seeds),
                   "best_val_f1": float(study.best_value),
                   "best_params": best_params}, f, indent=2)

    elapsed = time.time() - t0
    seeds_tag = f" seeds={seeds_per_trial}" if seeds_per_trial > 1 else ""
    print(f"  {name:>40s}: acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f} "
          f"({elapsed:.0f}s, {remaining}/{n_trials} new trials{seeds_tag})")
    return dict(split=split, feature_set=feature_set, model=classifier_type, fold=fold,
                n_test=int(len(y_test)), seconds=round(elapsed, 1),
                n_trials=n_trials, seeds_per_trial=int(seeds_per_trial),
                **metrics)


# ---------------------------------------------------------------------------
# Parallelism plumbing
# ---------------------------------------------------------------------------
def _init_sqlite_wal(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()
    finally:
        conn.close()
    storage = f"sqlite:///{db_path}"
    optuna.create_study(study_name="__bootstrap__", storage=storage,
                        direction="maximize", load_if_exists=True)


def _worker_init(intra_op_threads: int = 2) -> None:
    torch.set_num_threads(intra_op_threads)
    torch.set_num_interop_threads(1)


def _worker_run(task):
    feature_set, classifier_type, split, fold, n_trials, force, seeds_per_trial = task
    df = pd.read_parquet(DATA_PATH)
    train_df, val_df, test_df = _build_split_indices(df, split, fold)
    Xt, Xv, Xte, yt, yv, yte = _build_features(feature_set, train_df, val_df, test_df)
    fit_authors = set(train_df["screen_name"].fillna("").str.casefold())
    development_authors = fit_authors | set(
        val_df["screen_name"].fillna("").str.casefold()
    )
    test_authors = test_df["screen_name"].fillna("").str.casefold()
    test_metadata = test_df[["record_id", "tweet_id", "categoria"]].copy()
    test_metadata["author_seen_in_fit_train"] = test_authors.isin(fit_authors)
    test_metadata["author_seen_in_development"] = test_authors.isin(
        development_authors
    )
    return optimize_and_train(
        feature_set, classifier_type, split, fold,
        Xt, yt, Xv, yv, Xte, yte,
        test_metadata, n_trials, force=force, seeds_per_trial=seeds_per_trial,
    )


def _build_tasks(mode, n_trials, force, seeds_per_trial):
    """Enumerate (feature_set, classifier, split, fold) combos to run."""
    df = pd.read_parquet(DATA_PATH)
    domains = df["categoria"].unique().tolist()
    feature_sets = []
    if mode in ("tfidf", "all"):
        feature_sets.append(("tfidf", TFIDF_CLASSIFIERS))
    if mode in ("bert", "all"):
        feature_sets.append(("bert", BERT_CLASSIFIERS))
    if mode == "all":
        feature_sets.extend([
            ("bert_num", BERT_AUGMENTED_CLASSIFIERS),
            ("bert_llm", BERT_AUGMENTED_CLASSIFIERS),
            ("bert_meta", BERT_AUGMENTED_CLASSIFIERS),
        ])
    if mode in ("bert_num", "bert_llm", "bert_meta"):
        feature_sets.append((mode, BERT_AUGMENTED_CLASSIFIERS))
    if mode == "bert_ablation":
        feature_sets.extend([
            ("bert_num", BERT_AUGMENTED_CLASSIFIERS),
            ("bert_llm", BERT_AUGMENTED_CLASSIFIERS),
        ])

    tasks = []
    for feature_set, classifiers in feature_sets:
        for clf in classifiers:
            tasks.append((feature_set, clf, "temporal", "all",
                          n_trials, force, seeds_per_trial))
            for d in domains:
                tasks.append((feature_set, clf, "lodo", d,
                              n_trials, force, seeds_per_trial))
    return tasks


def _execute(tasks, workers):
    if workers <= 1:
        for task in tasks:
            row = _worker_run(task)
            if row is not None:
                append_metrics([row])
        return
    intra_op = max(1, 8 // workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(intra_op,)) as pool:
        for row in pool.imap_unordered(_worker_run, tasks):
            if row is not None:
                append_metrics([row])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "mode",
        choices=["tfidf", "bert", "bert_num", "bert_llm", "bert_meta",
                 "bert_ablation", "all"],
    )
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Optuna trials per (model, fold). Default 200.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if predictions are already cached.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes (default 1 = serial).")
    parser.add_argument("--seeds-per-trial", type=int, default=3,
                        help="Random seeds averaged per trial (default 3, as in the paper).")
    args = parser.parse_args()

    print(f"Outputs: {OUT_DIR.resolve()}")
    print(f"Optuna DB: {STUDIES_DB_PATH.resolve()}")
    _init_sqlite_wal(STUDIES_DB_PATH)

    tasks = _build_tasks(args.mode, args.n_trials, args.force, args.seeds_per_trial)
    print(f"[{args.mode.upper()}] tasks={len(tasks)} workers={args.workers} "
          f"seeds_per_trial={args.seeds_per_trial}")
    t0 = time.time()
    _execute(tasks, args.workers)
    print(f"[{args.mode.upper()}] DONE in {time.time() - t0:.1f}s")
