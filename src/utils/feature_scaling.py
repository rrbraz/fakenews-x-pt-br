"""Z-score scaling for numerical features, applied per-split to avoid leakage.

Background
----------
The `features` column in `data/dataset_processed.parquet` is a concatenation
of:

- the first `NUMERICAL_FEATURES_COUNT` (= 7) values: numerical engagement
  and profile metrics (favorite_count, retweet_count, reply_count,
  followers_count, friends_count, statuses_count, favourites_count) after a
  `log1p` transform — they are stored RAW (post-log1p, but pre-z-score);
- the remaining values: one-hot polarity and multi-hot LLM linguistic
  markers — these are categorical and need no scaling.

Z-score depends on dataset statistics (mean, std). Computing it over the
full dataset before splitting would leak the test distribution into the
training data. The functions here let you fit a `StandardScaler` on the
training split only and apply it to validation and test splits.

Usage
-----

    from src.utils.feature_scaling import (
        fit_numerical_scaler, apply_numerical_scaler
    )

    train_df, val_df, test_df = ...  # any per-protocol split
    scaler = fit_numerical_scaler(train_df)
    train_df = apply_numerical_scaler(train_df, scaler)
    val_df   = apply_numerical_scaler(val_df, scaler)
    test_df  = apply_numerical_scaler(test_df, scaler)

This must happen before constructing `FakenewsDataset` from each frame.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

NUMERICAL_FEATURES_COUNT = 7


def fit_numerical_scaler(train_df, n_numerical: int = NUMERICAL_FEATURES_COUNT) -> StandardScaler:
    """Fit a StandardScaler on the numerical slice of the train features."""
    train_features = np.array(train_df["features"].tolist(), dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(train_features[:, :n_numerical])
    return scaler


def apply_numerical_scaler(df, scaler: StandardScaler, n_numerical: int = NUMERICAL_FEATURES_COUNT):
    """Return a copy of `df` with the numerical slice of `features` scaled.

    The categorical tail (indices `n_numerical:` of every row) is preserved.
    """
    df = df.copy()
    features_arr = np.array(df["features"].tolist(), dtype=np.float64)
    features_arr[:, :n_numerical] = scaler.transform(features_arr[:, :n_numerical])
    df["features"] = list(features_arr)
    return df
