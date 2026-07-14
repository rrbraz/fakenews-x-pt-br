"""Validate the integrity and distribution boundary of the public release."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ID_COLUMNS = {"record_id", "tweet_id", "categoria", "label"}
ANNOTATION_COLUMNS = {
    "record_id",
    "tweet_id",
    "llm_polaridade",
    "llm_marcadores_linguisticos",
}
REQUIRED_PREDICTION_COLUMNS = {
    "record_id",
    "tweet_id",
    "categoria",
    "author_seen_in_fit_train",
    "author_seen_in_development",
    "y_true",
    "y_pred",
    "y_proba",
}
FORBIDDEN_PUBLIC_COLUMNS = {
    "tweet_url",
    "full_text",
    "created_at",
    "name",
    "screen_name",
    "username",
    "description",
    "favorite_count",
    "retweet_count",
    "reply_count",
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "verified",
    "embedding",
    "features",
    "llm_raciocinio",
    "text_sha256",
}
FORBIDDEN_PUBLIC_FILES = {
    "dataset.parquet",
    "dataset.csv",
    "dataset_unique.parquet",
    "dataset_unique.csv",
    "dataset_expanded.parquet",
    "dataset_processed.parquet",
}


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _validate_checksums(errors: list[str]) -> int:
    checksum_file = ROOT / "metadata" / "artifact_checksums.sha256"
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = ROOT / relative
        _check(path.is_file(), f"Missing checksummed artifact: {relative}", errors)
        if not path.is_file():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _check(actual == expected, f"Checksum mismatch: {relative}", errors)
        checked += 1
    return checked


def _validate_distribution_boundary(errors: list[str]) -> None:
    data_dir = ROOT / "data"
    for name in FORBIDDEN_PUBLIC_FILES:
        _check(not (data_dir / name).exists(), f"Excluded public file exists: data/{name}", errors)

    for path in data_dir.rglob("*"):
        if not path.is_file() or "generated" in path.parts:
            continue
        try:
            if path.suffix == ".csv":
                columns = set(pd.read_csv(path, nrows=1).columns)
            elif path.suffix == ".parquet":
                columns = set(pd.read_parquet(path).columns)
            else:
                continue
        except (OSError, ValueError) as error:
            errors.append(f"Could not inspect {path.relative_to(ROOT)}: {error}")
            continue
        forbidden = columns & FORBIDDEN_PUBLIC_COLUMNS
        _check(
            not forbidden,
            f"Excluded columns in {path.relative_to(ROOT)}: {sorted(forbidden)}",
            errors,
        )


def _validate_data(errors: list[str]) -> None:
    records = pd.read_parquet(ROOT / "data" / "dataset_ids.parquet")
    unique = pd.read_parquet(ROOT / "data" / "dataset_ids_unique.parquet")
    annotations = pd.read_parquet(ROOT / "data" / "llm_annotations.parquet")
    duplicates = pd.read_csv(ROOT / "data" / "integrity" / "duplicate_records.csv")

    _check(set(records.columns) == ID_COLUMNS, "Unexpected dataset_ids columns.", errors)
    _check(len(records) == 660, "Paper ID table must have 660 rows.", errors)
    _check(records["record_id"].nunique() == 660, "record_id must be unique.", errors)
    _check(records["tweet_id"].nunique() == 659, "Expected 659 unique post IDs.", errors)
    _check(set(records["label"]) == {"fake", "true"}, "Unexpected labels.", errors)

    _check(len(unique) == 659, "Deduplicated ID view must have 659 rows.", errors)
    _check(unique["tweet_id"].is_unique, "Deduplicated tweet_id is not unique.", errors)
    _check(ID_COLUMNS.issubset(unique.columns), "Deduplicated ID columns are incomplete.", errors)

    _check(set(annotations.columns) == ANNOTATION_COLUMNS, "Unexpected annotation columns.", errors)
    _check(len(annotations) == 659, "Expected 659 structured LLM annotations.", errors)
    _check(annotations["tweet_id"].is_unique, "Annotation tweet_id is not unique.", errors)
    _check(set(annotations["tweet_id"]) == set(unique["tweet_id"]), "Annotation IDs differ.", errors)

    _check(len(duplicates) == 2, "Expected two source rows in duplicate report.", errors)
    _check(
        set(duplicates.columns)
        == {"record_id", "tweet_id", "categoria", "label", "duplicate_group", "issue"},
        "Unexpected duplicate-report columns.",
        errors,
    )


def _validate_splits(errors: list[str]) -> None:
    temporal = pd.read_csv(ROOT / "data" / "splits" / "temporal_paper.csv")
    lodo = pd.read_csv(ROOT / "data" / "splits" / "lodo_paper.csv")
    expected_columns = {
        "protocol",
        "fold",
        "partition",
        "partition_order",
        "record_id",
        "tweet_id",
        "categoria",
        "label",
    }
    _check(set(temporal.columns) == expected_columns, "Unexpected temporal columns.", errors)
    _check(set(lodo.columns) == expected_columns, "Unexpected LODO columns.", errors)
    temporal_counts = temporal["partition"].value_counts().to_dict()
    _check(
        temporal_counts == {"train": 476, "test": 99, "validation": 85},
        f"Unexpected temporal split counts: {temporal_counts}",
        errors,
    )
    _check(temporal["record_id"].nunique() == 660, "Temporal split misses records.", errors)
    _check(len(lodo) == 3960, "LODO table must have 6 x 660 rows.", errors)
    for fold, group in lodo.groupby("fold"):
        _check(len(group) == 660, f"LODO fold {fold} has the wrong size.", errors)
        _check(group["record_id"].nunique() == 660, f"LODO fold {fold} repeats records.", errors)
        test = group[group["partition"] == "test"]
        _check(test["categoria"].eq(fold).all(), f"LODO membership is inconsistent for {fold}.", errors)


def _validate_results(errors: list[str]) -> None:
    result_dir = ROOT / "results" / "paper"
    metrics = pd.read_csv(result_dir / "metrics.csv")
    prediction_paths = sorted((result_dir / "predictions").glob("*.csv"))
    parameter_paths = sorted((result_dir / "best_params").glob("*.json"))
    records = pd.read_parquet(ROOT / "data" / "dataset_ids.parquet")
    labels = records.set_index("record_id")["label"].map({"fake": 0, "true": 1}).astype(int)

    _check(len(metrics) == 175, "Expected 175 metric rows.", errors)
    _check(len(prediction_paths) == 175, "Expected 175 prediction files.", errors)
    _check(len(parameter_paths) == 175, "Expected 175 best-parameter files.", errors)

    holm = pd.read_csv(result_dir / "posthoc" / "holm_corrections.csv")
    _check(len(holm) == 30, "Expected 15 planned contrasts per protocol.", errors)
    _check(int(holm["reject_at_0.05"].sum()) == 2, "Expected two Holm-significant contrasts.", errors)
    _check(
        int(holm.loc[holm["protocol"] == "temporal", "reject_at_0.05"].sum()) == 0,
        "No temporal contrast should survive Holm correction.",
        errors,
    )
    correlation = pd.read_csv(result_dir / "posthoc" / "correlation_lodo_temporal.csv")
    _check(correlation["n_models"].eq(25).all(), "Correlations must use 25 configurations.", errors)
    sensitivity = pd.read_csv(result_dir / "author_overlap" / "sensitivity_summary.csv")
    _check(len(sensitivity) == 100, "Unexpected author-overlap summary size.", errors)

    for path in prediction_paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        forbidden = set(frame.columns) & FORBIDDEN_PUBLIC_COLUMNS
        _check(not missing, f"{path.name} is missing columns: {sorted(missing)}", errors)
        _check(not forbidden, f"{path.name} has excluded columns: {sorted(forbidden)}", errors)
        if missing:
            continue
        _check(frame["record_id"].is_unique, f"Duplicate record IDs in {path.name}.", errors)
        expected = frame["record_id"].map(labels)
        _check(expected.notna().all(), f"Unknown record ID in {path.name}.", errors)
        _check(
            np.array_equal(expected.to_numpy(dtype=int), frame["y_true"].to_numpy(dtype=int)),
            f"Label mismatch in {path.name}.",
            errors,
        )
        _check(frame["y_proba"].between(0, 1).all(), f"Invalid probability in {path.name}.", errors)

    for path in parameter_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _check(payload.get("n_trials_completed") == 200, f"Non-200-trial file: {path.name}", errors)
        _check(payload.get("seeds_per_trial") == 3, f"Non-three-seed file: {path.name}", errors)


def _validate_license_scope(errors: list[str]) -> None:
    required = [
        ROOT / "LICENSE",
        ROOT / "LICENSE.md",
        ROOT / "NOTICE",
        ROOT / "data" / "LICENSE.md",
        ROOT / "THIRD_PARTY_DATA_NOTICE.md",
    ]
    for path in required:
        _check(path.is_file(), f"Missing license document: {path.relative_to(ROOT)}", errors)


def _validate_no_embedded_secrets(errors: list[str]) -> None:
    pattern = re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}"
    )
    for base in (ROOT / "scripts", ROOT / "src", ROOT / "metadata"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                _check(not pattern.search(content), f"Possible embedded credential: {path}", errors)


def main() -> int:
    errors: list[str] = []
    checked = _validate_checksums(errors)
    _validate_distribution_boundary(errors)
    _validate_data(errors)
    _validate_splits(errors)
    _validate_results(errors)
    _validate_license_scope(errors)
    _validate_no_embedded_secrets(errors)
    if errors:
        print("Release validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Release validation passed ({checked} checksummed artifacts).")
    print("Public data: 660 ID-only records, 659 unique post IDs, 659 structured annotations.")
    print("Results: 25 configurations, 175 prediction files and parameter files.")
    print("Distribution boundary: no hydrated X content or row-level embeddings detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
