"""Join hydrated X content with released labels and structured annotations."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hydrated",
        type=Path,
        default=ROOT / "data" / "generated" / "hydrated_posts.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "generated" / "dataset_expanded.parquet",
    )
    args = parser.parse_args()

    identifiers = pd.read_parquet(ROOT / "data" / "dataset_ids.parquet")
    annotations = pd.read_parquet(ROOT / "data" / "llm_annotations.parquet")
    hydrated = pd.read_parquet(args.hydrated)
    for frame in (identifiers, annotations, hydrated):
        frame["tweet_id"] = frame["tweet_id"].astype(str)

    annotations = annotations.drop(columns="record_id", errors="ignore")
    merged = identifiers.merge(hydrated, on="tweet_id", how="inner", validate="many_to_one")
    merged = merged.merge(annotations, on="tweet_id", how="left", validate="many_to_one")
    merged["tweet_url"] = "https://x.com/i/web/status/" + merged["tweet_id"]
    merged["llm_marcadores_linguisticos"] = merged[
        "llm_marcadores_linguisticos"
    ].map(lambda value: ",".join(value) if isinstance(value, list) else value)
    merged["llm_raciocinio"] = pd.NA
    for column in (
        "favorite_count",
        "retweet_count",
        "reply_count",
        "followers_count",
        "friends_count",
        "statuses_count",
        "favourites_count",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(
        f"Prepared {len(merged)} rows from {identifiers['tweet_id'].nunique()} released IDs."
    )
    print("Counts and text are current hydration values, not the historical paper snapshot.")


if __name__ == "__main__":
    main()
