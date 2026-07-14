"""Hydrate released Post IDs through the official X API v2.

Hydrated content is written only under ``data/generated/``, which is excluded
from version control. The script retrieves up to 100 Posts per request using
app-only Bearer Token authentication.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.x.com/2/tweets"
MAX_BATCH_SIZE = 100


def batched(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def request_batch(
    session: requests.Session,
    post_ids: list[str],
    bearer_token: str,
    max_retries: int,
) -> dict:
    parameters = {
        "ids": ",".join(post_ids),
        "tweet.fields": "author_id,created_at,public_metrics",
        "expansions": "author_id",
        "user.fields": "description,name,public_metrics,username,verified",
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    for attempt in range(max_retries):
        response = session.get(
            API_URL,
            params=parameters,
            headers=headers,
            timeout=60,
        )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == max_retries - 1:
                response.raise_for_status()
            reset = response.headers.get("x-rate-limit-reset")
            if reset and reset.isdigit():
                delay = max(1, min(900, int(reset) - int(time.time()) + 1))
            else:
                delay = min(60, 2 ** attempt)
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("X API request exhausted retries.")


def normalize_response(payload: dict) -> tuple[list[dict], list[dict]]:
    users = {
        user["id"]: user for user in payload.get("includes", {}).get("users", [])
    }
    rows = []
    for post in payload.get("data", []):
        author = users.get(post.get("author_id"), {})
        post_metrics = post.get("public_metrics", {})
        user_metrics = author.get("public_metrics", {})
        rows.append(
            {
                "tweet_id": str(post["id"]),
                "full_text": post.get("text"),
                "created_at": post.get("created_at"),
                "author_id": post.get("author_id"),
                "name": author.get("name"),
                "screen_name": author.get("username"),
                "description": author.get("description"),
                "verified": author.get("verified"),
                "favorite_count": post_metrics.get("like_count"),
                "retweet_count": post_metrics.get("retweet_count"),
                "reply_count": post_metrics.get("reply_count"),
                "followers_count": user_metrics.get("followers_count"),
                "friends_count": user_metrics.get("following_count"),
                "statuses_count": user_metrics.get("tweet_count"),
                "favourites_count": user_metrics.get("like_count"),
            }
        )
    errors = []
    for error in payload.get("errors", []):
        errors.append(
            {
                "tweet_id": str(error.get("value", "")),
                "title": error.get("title"),
                "detail": error.get("detail"),
                "type": error.get("type"),
            }
        )
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "dataset_ids.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "generated" / "hydrated_posts.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--pause", type=float, default=0.0)
    args = parser.parse_args()

    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise SystemExit("--batch-size must be between 1 and 100.")
    bearer_token = os.environ.get("X_BEARER_TOKEN")
    if not bearer_token:
        raise SystemExit("Set X_BEARER_TOKEN in the environment; never commit it.")

    identifiers = pd.read_csv(args.input, dtype={"tweet_id": "string"})
    post_ids = identifiers["tweet_id"].drop_duplicates().astype(str).tolist()
    hydrated_rows, error_rows = [], []
    session = requests.Session()
    batches = list(batched(post_ids, args.batch_size))
    for index, batch in enumerate(batches, start=1):
        payload = request_batch(session, batch, bearer_token, args.max_retries)
        rows, errors = normalize_response(payload)
        hydrated_rows.extend(rows)
        error_rows.extend(errors)
        print(f"Batch {index}/{len(batches)}: {len(rows)} available, {len(errors)} errors")
        if args.pause and index < len(batches):
            time.sleep(args.pause)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hydrated = pd.DataFrame(hydrated_rows).drop_duplicates("tweet_id")
    hydrated.to_parquet(args.output, index=False)
    hydrated.to_csv(args.output.with_suffix(".csv"), index=False)
    pd.DataFrame(error_rows).to_csv(
        args.output.with_name("hydration_errors.csv"), index=False
    )
    metadata = {
        "hydrated_at": datetime.now(timezone.utc).isoformat(),
        "api_endpoint": API_URL,
        "requested_ids": len(post_ids),
        "available_posts": len(hydrated),
        "errors": len(error_rows),
        "warning": "Current API values are not the historical collection snapshot used in the paper.",
    }
    args.output.with_name("hydration_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Hydrated content written to {args.output}")


if __name__ == "__main__":
    main()
