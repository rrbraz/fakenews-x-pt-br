# Public data documentation

## Distribution boundary

This directory is an ID-only research release. It does not contain hydrated X
post text, URLs, names, usernames, profile descriptions, timestamps,
engagement/profile snapshots, row-level embeddings, or free-text LLM
rationales. Optional hydration writes to the ignored `data/generated/`
directory and is governed by the user's X access and applicable terms.

## Paper record table

`dataset_ids.parquet` and `dataset_ids.csv` contain the 660 experimental
records used in the article.

| Field | Description |
|---|---|
| `record_id` | Stable release identifier for the experimental row |
| `tweet_id` | X post identifier |
| `categoria` | Thematic domain assigned by the researchers |
| `label` | Research label: `fake` or `true` |

There are 660 `record_id` values and 659 unique `tweet_id` values.

## Deduplicated ID view

`dataset_ids_unique.parquet` and `.csv` contain one row per `tweet_id` and add:

- `source_record_count`;
- `all_assigned_categories`;
- `source_record_ids`;
- `category_conflict`.

The duplicated post has conflicting domain assignments. The view keeps the
first canonical row for a deterministic 659-post table and retains both source
assignments in the audit fields. It was not used for the paper results.

## Structured LLM annotations

`llm_annotations.parquet` and `.csv` contain one annotation per unique post ID:

- `record_id`: canonical source record used for the annotation;
- `tweet_id`: join key;
- `llm_polaridade`: `positivo`, `neutro`, or `negativo`;
- `llm_marcadores_linguisticos`: labels from the fixed 29-marker vocabulary.

The table excludes the input text, text hashes, and free-text rationale. The
structured fields are researcher-created annotations; see `data/LICENSE.md`.

## Exact split assignments

`splits/temporal_paper.csv` records the exact paper memberships: 476 training,
85 validation, and 99 test records. It intentionally omits the timestamps from
which the chronological assignment was originally derived.

`splits/lodo_paper.csv` records train, validation, and test membership for all
six held-out domains. It contains six assignments per experimental record.

## Integrity record

`integrity/duplicate_records.csv` is the machine-readable disclosure of the two
experimental rows representing the same post ID. It contains only release IDs,
labels, domains, and the integrity issue.

## Locally generated data

With authorized X API access, the following sequence creates private local
artifacts:

```bash
python scripts/hydrate_x_posts.py
python scripts/prepare_hydrated_dataset.py
python scripts/compute_features.py
```

All outputs are placed under `data/generated/` and excluded from version
control. Current hydration cannot reconstruct unavailable posts or guarantee
the historical profile and engagement snapshots used in the paper.
