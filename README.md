# FakeNews-X-PT-BR: ID-only dataset and reproducibility package

This repository accompanies the FakeNews-X-PT-BR JBCS article. The public
dataset is distributed in an **ID-only** form: it contains X post IDs and the
researchers' labels, domain assignments, structured LLM annotations, split
assignments, predictions, and aggregate results. It does not redistribute post
text, URLs, account/profile fields, engagement snapshots, row-level embeddings,
or free-text LLM rationales.

This separation preserves a useful audit trail while avoiding republication of
hydrated X content. Researchers who are authorized to use the X API can hydrate
currently available posts into a local, ignored directory.

## What is included

- The 660 paper records as IDs, labels, and thematic domains, with stable
  `record_id` identifiers.
- A 659-post deduplicated ID view for future analyses.
- The 659 structured LLM polarity and marker annotations used by the study.
- Explicit temporal and LODO split assignments, without post timestamps or
  account identifiers.
- Metrics, hyperparameters, post-hoc tables, and all 175 per-sample prediction
  files used by the 25 benchmark configurations.
- Code for optional X API hydration, feature generation, model reruns,
  post-hoc analyses, validation, and author-overlap sensitivity summaries.

No API call or model retraining is needed to verify the reported metrics and
statistical analyses from the saved predictions.

## Public dataset files

| File | Rows | Intended use |
|---|---:|---|
| `data/dataset_ids.parquet` | 660 | Paper record IDs, labels, and domains |
| `data/dataset_ids_unique.parquet` | 659 | Deduplicated ID view for new analyses |
| `data/llm_annotations.parquet` | 659 | Structured annotations, one per post ID |
| `data/splits/temporal_paper.csv` | 660 | Exact paper temporal memberships |
| `data/splits/lodo_paper.csv` | 3,960 | Exact six-fold LODO memberships |

The paper snapshot contains two records with the same `tweet_id`, assigned to
different domains. They remain separate because they were separate experimental
records. `data/integrity/duplicate_records.csv` discloses both assignments; the
deduplicated view retains audit fields for the conflict.

## Quick verification

Python 3.9--3.12 is supported. The package was validated with the versions in
`requirements-lock.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
make validate
```

The validator checks identifiers, allowed public columns, split sizes,
prediction-to-label alignment, saved hyperparameter counts, post-hoc artifacts,
and SHA-256 checksums. It also fails if excluded hydrated-content files appear
in the public data directory.

## Recalculate paper statistics without training

```bash
make posthoc
make author-overlap
```

The 175 prediction files retain stable record/post IDs, domain, labels,
probabilities, predictions, and precomputed author-overlap flags. They exclude
timestamps and account identifiers.

## Optional local hydration and model rerun

Hydration requires an X developer account and a Bearer token with access to the
post lookup endpoint. Keep the token in the environment:

```bash
export X_BEARER_TOKEN="..."
make hydrate
make prepare-hydrated
make features
```

These commands write only to `data/generated/`, which is ignored by Git. Do not
commit or redistribute generated hydrated content. The hydration script uses the
official X API v2 post-lookup endpoint in batches of at most 100 IDs:

- <https://docs.x.com/x-api/posts/lookup/quickstart>
- <https://docs.x.com/x-api/posts/lookup/introduction>

Hydration is best-effort. Deleted, protected, suspended, or otherwise
unavailable posts cannot be recovered. Returned profile and engagement values
reflect the hydration time, not the historical collection snapshot used in the
paper. Consequently, a model rerun from freshly hydrated data is a new
replication, not a byte-identical reconstruction of the original training data.

After generating local features, a smoke run is:

```bash
make benchmark-smoke
```

The paper protocol uses 200 Optuna trials and three seeds per trial:

```bash
make benchmark-full
```

New outputs are written under `results/rerun/`; the paper artifacts under
`results/paper/` are not overwritten.

## Provenance

BERTimbau revision `94d69c95f98f7d5b2a8700c420230ae10def0baa` was used as a
frozen encoder with the `[CLS]` representation, maximum length 128, and 768
output dimensions. Row-level embeddings are not redistributed.

The original Portuguese annotation prompt is included for provenance. The
paper annotations were requested through OpenRouter with
`anthropic/claude-haiku-4.5`, provider identifier
`anthropic/claude-4.5-haiku-20251001`, temperature 0.7, and 1,024 maximum
tokens. The public table contains only the resulting structured polarity and
fixed-vocabulary markers, not the source text or free-text rationales.

## Directory guide

```text
data/                 ID-only records, annotations, integrity data, and splits
metadata/             manifests and SHA-256 checksums
results/paper/        reported results and sanitized per-sample predictions
results/rerun/        generated only when experiments are rerun
scripts/              hydration, benchmark, validation, and audit entry points
src/utils/            feature and classifier implementation
```

See `data/README.md` for the public schema and `REPRODUCIBILITY.md` for the
boundary between exact result verification and best-effort replication.

## Citation and licenses

Citation metadata is provided in `CITATION.cff`. Source code and documentation
are licensed under Apache-2.0. Researcher-created data fields are licensed under
CC BY 4.0. X post IDs are identifiers and are not claimed as original authored
content or placed under CC BY. See `LICENSE`, `LICENSE.md`, `data/LICENSE.md`,
and `THIRD_PARTY_DATA_NOTICE.md` for the precise scope.
