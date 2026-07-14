# Reproducibility notes

## What can be reproduced exactly

The public package supports exact verification of the article's downstream
results from the supplied artifacts:

1. recompute metrics from all 175 per-sample prediction files;
2. recompute confidence intervals, paired tests, Holm corrections, and
   cross-protocol correlations;
3. verify every test record's label and split membership;
4. inspect the saved best hyperparameters and trial counts.

No X API access, model download, or retraining is needed for these checks. This
is the recommended route for auditing the paper's numerical claims.

## What cannot be reconstructed exactly from the public package

The release intentionally excludes hydrated post/profile content, historical
engagement and profile snapshots, and row-level BERTimbau embeddings. Therefore
it does not support byte-identical reconstruction of the original upstream
feature matrix or training run from public files alone.

Authorized users may hydrate currently available post IDs and run the included
feature/training pipeline. That exercise is a replication: unavailable posts,
changed profile/engagement values, model-service changes, and stochastic
estimators can make its outputs differ from the reported experiment.

## Experimental matrix

The benchmark contains 25 configurations:

- nine classifiers with TF-IDF;
- seven classifiers with frozen BERTimbau embeddings;
- logistic regression, SVM, and random forest under a controlled 2x2 feature
  comparison: BERT only, BERT plus numerical metadata, BERT plus LLM features,
  and BERT plus both blocks.

In result filenames, `bert_meta` means BERT plus both numerical and LLM blocks
(`BERT + all` in the article).

Each model/fold used 200 Optuna trials. The paper runner used seeds 42, 43, and
44 per trial, optimized validation F1 averaged over those seeds, averaged final
test probabilities, and applied a 0.5 decision threshold.

## Protocols

- **LODO:** the held-out domain is absent from training and validation. The
  remaining records use a 15% stratified validation share with seed 42.
- **Temporal:** the original private collection snapshot was ordered by post
  time; the newest 15% formed the test set and the newest 15% of the remainder
  formed validation.

The public split CSVs are the authoritative memberships. Timestamps are not
redistributed.

## Statistical artifacts

`results/paper/predictions/` contains the sample-level basis for the reported
metrics. `results/paper/posthoc/` includes Wilson and bootstrap intervals,
pooled paired McNemar tests, planned multiple-comparison corrections, and
protocol correlations.

LODO macro accuracy gives equal weight to domains; pooled McNemar tests give
equal weight to records. Both are retained because they answer different
questions.

## Author-overlap sensitivity

Prediction files retain two Boolean flags computed on the private source
snapshot:

- `author_seen_in_fit_train`;
- `author_seen_in_development`.

They do not expose account identifiers. `results/paper/author_overlap/`
contains counts and accuracy breakdowns using these flags. This is a
sensitivity analysis of saved predictions, not author-grouped retraining.

## Provenance and variability

The original frozen text encoder was BERTimbau revision
`94d69c95f98f7d5b2a8700c420230ae10def0baa`. Regenerated embeddings may vary
with model revision and numerical backend. New LLM annotations may vary because
the original annotation temperature was 0.7 and provider behavior is not fully
deterministic. The structured paper annotations are supplied so their role can
be audited without redistributing source text or rationales.
