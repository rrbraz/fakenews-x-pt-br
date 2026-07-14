.PHONY: validate checksums posthoc author-overlap hydrate prepare-hydrated features benchmark-smoke benchmark-full

validate:
	python scripts/validate_release.py

checksums:
	python scripts/update_checksums.py

posthoc:
	python scripts/posthoc.py --input-dir results/paper

author-overlap:
	python scripts/author_overlap.py

hydrate:
	python scripts/hydrate_x_posts.py

prepare-hydrated:
	python scripts/prepare_hydrated_dataset.py

features:
	python scripts/compute_features.py

benchmark-smoke:
	python scripts/run_benchmark.py bert_llm --n-trials 2 --seeds-per-trial 1

benchmark-full:
	python scripts/run_benchmark.py all --n-trials 200 --seeds-per-trial 3
	python scripts/posthoc.py --input-dir results/rerun
