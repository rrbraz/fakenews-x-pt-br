"""Recalculate author-overlap sensitivity from released prediction files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FLAG_COLUMNS = ("author_seen_in_fit_train", "author_seen_in_development")


def summarize(prediction_dir: Path) -> pd.DataFrame:
    grouped: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for path in sorted(prediction_dir.glob("*.csv")):
        protocol, model, _fold = path.name.split("__", maxsplit=2)
        grouped.setdefault((protocol, model), []).append(pd.read_csv(path))

    rows = []
    for (protocol, model), frames in grouped.items():
        predictions = pd.concat(frames, ignore_index=True)
        for flag in FLAG_COLUMNS:
            for seen, subset in predictions.groupby(flag, dropna=False):
                rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "overlap_definition": flag,
                        "author_seen": bool(seen),
                        "n": len(subset),
                        "accuracy": float((subset["y_true"] == subset["y_pred"]).mean()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "model", "overlap_definition", "author_seen"]
    )


def build_wide_summary(long_table: pd.DataFrame) -> pd.DataFrame:
    wide = long_table.pivot_table(
        index=["protocol", "model", "overlap_definition"],
        columns="author_seen",
        values=["n", "accuracy"],
        aggfunc="first",
    )
    wide.columns = [
        f"{metric}_{'seen' if seen else 'unseen'}" for metric, seen in wide.columns
    ]
    wide = wide.reset_index()
    wide["accuracy_gap_seen_minus_unseen"] = wide.get("accuracy_seen") - wide.get(
        "accuracy_unseen"
    )
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=ROOT / "results" / "paper" / "predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper" / "author_overlap",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_table = summarize(args.prediction_dir)
    wide_table = build_wide_summary(long_table)
    long_table.to_csv(args.output_dir / "sensitivity_pooled.csv", index=False)
    wide_table.to_csv(args.output_dir / "sensitivity_summary.csv", index=False)
    print(f"Wrote {len(long_table)} long rows and {len(wide_table)} summary rows.")


if __name__ == "__main__":
    main()
