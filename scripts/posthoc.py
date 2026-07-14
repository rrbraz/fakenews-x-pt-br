"""Recalculate statistical analyses from released per-sample predictions."""
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_NAME = re.compile(
    r"^(?P<protocol>lodo|temporal)__(?P<model>.+)__(?P<fold>.+)\.csv$"
)


def load_predictions(prediction_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(prediction_dir.glob("*.csv")):
        match = PREDICTION_NAME.match(path.name)
        if not match:
            continue
        frame = pd.read_csv(path)
        required = {"record_id", "y_true", "y_pred", "y_proba"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing {sorted(missing)}")
        rows.append(
            {
                "protocol": match.group("protocol"),
                "model": match.group("model"),
                "fold": match.group("fold"),
                "frame": frame[["record_id", "y_true", "y_pred", "y_proba"]].copy(),
            }
        )
    return rows


def accuracy_with_ci(predictions: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in predictions:
        frame = item["frame"]
        correct = frame["y_true"].eq(frame["y_pred"])
        low, high = proportion_confint(
            int(correct.sum()), len(frame), alpha=0.05, method="wilson"
        )
        rows.append(
            {
                "split": item["protocol"],
                "model": item["model"],
                "fold": item["fold"],
                "n": len(frame),
                "n_correct": int(correct.sum()),
                "accuracy": float(correct.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def _paired_mcnemar(model_a: str, frame_a: pd.DataFrame, model_b: str,
                    frame_b: pd.DataFrame) -> dict[str, object]:
    left = frame_a.rename(
        columns={"y_true": "y_true_a", "y_pred": "y_pred_a", "y_proba": "y_proba_a"}
    )
    right = frame_b.rename(
        columns={"y_true": "y_true_b", "y_pred": "y_pred_b", "y_proba": "y_proba_b"}
    )
    paired = left.merge(right, on="record_id", how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(f"Sample alignment differs between {model_a} and {model_b}")
    if not paired["y_true_a"].equals(paired["y_true_b"]):
        raise ValueError(f"Label alignment differs between {model_a} and {model_b}")
    correct_a = paired["y_pred_a"].eq(paired["y_true_a"])
    correct_b = paired["y_pred_b"].eq(paired["y_true_b"])
    n11 = int((correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    n00 = int((~correct_a & ~correct_b).sum())
    result = mcnemar([[n11, n10], [n01, n00]], exact=True)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "a_acc": float(correct_a.mean()),
        "b_acc": float(correct_b.mean()),
        "delta_acc": float(correct_a.mean() - correct_b.mean()),
        "n": len(paired),
        "n10": n10,
        "n01": n01,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def pairwise_temporal(predictions: list[dict[str, object]]) -> pd.DataFrame:
    temporal = [item for item in predictions if item["protocol"] == "temporal"]
    return pd.DataFrame(
        _paired_mcnemar(a["model"], a["frame"], b["model"], b["frame"])
        for a, b in combinations(temporal, 2)
    )


def pairwise_lodo(predictions: list[dict[str, object]]) -> pd.DataFrame:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for item in predictions:
        if item["protocol"] == "lodo":
            grouped.setdefault(item["model"], []).append(item["frame"])
    pooled = {
        model: pd.concat(frames, ignore_index=True).sort_values("record_id")
        for model, frames in grouped.items()
    }
    return pd.DataFrame(
        _paired_mcnemar(model_a, pooled[model_a], model_b, pooled[model_b])
        for model_a, model_b in combinations(sorted(pooled), 2)
    )


def bootstrap_temporal(
    predictions: list[dict[str, object]], n_bootstrap: int = 1000, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item in predictions:
        if item["protocol"] != "temporal":
            continue
        frame = item["frame"]
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_pred = frame["y_pred"].to_numpy(dtype=int)
        y_proba = frame["y_proba"].to_numpy(dtype=float)
        accuracy, f1_values, auc = [], [], []
        for _ in range(n_bootstrap):
            index = rng.integers(0, len(frame), len(frame))
            truth, prediction, probability = y_true[index], y_pred[index], y_proba[index]
            accuracy.append(accuracy_score(truth, prediction))
            f1_values.append(f1_score(truth, prediction, average="weighted", zero_division=0))
            auc.append(roc_auc_score(truth, probability) if len(np.unique(truth)) > 1 else np.nan)
        rows.append(
            {
                "model": item["model"],
                "n": len(frame),
                "acc_mean": float(np.mean(accuracy)),
                "acc_ci_low": float(np.percentile(accuracy, 2.5)),
                "acc_ci_high": float(np.percentile(accuracy, 97.5)),
                "f1_mean": float(np.mean(f1_values)),
                "f1_ci_low": float(np.percentile(f1_values, 2.5)),
                "f1_ci_high": float(np.percentile(f1_values, 97.5)),
                "auc_mean": float(np.nanmean(auc)),
                "auc_ci_low": float(np.nanpercentile(auc, 2.5)),
                "auc_ci_high": float(np.nanpercentile(auc, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def protocol_correlation(predictions: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lodo: dict[str, list[float]] = {}
    temporal = {}
    for item in predictions:
        accuracy = float(item["frame"]["y_true"].eq(item["frame"]["y_pred"]).mean())
        if item["protocol"] == "lodo":
            lodo.setdefault(item["model"], []).append(accuracy)
        else:
            temporal[item["model"]] = accuracy
    lodo_macro = {model: float(np.mean(values)) for model, values in lodo.items()}
    common = sorted(set(lodo_macro) & set(temporal))
    paired = pd.DataFrame(
        {
            "model": common,
            "lodo_macro_acc": [lodo_macro[model] for model in common],
            "temporal_acc": [temporal[model] for model in common],
        }
    )
    methods = [
        ("Pearson", stats.pearsonr),
        ("Spearman", stats.spearmanr),
        ("Kendall", stats.kendalltau),
    ]
    rows = []
    for name, function in methods:
        coefficient, p_value = function(paired["lodo_macro_acc"], paired["temporal_acc"])
        rows.append(
            {"method": name, "r": coefficient, "p_value": p_value, "n_models": len(paired)}
        )
    return pd.DataFrame(rows), paired


def factorial_holm(temporal: pd.DataFrame, lodo: pd.DataFrame) -> pd.DataFrame:
    classifiers = ["LogisticRegression", "SVC", "RandomForest"]
    representation = {"text": "bert", "numerical": "bert_num", "llm": "bert_llm", "all": "bert_meta"}
    comparisons = [
        ("numerical_vs_text", "numerical", "text"),
        ("llm_vs_text", "llm", "text"),
        ("all_vs_text", "all", "text"),
        ("all_vs_numerical", "all", "numerical"),
        ("all_vs_llm", "all", "llm"),
    ]
    rows = []
    for protocol, source in (("lodo", lodo), ("temporal", temporal)):
        for classifier in classifiers:
            for contrast, left, right in comparisons:
                model_a = f"{representation[left]}-{classifier}"
                model_b = f"{representation[right]}-{classifier}"
                match = source[
                    ((source["model_a"] == model_a) & (source["model_b"] == model_b))
                    | ((source["model_a"] == model_b) & (source["model_b"] == model_a))
                ]
                if len(match) != 1:
                    raise ValueError(f"Expected one contrast for {protocol}: {model_a}, {model_b}")
                result = match.iloc[0]
                if result["model_a"] == model_a:
                    accuracy_a, accuracy_b = result["a_acc"], result["b_acc"]
                else:
                    accuracy_a, accuracy_b = result["b_acc"], result["a_acc"]
                rows.append(
                    {
                        "protocol": protocol,
                        "classifier": classifier,
                        "contrast": contrast,
                        "model_a": model_a,
                        "model_b": model_b,
                        "accuracy_a": accuracy_a,
                        "accuracy_b": accuracy_b,
                        "delta_accuracy": accuracy_a - accuracy_b,
                        "p_raw": result["p_value"],
                    }
                )
    output = pd.DataFrame(rows)
    output["p_holm"] = np.nan
    output["reject_at_0.05"] = False
    for _protocol, indices in output.groupby("protocol").groups.items():
        index = np.asarray(list(indices), dtype=int)
        reject, adjusted, _, _ = multipletests(
            output.loc[index, "p_raw"].to_numpy(dtype=float), method="holm"
        )
        output.loc[index, "p_holm"] = adjusted
        output.loc[index, "reject_at_0.05"] = reject
    return output


def aggregated_summary(
    predictions: list[dict[str, object]], temporal_bootstrap: pd.DataFrame
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    grouped: dict[str, list[float]] = {}
    for item in predictions:
        if item["protocol"] == "lodo":
            grouped.setdefault(item["model"], []).append(
                float(item["frame"]["y_true"].eq(item["frame"]["y_pred"]).mean())
            )
    rows = []
    for model, values in grouped.items():
        accuracies = np.asarray(values)
        bootstrap = np.asarray(
            [np.mean(rng.choice(accuracies, size=len(accuracies), replace=True)) for _ in range(2000)]
        )
        rows.append(
            {
                "model": model,
                "split": "lodo",
                "macro_acc": float(accuracies.mean()),
                "ci_low": float(np.percentile(bootstrap, 2.5)),
                "ci_high": float(np.percentile(bootstrap, 97.5)),
                "n_folds": len(accuracies),
            }
        )
    for row in temporal_bootstrap.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "split": "temporal",
                "macro_acc": row.acc_mean,
                "ci_low": row.acc_ci_low,
                "ci_high": row.acc_ci_high,
                "n_folds": 1,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results" / "paper")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    prediction_dir = args.input_dir / "predictions"
    output_dir = args.input_dir / "posthoc"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(prediction_dir)
    if not predictions:
        raise SystemExit(f"No predictions found in {prediction_dir}")
    accuracy = accuracy_with_ci(predictions)
    temporal = pairwise_temporal(predictions)
    lodo = pairwise_lodo(predictions)
    bootstrap = bootstrap_temporal(predictions, args.bootstrap_samples)
    correlation, correlation_data = protocol_correlation(predictions)
    holm = factorial_holm(temporal, lodo)
    summary = aggregated_summary(predictions, bootstrap)

    accuracy.to_csv(output_dir / "accuracy_with_ci.csv", index=False)
    temporal.to_csv(output_dir / "temporal_mcnemar.csv", index=False)
    lodo.to_csv(output_dir / "lodo_mcnemar.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_ci_temporal.csv", index=False)
    correlation.to_csv(output_dir / "correlation_lodo_temporal.csv", index=False)
    correlation_data.to_csv(output_dir / "correlation_data.csv", index=False)
    holm.to_csv(output_dir / "holm_corrections.csv", index=False)
    summary.to_csv(output_dir / "summary_aggregated.csv", index=False)

    print(f"Loaded {len(predictions)} prediction sets.")
    print(f"Pairwise tests: {len(lodo)} LODO and {len(temporal)} temporal.")
    print(
        f"Factorial Holm family: {len(holm)} contrasts; "
        f"{int(holm['reject_at_0.05'].sum())} significant."
    )
    print(correlation.to_string(index=False))


if __name__ == "__main__":
    main()
