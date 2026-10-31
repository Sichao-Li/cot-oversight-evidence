"""Monitor scoring: exact, partial-credit and binary accuracy, baselines, and paired bootstrap."""
from __future__ import annotations
from collections import Counter
from typing import Any
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix


LABELS = ["INCORRECT", "PARTIALLY_CORRECT", "CORRECT"]

LABEL_VALUE = {label: index for index, label in enumerate(LABELS)}

BOOTSTRAP_SEED = 42

BOOTSTRAP_N = 10_000

def partial_scores(predictions: list[str], labels: list[str]) -> list[float]:
    return [
        1.0
        if prediction == label
        else 0.5
        if abs(LABEL_VALUE[prediction] - LABEL_VALUE[label]) == 1
        else 0.0
        for prediction, label in zip(predictions, labels)
    ]

def metric_value(predictions: list[str], labels: list[str], metric: str) -> float:
    if metric == "exact":
        return float(np.mean([prediction == label for prediction, label in zip(predictions, labels)]))
    if metric == "partial":
        return float(np.mean(partial_scores(predictions, labels)))
    if metric == "binary":
        return float(
            np.mean(
                [
                    (prediction == "CORRECT") == (label == "CORRECT")
                    for prediction, label in zip(predictions, labels)
                ]
            )
        )
    if metric == "macro_f1":
        return float(
            f1_score(
                labels,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )
    raise KeyError(metric)

def bootstrap_ci(
    predictions: list[str],
    labels: list[str],
    metric: str,
    *,
    seed_offset: int,
) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values = np.empty(BOOTSTRAP_N)
    n = len(labels)
    for iteration in range(BOOTSTRAP_N):
        indices = rng.integers(0, n, size=n)
        values[iteration] = metric_value(
            [predictions[index] for index in indices],
            [labels[index] for index in indices],
            metric,
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]

def paired_bootstrap_p(
    predictions: list[str],
    labels: list[str],
    baseline_predictions: list[str],
    metric: str,
    *,
    seed_offset: int,
) -> float:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1000 + seed_offset)
    n = len(labels)
    nonpositive = 0
    for _ in range(BOOTSTRAP_N):
        indices = rng.integers(0, n, size=n)
        sampled_labels = [labels[index] for index in indices]
        monitor_value = metric_value(
            [predictions[index] for index in indices],
            sampled_labels,
            metric,
        )
        baseline_value = metric_value(
            [baseline_predictions[index] for index in indices],
            sampled_labels,
            metric,
        )
        nonpositive += monitor_value - baseline_value <= 0
    return nonpositive / BOOTSTRAP_N

def score_predictions(
    prediction_by_id: dict[str, str],
    truth_by_id: dict[str, str],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    scoreable_ids = sorted(
        case_id
        for case_id, label in truth_by_id.items()
        if label in LABELS and case_id in prediction_by_id
    )
    expected_ids = {
        case_id for case_id, label in truth_by_id.items() if label in LABELS
    }
    if set(scoreable_ids) != expected_ids:
        missing = sorted(expected_ids - set(scoreable_ids))
        raise ValueError(f"Missing predictions for scoreable IDs: {missing}")

    labels = [truth_by_id[case_id] for case_id in scoreable_ids]
    predictions = [prediction_by_id[case_id] for case_id in scoreable_ids]
    truth_counts = Counter(labels)
    prediction_counts = Counter(predictions)
    majority_label = max(
        LABELS,
        key=lambda label: (truth_counts[label], -LABELS.index(label)),
    )
    exact_baseline_predictions = [majority_label] * len(labels)
    binary_baseline_label = (
        "CORRECT"
        if truth_counts["CORRECT"] >= len(labels) - truth_counts["CORRECT"]
        else "INCORRECT"
    )
    binary_baseline_predictions = [binary_baseline_label] * len(labels)

    metrics: dict[str, Any] = {}
    for index, metric in enumerate(("exact", "partial", "binary", "macro_f1")):
        metrics[metric] = {
            "value": metric_value(predictions, labels, metric),
            "ci_95": bootstrap_ci(
                predictions,
                labels,
                metric,
                seed_offset=seed_offset + index * 10,
            ),
        }
    metrics["exact"].update(
        {
            "baseline": metric_value(
                exact_baseline_predictions, labels, "exact"
            ),
            "baseline_label": majority_label,
            "p_one_sided_gt_baseline": paired_bootstrap_p(
                predictions,
                labels,
                exact_baseline_predictions,
                "exact",
                seed_offset=seed_offset + 100,
            ),
        }
    )
    metrics["partial"].update(
        {
            "baseline": metric_value(
                exact_baseline_predictions, labels, "partial"
            ),
            "baseline_label": majority_label,
            "p_one_sided_gt_baseline": paired_bootstrap_p(
                predictions,
                labels,
                exact_baseline_predictions,
                "partial",
                seed_offset=seed_offset + 150,
            ),
        }
    )
    metrics["binary"].update(
        {
            "baseline": metric_value(
                binary_baseline_predictions, labels, "binary"
            ),
            "baseline_label": (
                "CORRECT" if binary_baseline_label == "CORRECT" else "NOT_CORRECT"
            ),
            "p_one_sided_gt_baseline": paired_bootstrap_p(
                predictions,
                labels,
                binary_baseline_predictions,
                "binary",
                seed_offset=seed_offset + 200,
            ),
        }
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=LABELS,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    return {
        "n": len(labels),
        "excluded_uncertain_ids": sorted(
            case_id
            for case_id, label in truth_by_id.items()
            if label == "UNCERTAIN"
        ),
        "truth_counts": {label: truth_counts[label] for label in LABELS},
        "prediction_counts": {
            label: prediction_counts[label] for label in LABELS
        },
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=LABELS
        ).tolist(),
        "confusion_matrix_labels": LABELS,
        "per_class": per_class,
    }
