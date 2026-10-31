"""Validated scoring; unresolved labels and incomplete predictions remain explicit."""
from collections import Counter
from . import scoring


def evaluate(predictions, labels, *, bootstrap=False, seed_offset=0):
    invalid = [key for key, value in labels.items() if value not in scoring.LABELS + ["UNCERTAIN"]]
    if invalid:
        raise ValueError(f"Unknown annotation labels: {invalid}")
    ids = sorted(key for key, value in labels.items() if value in scoring.LABELS)
    if not ids:
        raise ValueError("No scoreable annotations.")
    missing = [key for key in ids if predictions.get(key) not in scoring.LABELS]
    if missing:
        raise ValueError(f"Missing or invalid predictions for scoreable cases: {missing}")
    if bootstrap:
        return scoring.score_predictions(predictions, labels, seed_offset=seed_offset)
    truth, pred = [labels[key] for key in ids], [predictions[key] for key in ids]
    counts = Counter(truth)
    majority = max(scoring.LABELS, key=lambda label: (counts[label], -scoring.LABELS.index(label)))
    binary = "CORRECT" if counts["CORRECT"] >= len(ids) - counts["CORRECT"] else "INCORRECT"
    metrics = {name: {"value": scoring.metric_value(pred, truth, name)} for name in ("exact", "partial", "binary", "macro_f1")}
    for name in ("exact", "partial", "binary"):
        constant = binary if name == "binary" else majority
        metrics[name].update(baseline=scoring.metric_value([constant] * len(ids), truth, name), baseline_label=constant)
    return {"n": len(ids), "excluded_uncertain_ids": sorted(k for k, v in labels.items() if v == "UNCERTAIN"),
            "metrics": metrics, "label_order": scoring.LABELS}
