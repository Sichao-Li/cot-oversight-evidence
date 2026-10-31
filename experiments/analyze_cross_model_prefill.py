"""Compare activation-minus-text probe advantage on the model's own traces versus another model's traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _path in (EXPERIMENTS_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cot_oversight_case_studies.io import cache_path, load_jsonl, write_jsonl  # noqa: E402
from cot_oversight_case_studies.reasoning_probe import (  # noqa: E402
    REASONING_STEPS,
    normalize_reasoning_text,
    parse_layer_list,
    step_presence_labels,
)


FIXED_LAYERS = {
    "identifies_vulnerability": 28,
    "traces_data_flow": 8,
    "assesses_exploitability": 28,
}


def security_rows(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("status") == "ok"
        and row.get("domain") == "domain_b"
        and row.get("variant") == "standard"
    ]
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {len(rows)}.")
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate record IDs.")
    return rows


def load_activation_matrices(
    activation_dir: Path,
    record_ids: list[str],
    *,
    expected_layers: list[int],
) -> dict[int, np.ndarray]:
    by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in expected_layers}
    for record_id in record_ids:
        path = cache_path(activation_dir / "activation_cache", record_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing activation cache for {record_id}: {path}")
        with np.load(path, allow_pickle=False) as payload:
            cached_id = str(payload["record_id"].item())
            layers = [int(item) for item in payload["layers"].tolist()]
            matrix = np.asarray(payload["activations"], dtype=np.float32)
        if cached_id != record_id:
            raise ValueError(f"Cache record mismatch in {path}: {cached_id} != {record_id}.")
        if not set(expected_layers).issubset(layers):
            raise ValueError(f"Cache {path} does not contain all expected layers.")
        layer_to_offset = {layer: index for index, layer in enumerate(layers)}
        for layer in expected_layers:
            by_layer[layer].append(matrix[layer_to_offset[layer]])
    return {layer: np.stack(values) for layer, values in by_layer.items()}


def stratified_splits(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return list(splitter.split(np.zeros(len(y)), y))


def activation_oof(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    probabilities = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    fold_aps: list[float] = []
    for train_indices, test_indices in splits:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(x[train_indices], y[train_indices])
        fold_probabilities = classifier.predict_proba(x[test_indices])[:, 1]
        probabilities[test_indices] = fold_probabilities
        fold_aucs.append(float(roc_auc_score(y[test_indices], fold_probabilities)))
        fold_aps.append(
            float(average_precision_score(y[test_indices], fold_probabilities))
        )
    return {
        "probabilities": probabilities,
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "fold_ap_mean": float(np.mean(fold_aps)),
        "fold_ap_std": float(np.std(fold_aps)),
        "oof_auc": float(roc_auc_score(y, probabilities)),
        "oof_ap": float(average_precision_score(y, probabilities)),
    }


def text_oof(
    texts: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline

    probabilities = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    fold_aps: list[float] = []
    for train_indices, test_indices in splits:
        pipeline = Pipeline(
            [
                ("text", CountVectorizer(ngram_range=(1, 1), binary=True)),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        )
        train_texts = [texts[index] for index in train_indices]
        test_texts = [texts[index] for index in test_indices]
        pipeline.fit(train_texts, y[train_indices])
        fold_probabilities = pipeline.predict_proba(test_texts)[:, 1]
        probabilities[test_indices] = fold_probabilities
        fold_aucs.append(float(roc_auc_score(y[test_indices], fold_probabilities)))
        fold_aps.append(
            float(average_precision_score(y[test_indices], fold_probabilities))
        )
    return {
        "probabilities": probabilities,
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "fold_ap_mean": float(np.mean(fold_aps)),
        "fold_ap_std": float(np.std(fold_aps)),
        "oof_auc": float(roc_auc_score(y, probabilities)),
        "oof_ap": float(average_precision_score(y, probabilities)),
    }


def fixed_probe_transfer(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    probabilities = np.zeros(len(target_y), dtype=float)
    valid_fold_aucs: list[float] = []
    for train_indices, test_indices in splits:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(source_x[train_indices], source_y[train_indices])
        fold_probabilities = classifier.predict_proba(target_x[test_indices])[:, 1]
        probabilities[test_indices] = fold_probabilities
        if len(np.unique(target_y[test_indices])) == 2:
            valid_fold_aucs.append(
                float(roc_auc_score(target_y[test_indices], fold_probabilities))
            )
    return {
        "probabilities": probabilities,
        "oof_auc": float(roc_auc_score(target_y, probabilities)),
        "valid_fold_auc_mean": (
            float(np.mean(valid_fold_aucs)) if valid_fold_aucs else None
        ),
        "valid_fold_count": len(valid_fold_aucs),
    }


def first_trigger_token(
    tokenizer: Any,
    token_ids: list[int],
    *,
    step_name: str,
    max_tokens: int,
) -> int | None:
    upper = min(len(token_ids), max_tokens)
    for token_count in range(1, upper + 1):
        prefix = tokenizer.decode(
            token_ids[:token_count],
            skip_special_tokens=False,
        )
        if step_presence_labels(prefix)[step_name] == 1:
            return token_count
    return None


def auc_or_none(y: np.ndarray, probabilities: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probabilities))


def paired_bootstrap(
    statistic: Callable[[np.ndarray], float | None],
    *,
    n_rows: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, n_rows, size=n_rows)
        value = statistic(indices)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        raise ValueError("No valid paired-bootstrap draws.")
    values_array = np.asarray(values)
    return {
        "draws_requested": draws,
        "draws_valid": len(values),
        "ci95": [
            float(np.percentile(values_array, 2.5)),
            float(np.percentile(values_array, 97.5)),
        ],
        "bootstrap_mean": float(np.mean(values_array)),
        "fraction_le_zero": float(np.mean(values_array <= 0)),
        "fraction_ge_zero": float(np.mean(values_array >= 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare self-trace and foreign-trace prefill with matched text "
            "baselines and paired task bootstrap."
        )
    )
    parser.add_argument("--self-inference", type=Path, required=True)
    parser.add_argument("--foreign-inference", type=Path, required=True)
    parser.add_argument("--self-activation-dir", type=Path, required=True)
    parser.add_argument("--foreign-activation-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="8,12,16,20,24,28")
    parser.add_argument("--max-reasoning-tokens", type=int, default=256)
    parser.add_argument("--expected-rows", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from tokenizers import Tokenizer

    layers = parse_layer_list(args.layers)
    self_rows = security_rows(args.self_inference, expected_rows=args.expected_rows)
    foreign_rows = security_rows(
        args.foreign_inference,
        expected_rows=args.expected_rows,
    )
    self_ids = [str(row["record_id"]) for row in self_rows]
    foreign_ids = [str(row["record_id"]) for row in foreign_rows]
    if self_ids != foreign_ids:
        raise ValueError("Condition rows are not aligned by record ID.")
    if any(a.get("prompt") != b.get("prompt") for a, b in zip(self_rows, foreign_rows)):
        raise ValueError("Condition prompts differ.")

    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    conditions = {"self": self_rows, "foreign": foreign_rows}
    texts: dict[str, list[str]] = {}
    labels: dict[str, dict[str, np.ndarray]] = {}
    matched_texts: dict[str, list[str]] = {}
    trigger_summaries: list[dict[str, Any]] = []
    for condition, rows in conditions.items():
        condition_texts = [normalize_reasoning_text(row) for row in rows]
        condition_token_ids = [
            [
                int(token_id)
                for token_id in tokenizer.encode(
                    text,
                    add_special_tokens=False,
                ).ids
            ]
            for text in condition_texts
        ]
        texts[condition] = condition_texts
        matched_texts[condition] = [
            tokenizer.decode(
                ids[: args.max_reasoning_tokens],
                skip_special_tokens=False,
            )
            for ids in condition_token_ids
        ]
        labels[condition] = {
            step: np.asarray(
                [step_presence_labels(text)[step] for text in condition_texts],
                dtype=int,
            )
            for step in REASONING_STEPS
        }
        for step in REASONING_STEPS:
            y = labels[condition][step]
            positive_indices = np.flatnonzero(y == 1).tolist()
            triggers = [
                first_trigger_token(
                    tokenizer,
                    condition_token_ids[index],
                    step_name=step,
                    max_tokens=args.max_reasoning_tokens,
                )
                for index in positive_indices
            ]
            within = sum(trigger is not None for trigger in triggers)
            trigger_summaries.append(
                {
                    "condition": condition,
                    "step_name": step,
                    "positive_count": len(positive_indices),
                    "first_trigger_within_span_count": within,
                    "first_trigger_within_span_fraction": (
                        float(within / len(positive_indices))
                        if positive_indices
                        else None
                    ),
                    "max_reasoning_tokens": args.max_reasoning_tokens,
                }
            )

    activations = {
        "self": load_activation_matrices(
            args.self_activation_dir,
            self_ids,
            expected_layers=layers,
        ),
        "foreign": load_activation_matrices(
            args.foreign_activation_dir,
            self_ids,
            expected_layers=layers,
        ),
    }

    condition_results: dict[str, dict[str, dict[str, Any]]] = {
        condition: {} for condition in conditions
    }
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for step in REASONING_STEPS:
            y = labels[condition][step]
            splits = stratified_splits(y)
            bow = text_oof(matched_texts[condition], y, splits)
            per_layer = {
                layer: activation_oof(activations[condition][layer], y, splits)
                for layer in layers
            }
            fixed_layer = FIXED_LAYERS[step]
            if fixed_layer not in per_layer:
                raise ValueError(f"Fixed layer {fixed_layer} was not captured.")
            best_layer = max(
                layers,
                key=lambda layer: per_layer[layer]["fold_auc_mean"],
            )
            fixed = per_layer[fixed_layer]
            best = per_layer[best_layer]
            condition_results[condition][step] = {
                "labels": y,
                "splits": splits,
                "bow": bow,
                "fixed": fixed,
                "best": best,
                "fixed_layer": fixed_layer,
                "best_layer": best_layer,
            }
            metric_rows.append(
                {
                    "record_type": "condition_refit",
                    "condition": condition,
                    "step_name": step,
                    "positive_count": int(y.sum()),
                    "negative_count": int(len(y) - y.sum()),
                    "fixed_layer": fixed_layer,
                    "fixed_activation_fold_auc_mean": fixed["fold_auc_mean"],
                    "fixed_activation_oof_auc": fixed["oof_auc"],
                    "bow_fold_auc_mean": bow["fold_auc_mean"],
                    "bow_oof_auc": bow["oof_auc"],
                    "fixed_activation_minus_bow_oof_auc": (
                        fixed["oof_auc"] - bow["oof_auc"]
                    ),
                    "best_layer": best_layer,
                    "best_activation_fold_auc_mean": best["fold_auc_mean"],
                    "best_activation_oof_auc": best["oof_auc"],
                }
            )
            for index, record_id in enumerate(self_ids):
                prediction_rows.append(
                    {
                        "record_id": record_id,
                        "condition": condition,
                        "step_name": step,
                        "label": int(y[index]),
                        "fixed_layer": fixed_layer,
                        "fixed_activation_probability": float(
                            fixed["probabilities"][index]
                        ),
                        "bow_probability": float(bow["probabilities"][index]),
                        "best_layer": best_layer,
                        "best_activation_probability": float(
                            best["probabilities"][index]
                        ),
                    }
                )

    transfer_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for step_index, step in enumerate(REASONING_STEPS):
        fixed_layer = FIXED_LAYERS[step]
        self_result = condition_results["self"][step]
        foreign_result = condition_results["foreign"][step]
        self_to_foreign = fixed_probe_transfer(
            activations["self"][fixed_layer],
            self_result["labels"],
            activations["foreign"][fixed_layer],
            foreign_result["labels"],
            self_result["splits"],
        )
        foreign_to_self = fixed_probe_transfer(
            activations["foreign"][fixed_layer],
            foreign_result["labels"],
            activations["self"][fixed_layer],
            self_result["labels"],
            foreign_result["splits"],
        )
        transfer_rows.extend(
            [
                {
                    "step_name": step,
                    "layer": fixed_layer,
                    "train_condition": "self",
                    "test_condition": "foreign",
                    **{
                        key: value
                        for key, value in self_to_foreign.items()
                        if key != "probabilities"
                    },
                },
                {
                    "step_name": step,
                    "layer": fixed_layer,
                    "train_condition": "foreign",
                    "test_condition": "self",
                    **{
                        key: value
                        for key, value in foreign_to_self.items()
                        if key != "probabilities"
                    },
                },
            ]
        )

        self_y = self_result["labels"]
        foreign_y = foreign_result["labels"]
        self_activation = self_result["fixed"]["probabilities"]
        self_bow = self_result["bow"]["probabilities"]
        foreign_activation = foreign_result["fixed"]["probabilities"]
        foreign_bow = foreign_result["bow"]["probabilities"]

        def activation_minus_bow_self(indices: np.ndarray) -> float | None:
            activation_auc = auc_or_none(self_y[indices], self_activation[indices])
            bow_auc = auc_or_none(self_y[indices], self_bow[indices])
            if activation_auc is None or bow_auc is None:
                return None
            return activation_auc - bow_auc

        def activation_minus_bow_foreign(indices: np.ndarray) -> float | None:
            activation_auc = auc_or_none(
                foreign_y[indices],
                foreign_activation[indices],
            )
            bow_auc = auc_or_none(foreign_y[indices], foreign_bow[indices])
            if activation_auc is None or bow_auc is None:
                return None
            return activation_auc - bow_auc

        def difference_in_differences(indices: np.ndarray) -> float | None:
            self_gap = activation_minus_bow_self(indices)
            foreign_gap = activation_minus_bow_foreign(indices)
            if self_gap is None or foreign_gap is None:
                return None
            return self_gap - foreign_gap

        statistics = {
            "self_activation_minus_bow": activation_minus_bow_self,
            "foreign_activation_minus_bow": activation_minus_bow_foreign,
            "difference_in_differences": difference_in_differences,
        }
        point_estimates = {
            name: statistic(np.arange(len(self_ids)))
            for name, statistic in statistics.items()
        }
        for statistic_offset, (name, statistic) in enumerate(statistics.items()):
            bootstrap_rows.append(
                {
                    "step_name": step,
                    "layer": fixed_layer,
                    "statistic": name,
                    "point_estimate": point_estimates[name],
                    **paired_bootstrap(
                        statistic,
                        n_rows=len(self_ids),
                        draws=args.bootstrap_draws,
                        seed=(
                            args.bootstrap_seed
                            + 100 * step_index
                            + statistic_offset
                        ),
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "condition_metrics.jsonl", metric_rows)
    write_jsonl(args.output_dir / "transfer_metrics.jsonl", transfer_rows)
    write_jsonl(args.output_dir / "paired_bootstrap.jsonl", bootstrap_rows)
    write_jsonl(args.output_dir / "trigger_coverage.jsonl", trigger_summaries)
    write_jsonl(args.output_dir / "oof_predictions.jsonl", prediction_rows)
    protocol = {
        "self_inference": str(args.self_inference.resolve()),
        "foreign_inference": str(args.foreign_inference.resolve()),
        "self_activation_dir": str(args.self_activation_dir.resolve()),
        "foreign_activation_dir": str(args.foreign_activation_dir.resolve()),
        "tokenizer_json": str(args.tokenizer_json.resolve()),
        "layers": layers,
        "fixed_layers_from_original_200_row_cv": FIXED_LAYERS,
        "fixed_layer_selection_disclosure": (
            "The fixed layers were originally selected using all 200 rows."
        ),
        "labels": "frozen_full_trace_deterministic_step_presence_rules",
        "matched_text_baseline": "binary_unigram_logistic_regression",
        "cv": "five_fold_stratified_shuffle_random_state_0",
        "primary_metric": (
            "(activation_oof_auc - bow_oof_auc)_self - "
            "(activation_oof_auc - bow_oof_auc)_foreign"
        ),
        "bootstrap": (
            f"{args.bootstrap_draws} paired resamples over aligned task IDs"
        ),
        "rows": len(self_ids),
        "max_reasoning_tokens": args.max_reasoning_tokens,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol": protocol,
                "condition_metrics": metric_rows,
                "transfer_metrics": transfer_rows,
                "paired_bootstrap": bootstrap_rows,
                "trigger_coverage": trigger_summaries,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
