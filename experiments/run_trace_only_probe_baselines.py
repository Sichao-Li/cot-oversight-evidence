"""Text-only baselines on the same traces, labels, spans and folds as the activation probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _path in (EXPERIMENTS_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cot_oversight_case_studies.io import load_jsonl, write_jsonl  # noqa: E402
from cot_oversight_case_studies.reasoning_probe import (  # noqa: E402
    REASONING_STEPS,
    normalize_reasoning_text,
    step_presence_labels,
)


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"


def sha256_json(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def first_satisfying_prefix_token(
    tokenizer: Any,
    token_ids: list[int],
    *,
    step_name: str,
) -> int | None:
    """Return the first token prefix that satisfies one frozen step rule."""

    if not token_ids or step_presence_labels(tokenizer.decode(token_ids))[step_name] != 1:
        return None
    for token_count in range(1, len(token_ids) + 1):
        prefix = tokenizer.decode(token_ids[:token_count], skip_special_tokens=False)
        if step_presence_labels(prefix)[step_name] == 1:
            return token_count
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe trace-only baselines for the reasoning-step probes."
    )
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default=MODEL_ID)
    parser.add_argument("--tokenizer-revision", default=MODEL_REVISION)
    parser.add_argument("--max-reasoning-tokens", type=int, default=256)
    parser.add_argument("--expected-rows", type=int, default=200)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_raw_tokenizer(
    tokenizer_source: str,
    *,
    revision: str,
    local_files_only: bool,
) -> tuple[Any, str]:
    """Load tokenizer.json directly, avoiding model-class tokenizer rewrites."""

    from tokenizers import Tokenizer

    source_path = Path(tokenizer_source)
    if source_path.is_dir():
        tokenizer_path = source_path / "tokenizer.json"
    elif source_path.is_file():
        tokenizer_path = source_path
    else:
        from huggingface_hub import hf_hub_download

        tokenizer_path = Path(
            hf_hub_download(
                repo_id=tokenizer_source,
                filename="tokenizer.json",
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path)), str(tokenizer_path)


def main() -> int:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in load_jsonl(args.inference)
        if row.get("status") == "ok"
        and row.get("domain") == "domain_b"
        and row.get("variant") == "standard"
    ]
    if len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} security rows, found {len(rows)}.")
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Inference input contains duplicate record IDs.")

    tokenizer, tokenizer_path = load_raw_tokenizer(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    full_texts = [normalize_reasoning_text(row) for row in rows]
    token_ids = [
        [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False).ids]
        for text in full_texts
    ]
    matched_texts = [
        tokenizer.decode(ids[: args.max_reasoning_tokens], skip_special_tokens=False)
        for ids in token_ids
    ]
    labels_by_row = [step_presence_labels(text) for text in full_texts]

    # Binary word presence is primary because the frozen targets are themselves
    # presence rules. Count and TF-IDF unigrams are reported as sensitivities.
    vectorizers = {
        "count_unigram": CountVectorizer(ngram_range=(1, 1)),
        "binary_unigram_primary": CountVectorizer(ngram_range=(1, 1), binary=True),
        "tfidf_unigram": TfidfVectorizer(ngram_range=(1, 1)),
    }
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []

    for step_name in REASONING_STEPS:
        y = np.asarray([labels[step_name] for labels in labels_by_row], dtype=int)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        splits = list(cv.split(np.zeros(len(y)), y))
        for fold_index, (_train_indices, test_indices) in enumerate(splits):
            for row_index in test_indices.tolist():
                fold_rows.append(
                    {
                        "record_id": record_ids[row_index],
                        "step_name": step_name,
                        "fold": fold_index,
                        "label": int(y[row_index]),
                    }
                )

        positive_trigger_tokens: list[int] = []
        for row_index, ids in enumerate(token_ids):
            if y[row_index] != 1:
                continue
            trigger_token = first_satisfying_prefix_token(
                tokenizer,
                ids,
                step_name=step_name,
            )
            if trigger_token is not None:
                positive_trigger_tokens.append(trigger_token)
            trigger_rows.append(
                {
                    "record_id": record_ids[row_index],
                    "step_name": step_name,
                    "full_trace_label": 1,
                    "first_satisfying_prefix_token": trigger_token,
                    "rule_satisfied_within_probe_span": bool(
                        trigger_token is not None
                        and trigger_token <= args.max_reasoning_tokens
                    ),
                    "full_trace_token_count": len(ids),
                }
            )

        for baseline_name, vectorizer in vectorizers.items():
            probabilities = np.zeros(len(rows), dtype=float)
            fold_aucs: list[float] = []
            fold_aps: list[float] = []
            for fold_index, (train_indices, test_indices) in enumerate(splits):
                pipeline = Pipeline(
                    [
                        ("text", vectorizer),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=1000,
                                class_weight="balanced",
                            ),
                        ),
                    ]
                )
                train_texts = [matched_texts[index] for index in train_indices]
                test_texts = [matched_texts[index] for index in test_indices]
                pipeline.fit(train_texts, y[train_indices])
                fold_probabilities = pipeline.predict_proba(test_texts)[:, 1]
                probabilities[test_indices] = fold_probabilities
                fold_aucs.append(
                    float(roc_auc_score(y[test_indices], fold_probabilities))
                )
                fold_aps.append(
                    float(average_precision_score(y[test_indices], fold_probabilities))
                )
                for row_index, probability in zip(
                    test_indices.tolist(),
                    fold_probabilities.tolist(),
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "record_id": record_ids[row_index],
                            "step_name": step_name,
                            "baseline": baseline_name,
                            "fold": fold_index,
                            "label": int(y[row_index]),
                            "predicted_presence_prob": float(probability),
                            "matched_span_token_count": min(
                                len(token_ids[row_index]),
                                args.max_reasoning_tokens,
                            ),
                        }
                    )

            within_span = sum(
                trigger is not None and trigger <= args.max_reasoning_tokens
                for trigger in positive_trigger_tokens
            )
            metrics_rows.append(
                {
                    "step_name": step_name,
                    "baseline": baseline_name,
                    "positive_count": int(y.sum()),
                    "negative_count": int(len(y) - y.sum()),
                    "fold_auc_mean": float(np.mean(fold_aucs)),
                    "fold_auc_std": float(np.std(fold_aucs)),
                    "fold_ap_mean": float(np.mean(fold_aps)),
                    "fold_ap_std": float(np.std(fold_aps)),
                    "oof_auc": float(roc_auc_score(y, probabilities)),
                    "oof_ap": float(average_precision_score(y, probabilities)),
                    "positive_rule_trigger_observed": len(positive_trigger_tokens),
                    "positive_rule_trigger_within_span": int(within_span),
                    "positive_rule_trigger_within_span_fraction": (
                        float(within_span / int(y.sum()))
                        if int(y.sum()) > 0
                        else None
                    ),
                }
            )

    protocol = {
        "inference": str(args.inference),
        "rows": len(rows),
        "tokenizer": args.tokenizer,
        "resolved_tokenizer_json": tokenizer_path,
        "tokenizer_revision": args.tokenizer_revision,
        "max_reasoning_tokens": args.max_reasoning_tokens,
        "labels": "frozen_audit_refined_full_trace_step_presence_rules",
        "primary_baseline": "binary_unigram_primary",
        "classifier": "logistic_regression_max_iter_1000_class_weight_balanced",
        "cv": "five_fold_stratified_shuffle_random_state_0",
        "vectorizer_fitted_inside_training_fold": True,
        "mean_full_trace_tokens": float(np.mean([len(ids) for ids in token_ids])),
        "traces_exceeding_probe_span": int(
            sum(len(ids) > args.max_reasoning_tokens for ids in token_ids)
        ),
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "metrics.jsonl", metrics_rows)
    write_jsonl(args.output_dir / "oof_predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "fold_assignments.jsonl", fold_rows)
    write_jsonl(args.output_dir / "rule_trigger_positions.jsonl", trigger_rows)
    print(json.dumps({"protocol": protocol, "metrics": metrics_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
