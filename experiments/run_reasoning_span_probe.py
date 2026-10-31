"""Reasoning-step probes on teacher-forced activations over the first 256 reasoning tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
    cross_validated_probe_predictions,
    normalize_reasoning_text,
    parse_layer_list,
    relative_layer_sweep,
    sample_confirmation_indices,
    step_presence_labels,
)


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
MODEL_REVISION = "b1c0b44b4369b597ad119a196caf79a9c40e141e"
PRIMARY_LAYERS = "8,12,16,20,24,28"
SOURCE_MODEL_DEPTH = 48


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def protocol_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        return [int(token_id) for token_id in encoded.ids]
    if hasattr(encoded, "input_ids"):
        return [int(token_id) for token_id in encoded.input_ids]
    return [int(token_id) for token_id in encoded]


def load_cached_activations(
    path: Path,
    *,
    record_id: str,
    layers: list[int],
    expected_protocol_hash: str,
) -> tuple[dict[int, np.ndarray], int] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            cached_record_id = str(payload["record_id"].item())
            cached_hash = str(payload["protocol_hash"].item())
            cached_layers = [int(item) for item in payload["layers"].tolist()]
            if (
                cached_record_id != record_id
                or cached_hash != expected_protocol_hash
                or cached_layers != layers
            ):
                return None
            matrix = np.asarray(payload["activations"], dtype=np.float32)
            span_tokens = int(payload["span_token_count"].item())
    except Exception:
        return None
    return {
        layer: matrix[index]
        for index, layer in enumerate(cached_layers)
    }, span_tokens


def save_cached_activations(
    path: Path,
    *,
    record_id: str,
    layers: list[int],
    activations: dict[int, np.ndarray],
    span_token_count: int,
    expected_protocol_hash: str,
) -> None:
    matrix = np.stack([activations[layer] for layer in layers]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        record_id=np.asarray(record_id),
        protocol_hash=np.asarray(expected_protocol_hash),
        layers=np.asarray(layers, dtype=np.int32),
        activations=matrix,
        span_token_count=np.asarray(span_token_count, dtype=np.int32),
    )
    temporary.replace(path)


def extract_reasoning_span_activations(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    continuation: str,
    layers: list[int],
    max_continuation_tokens: int,
    pooling: str,
) -> tuple[dict[int, np.ndarray], int]:
    import torch

    prompt_ids = encode_without_special_tokens(tokenizer, prompt)
    continuation_ids = encode_without_special_tokens(tokenizer, continuation)
    continuation_ids = continuation_ids[:max_continuation_tokens]
    if not continuation_ids:
        return {}, 0
    combined_ids = prompt_ids + continuation_ids
    span_start = len(prompt_ids)
    span_end = span_start + len(continuation_ids)
    input_device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([combined_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)
    decoder = model.model
    decoder_layers = decoder.layers
    captured: dict[int, np.ndarray] = {}
    handles = []

    def pooled_hidden(hidden: Any) -> Any:
        span = hidden[0, span_start:span_end]
        return span[-1] if pooling == "last" else span.mean(dim=0)

    for layer in layers:
        def hook_fn(_module: Any, _inputs: Any, output: Any, *, layer_index: int = layer) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = pooled_hidden(hidden).detach().float().cpu().numpy()
            return output

        handles.append(decoder_layers[layer].register_forward_hook(hook_fn))

    try:
        with torch.inference_mode():
            decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return captured, len(continuation_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reasoning-span probe protocol used in the paper.")
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--primary-layers", default=PRIMARY_LAYERS)
    parser.add_argument("--source-model-depth", type=int, default=SOURCE_MODEL_DEPTH)
    parser.add_argument("--max-reasoning-tokens", type=int, default=256)
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
    parser.add_argument("--confirmation-size", type=int, default=50)
    parser.add_argument("--confirmation-seed", type=int, default=442)
    parser.add_argument("--expected-rows", type=int, default=200)
    parser.add_argument("--gpu-memory-gib", type=int, default=128)
    parser.add_argument(
        "--condition-label",
        default="unspecified",
        help="Frozen analysis-condition label included in the protocol hash.",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help=(
            "Use tokenizer.json directly. This preserves the original Qwen "
            "reasoning-span tokenization under newer Transformers versions."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument(
        "--disable-relative-depth-sensitivity",
        action="store_true",
        help="Run only the exact absolute-layer protocol.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in load_jsonl(args.inference)
        if row.get("status") == "ok"
        and row.get("domain") == "domain_b"
        and row.get("variant") == "standard"
    ]
    if len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} successful security rows, found {len(rows)}.")
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Inference input contains duplicate security record IDs.")

    reasoning_texts = [normalize_reasoning_text(row) for row in rows]
    labels_by_row = [step_presence_labels(text) for text in reasoning_texts]
    label_counts = {
        step: {
            "positive": int(sum(labels[step] for labels in labels_by_row)),
            "negative": int(len(rows) - sum(labels[step] for labels in labels_by_row)),
        }
        for step in REASONING_STEPS
    }
    labels_summary = {
        "inference": str(args.inference),
        "rows": len(rows),
        "label_counts": label_counts,
    }
    (args.output_dir / "label_summary.json").write_text(
        json.dumps(labels_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.labels_only:
        print(json.dumps(labels_summary, indent=2), flush=True)
        return 0

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    primary_layers = parse_layer_list(args.primary_layers)
    if args.tokenizer_json:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        tokenizer_protocol = {
            "implementation": "tokenizers.Tokenizer.from_file",
            "tokenizer_json": str(args.tokenizer_json.resolve()),
            "tokenizer_json_sha256": file_sha256(args.tokenizer_json),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.model_revision,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        tokenizer_protocol = {
            "implementation": "transformers.AutoTokenizer",
            "tokenizer_json": None,
            "tokenizer_json_sha256": None,
        }
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError("No CUDA devices are visible.")
    max_memory = {index: f"{args.gpu_memory_gib}GiB" for index in range(visible_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()
    model_depth = int(len(model.model.layers))
    if max(primary_layers) >= model_depth:
        raise ValueError(f"Primary layer sweep {primary_layers} exceeds model depth {model_depth}.")

    analysis_layers: dict[str, list[int]] = {"absolute_14b_protocol": primary_layers}
    if not args.disable_relative_depth_sensitivity:
        analysis_layers["relative_depth_sensitivity"] = relative_layer_sweep(
            primary_layers,
            source_depth=args.source_model_depth,
            target_depth=model_depth,
        )
    all_layers = sorted({layer for layers in analysis_layers.values() for layer in layers})

    activation_protocol = {
        "model": args.model,
        "model_revision": args.model_revision,
        "model_depth": model_depth,
        "condition_label": args.condition_label,
        "inference_sha256": file_sha256(args.inference),
        "tokenizer": tokenizer_protocol,
        "source_model_depth": args.source_model_depth,
        "analysis_layers": analysis_layers,
        "all_captured_layers": all_layers,
        "representation": "teacher_forced_reasoning_span",
        "prompt_rendering": "raw_prompt_add_special_tokens_false",
        "pooling": args.pooling,
        "max_reasoning_tokens": args.max_reasoning_tokens,
        "probe": "class_weighted_logistic_regression",
        "cv": "five_fold_stratified_shuffle_random_state_0",
        "confirmation_size": args.confirmation_size,
        "confirmation_seed": args.confirmation_seed,
        "confirmation_auc_threshold": 0.75,
    }
    expected_protocol_hash = protocol_hash(activation_protocol)
    protocol = {
        **activation_protocol,
        "git_commit": git_commit(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "visible_gpus": visible_gpus,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "activation_protocol_sha256": expected_protocol_hash,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )

    cache_dir = args.output_dir / "activation_cache"
    activations_by_row: list[dict[int, np.ndarray]] = []
    span_token_counts: list[int] = []
    for index, (row, reasoning_text) in enumerate(zip(rows, reasoning_texts, strict=True), start=1):
        path = cache_path(cache_dir, str(row["record_id"]))
        cached = load_cached_activations(
            path,
            record_id=str(row["record_id"]),
            layers=all_layers,
            expected_protocol_hash=expected_protocol_hash,
        )
        if cached is None:
            activations, span_token_count = extract_reasoning_span_activations(
                model,
                tokenizer,
                prompt=str(row["prompt"]),
                continuation=reasoning_text,
                layers=all_layers,
                max_continuation_tokens=args.max_reasoning_tokens,
                pooling=args.pooling,
            )
            if sorted(activations) != all_layers:
                raise RuntimeError(
                    f"Activation capture failed for {row['record_id']}: "
                    f"expected {all_layers}, got {sorted(activations)}."
                )
            save_cached_activations(
                path,
                record_id=str(row["record_id"]),
                layers=all_layers,
                activations=activations,
                span_token_count=span_token_count,
                expected_protocol_hash=expected_protocol_hash,
            )
        else:
            activations, span_token_count = cached
        activations_by_row.append(activations)
        span_token_counts.append(span_token_count)
        print(
            f"activation_rows={index}/{len(rows)} record_id={row['record_id']} "
            f"span_tokens={span_token_count} cached={cached is not None}",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()

    confirmation_indices = sample_confirmation_indices(
        rows,
        n=args.confirmation_size,
        seed=args.confirmation_seed,
    )
    per_layer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    summary_analyses: dict[str, Any] = {}

    for analysis_name, layers in analysis_layers.items():
        step_summaries: dict[str, Any] = {}
        for step in REASONING_STEPS:
            labels = [row_labels[step] for row_labels in labels_by_row]
            layer_metrics: dict[int, dict[str, Any]] = {}
            for layer in layers:
                activation_matrix = np.stack([row_activations[layer] for row_activations in activations_by_row])
                metrics = cross_validated_probe_predictions(activation_matrix, labels)
                layer_metrics[layer] = metrics
                per_layer_rows.append(
                    {
                        "analysis": analysis_name,
                        "step_name": step,
                        "layer_idx": layer,
                        "probe_auc_mean": metrics["auc_mean"],
                        "probe_auc_std": metrics["auc_std"],
                        "probe_ap_mean": metrics["ap_mean"],
                        "probe_ap_std": metrics["ap_std"],
                        "cv_folds": metrics["cv_folds"],
                        "positive_count": int(sum(labels)),
                        "negative_count": int(len(labels) - sum(labels)),
                    }
                )
            best_layer = max(layers, key=lambda layer: layer_metrics[layer]["auc_mean"])
            best = layer_metrics[best_layer]
            probabilities = np.asarray(best["probabilities"])
            for row, label, probability, span_tokens in zip(
                rows,
                labels,
                probabilities.tolist(),
                span_token_counts,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "analysis": analysis_name,
                        "record_id": row["record_id"],
                        "id": row["id"],
                        "step_name": step,
                        "layer_idx": best_layer,
                        "predicted_presence_prob": float(probability),
                        "step_present_in_cot": int(label),
                        "is_correct": row.get("is_correct"),
                        "probe_span_token_count": span_tokens,
                    }
                )

            confirmed_count = 0
            present_count = 0
            for row_index in confirmation_indices:
                row = rows[row_index]
                label = int(labels[row_index])
                probability = float(probabilities[row_index])
                confirmed = bool(best["auc_mean"] >= 0.75 and probability >= 0.5 and label == 1)
                present_count += label
                confirmed_count += int(confirmed)
                confirmation_rows.append(
                    {
                        "analysis": analysis_name,
                        "record_id": row["record_id"],
                        "id": row["id"],
                        "step_name": step,
                        "layer_idx": best_layer,
                        "probe_auc_mean": best["auc_mean"],
                        "probe_auc_std": best["auc_std"],
                        "probe_ap_mean": best["ap_mean"],
                        "probe_ap_std": best["ap_std"],
                        "predicted_presence_prob": probability,
                        "step_present_in_cot": label,
                        "probe_supported_confirmation": confirmed,
                        "is_correct": row.get("is_correct"),
                        "probe_span_token_count": span_token_counts[row_index],
                        "reasoning_excerpt": reasoning_texts[row_index][:280],
                    }
                )
            step_summaries[step] = {
                "best_layer": best_layer,
                "probe_auc_mean": best["auc_mean"],
                "probe_auc_std": best["auc_std"],
                "probe_ap_mean": best["ap_mean"],
                "probe_ap_std": best["ap_std"],
                "positive_count": int(sum(labels)),
                "negative_count": int(len(labels) - sum(labels)),
                "confirmation_n": len(confirmation_indices),
                "confirmation_step_present_n": present_count,
                "confirmation_supported_n": confirmed_count,
            }
        summary_analyses[analysis_name] = step_summaries

    write_jsonl(args.output_dir / "per_layer_metrics.jsonl", per_layer_rows)
    write_jsonl(args.output_dir / "row_predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "confirmation_rows.jsonl", confirmation_rows)
    summary = {
        "protocol": protocol,
        "rows": len(rows),
        "correctly_solved_rows": int(sum(row.get("is_correct") is True for row in rows)),
        "label_counts": label_counts,
        "analyses": summary_analyses,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
