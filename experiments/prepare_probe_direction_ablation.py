"""Fit probe directions on cases outside the intervention set, plus norm-matched random controls."""
from __future__ import annotations

import argparse
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

from cot_oversight_case_studies.io import cache_path, load_jsonl, write_jsonl  # noqa: E402
from cot_oversight_case_studies.reasoning_probe import (  # noqa: E402
    REASONING_STEPS,
    normalize_reasoning_text,
    step_presence_labels,
)


FIXED_LAYERS = {
    "identifies_vulnerability": 28,
    "traces_data_flow": 8,
    "assesses_exploitability": 28,
}


def load_layer_activations(
    activation_dir: Path,
    record_ids: list[str],
    layers: set[int],
) -> dict[int, np.ndarray]:
    values: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for record_id in record_ids:
        path = cache_path(activation_dir / "activation_cache", record_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing activation cache: {path}")
        with np.load(path, allow_pickle=False) as payload:
            cached_id = str(payload["record_id"].item())
            cached_layers = [int(item) for item in payload["layers"].tolist()]
            matrix = np.asarray(payload["activations"], dtype=np.float32)
        if cached_id != record_id:
            raise ValueError(f"Cache record mismatch: {cached_id} != {record_id}.")
        offsets = {layer: index for index, layer in enumerate(cached_layers)}
        if not layers.issubset(offsets):
            raise ValueError(f"Cache {path} does not contain {sorted(layers)}.")
        for layer in layers:
            values[layer].append(matrix[offsets[layer]])
    return {layer: np.stack(rows) for layer, rows in values.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit frozen probe directions on the 141 non-held-out security rows "
            "and construct norm-matched controls."
        )
    )
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=200)
    parser.add_argument("--expected-heldout", type=int, default=59)
    parser.add_argument("--random-controls-per-step", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> int:
    from sklearn.linear_model import LogisticRegression

    args = parse_args()
    rows = [
        row
        for row in load_jsonl(args.inference)
        if row.get("status") == "ok"
        and row.get("domain") == "domain_b"
        and row.get("variant") == "standard"
    ]
    if len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} inference rows.")
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Inference rows contain duplicate record IDs.")

    heldout_rows = load_jsonl(args.heldout_manifest)
    heldout_ids = {str(row["id"]) for row in heldout_rows}
    if len(heldout_ids) != args.expected_heldout:
        raise ValueError(f"Expected {args.expected_heldout} unique held-out IDs.")
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if str(row["id"]) not in heldout_ids],
        dtype=int,
    )
    heldout_indices = np.asarray(
        [index for index, row in enumerate(rows) if str(row["id"]) in heldout_ids],
        dtype=int,
    )
    if len(train_indices) != args.expected_rows - args.expected_heldout:
        raise ValueError("The expected 141/59 direction-fit split was not recovered.")

    labels = {
        step: np.asarray(
            [
                step_presence_labels(normalize_reasoning_text(row))[step]
                for row in rows
            ],
            dtype=int,
        )
        for step in REASONING_STEPS
    }
    layers = set(FIXED_LAYERS.values())
    activations = load_layer_activations(
        args.activation_dir,
        record_ids,
        layers,
    )
    rng = np.random.default_rng(args.random_seed)
    arrays: dict[str, np.ndarray] = {}
    plan_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for step in REASONING_STEPS:
        layer = FIXED_LAYERS[step]
        x_train = activations[layer][train_indices]
        y_train = labels[step][train_indices]
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(x_train, y_train)
        raw_direction = np.asarray(classifier.coef_[0], dtype=np.float32)
        raw_norm = float(np.linalg.norm(raw_direction))
        if not np.isfinite(raw_norm) or raw_norm <= 0:
            raise ValueError(f"Degenerate fitted direction for {step}.")
        direction = raw_direction / raw_norm
        projection_mean = float(np.mean(x_train @ direction))
        vector_key = f"probe__{step}"
        arrays[vector_key] = direction
        plan_rows.append(
            {
                "condition_id": vector_key,
                "kind": "probe_direction",
                "target_step": step,
                "layer": layer,
                "vector_key": vector_key,
                "projection_mean": projection_mean,
                "ablation_span": "prompt_prefill_and_every_generated_step",
            }
        )
        fit_rows.append(
            {
                "step_name": step,
                "layer": layer,
                "train_rows": len(train_indices),
                "heldout_rows": len(heldout_indices),
                "train_positive": int(y_train.sum()),
                "train_negative": int(len(y_train) - y_train.sum()),
                "heldout_positive": int(labels[step][heldout_indices].sum()),
                "heldout_negative": int(
                    len(heldout_indices) - labels[step][heldout_indices].sum()
                ),
                "raw_coefficient_norm": raw_norm,
                "unit_direction_norm": float(np.linalg.norm(direction)),
                "train_projection_mean": projection_mean,
            }
        )

        for control_index in range(args.random_controls_per_step):
            random_direction = rng.normal(size=direction.shape).astype(np.float32)
            random_direction /= np.linalg.norm(random_direction)
            random_key = f"random__{step}__{control_index:02d}"
            random_projection_mean = float(np.mean(x_train @ random_direction))
            arrays[random_key] = random_direction
            plan_rows.append(
                {
                    "condition_id": random_key,
                    "kind": "random_norm_matched",
                    "target_step": step,
                    "layer": layer,
                    "vector_key": random_key,
                    "projection_mean": random_projection_mean,
                    "ablation_span": "prompt_prefill_and_every_generated_step",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "directions.npz", **arrays)
    write_jsonl(args.output_dir / "intervention_plan.jsonl", plan_rows)
    write_jsonl(args.output_dir / "direction_fit_summary.jsonl", fit_rows)
    protocol = {
        "inference": str(args.inference.resolve()),
        "activation_dir": str(args.activation_dir.resolve()),
        "heldout_manifest": str(args.heldout_manifest.resolve()),
        "rows": len(rows),
        "direction_fit_rows": len(train_indices),
        "frozen_generation_rows": len(heldout_indices),
        "fixed_layers": FIXED_LAYERS,
        "layer_selection_disclosure": (
            "Layers 28/8/28 were selected by the original all-200-row CV sweep."
        ),
        "direction_fit": (
            "class_weighted_logistic_regression_on_mean_pooled_reasoning_span"
        ),
        "direction_normalization": "unit_l2",
        "ablation": "h <- h - ((h dot d) - train_mean_projection) d",
        "random_controls_per_step": args.random_controls_per_step,
        "random_seed": args.random_seed,
        "intervention_conditions": len(plan_rows),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol": protocol, "fits": fit_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
