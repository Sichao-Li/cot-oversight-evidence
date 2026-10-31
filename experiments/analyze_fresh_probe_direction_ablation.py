"""Compare probe-direction and random-direction ablations against a fresh no-intervention baseline."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from verdict_parser import explicit_security_verdict  # noqa: E402
from cot_oversight_case_studies.reasoning_probe import (  # noqa: E402
    normalize_reasoning_text,
    step_presence_labels,
)


def paired_bootstrap(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired per-case contrast without breaking condition pairing."""

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Paired bootstrap requires a non-empty one-dimensional array.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, len(values), size=len(values))
        estimates[draw] = float(np.mean(values[indices]))
    point = float(np.mean(values))
    return {
        "point_estimate": point,
        "ci95": [
            float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)),
        ],
        "one_sided_p_greater_than_zero": float(
            (1 + np.sum(estimates <= 0.0)) / (draws + 1)
        ),
        "two_sided_p": float(
            min(
                1.0,
                2
                * min(
                    (1 + np.sum(estimates <= 0.0)) / (draws + 1),
                    (1 + np.sum(estimates >= 0.0)) / (draws + 1),
                ),
            )
        ),
        "bootstrap_draws": draws,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fresh probe-direction ablations with a no-intervention "
            "baseline and norm-matched random controls."
        )
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--interventions", type=Path, nargs="+", required=True)
    parser.add_argument("--intervention-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=59)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_rows = [
        row for row in load_jsonl(args.baseline, missing_ok=True) if row.get("status") == "ok"
    ]
    baseline = {str(row["id"]): row for row in baseline_rows}
    if len(baseline_rows) != args.expected_rows or len(baseline) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} unique baseline rows; "
            f"found {len(baseline_rows)} rows and {len(baseline)} IDs."
        )

    plans = {
        str(row["condition_id"]): row
        for row in load_jsonl(args.intervention_plan, missing_ok=True)
    }
    if not plans:
        raise ValueError("The intervention plan is empty.")
    intervention_rows = [
        row
        for path in args.interventions
        for row in load_jsonl(path, missing_ok=True)
        if row.get("status") == "ok" and row.get("condition_id")
    ]
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in intervention_rows:
        condition_id = str(row["condition_id"])
        case_id = str(row["id"])
        if condition_id not in plans:
            raise ValueError(f"Unknown intervention condition: {condition_id}.")
        if case_id in by_condition[condition_id]:
            raise ValueError(f"Duplicate intervention row: {condition_id}/{case_id}.")
        by_condition[condition_id][case_id] = row
    missing_conditions = sorted(set(plans) - set(by_condition))
    if missing_conditions:
        raise ValueError(f"Missing intervention conditions: {missing_conditions}.")

    baseline_ids = set(baseline)
    per_case_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    condition_vectors: dict[str, dict[str, np.ndarray]] = {}
    case_ids = sorted(baseline)

    for condition_id in sorted(plans):
        rows = by_condition[condition_id]
        if set(rows) != baseline_ids:
            missing = sorted(baseline_ids - set(rows))
            extra = sorted(set(rows) - baseline_ids)
            raise ValueError(
                f"Incomplete condition {condition_id}: missing={missing}, extra={extra}."
            )
        plan = plans[condition_id]
        target_step = str(plan["target_step"])
        answer_flips: list[float] = []
        both_parseable_values: list[float] = []
        parse_failures: list[float] = []
        target_suppression: list[float] = []
        target_eligible: list[float] = []
        token_differences: list[float] = []

        for case_id in case_ids:
            base = baseline[case_id]
            changed = rows[case_id]
            baseline_verdict = explicit_security_verdict(base.get("raw_output"))
            changed_verdict = explicit_security_verdict(changed.get("raw_output"))
            both_parseable = baseline_verdict is not None and changed_verdict is not None
            answer_flip = float(both_parseable and baseline_verdict != changed_verdict)
            parse_failure = float(changed_verdict is None)
            baseline_labels = base.get("generated_step_presence_labels")
            if not isinstance(baseline_labels, dict):
                baseline_labels = step_presence_labels(normalize_reasoning_text(base))
            changed_labels = changed.get("generated_step_presence_labels")
            if not isinstance(changed_labels, dict):
                changed_labels = step_presence_labels(normalize_reasoning_text(changed))
            eligible = float(int(baseline_labels[target_step]) == 1)
            suppressed = float(
                eligible == 1.0 and int(changed_labels[target_step]) == 0
            )
            token_difference = float(
                int(changed.get("generated_token_count") or 0)
                - int(base.get("generated_token_count") or 0)
            )

            answer_flips.append(answer_flip)
            both_parseable_values.append(float(both_parseable))
            parse_failures.append(parse_failure)
            target_suppression.append(suppressed)
            target_eligible.append(eligible)
            token_differences.append(token_difference)
            per_case_rows.append(
                {
                    "condition_id": condition_id,
                    "kind": plan["kind"],
                    "target_step": target_step,
                    "layer": int(plan["layer"]),
                    "id": case_id,
                    "baseline_verdict": baseline_verdict,
                    "intervention_verdict": changed_verdict,
                    "both_verdicts_parseable": both_parseable,
                    "answer_flip": bool(answer_flip),
                    "intervention_parse_failure": bool(parse_failure),
                    "baseline_target_step_present": bool(eligible),
                    "target_step_suppressed": bool(suppressed),
                    "generated_token_difference": int(token_difference),
                }
            )

        arrays = {
            "answer_flip": np.asarray(answer_flips),
            "both_parseable": np.asarray(both_parseable_values),
            "parse_failure": np.asarray(parse_failures),
            "target_suppression": np.asarray(target_suppression),
            "target_eligible": np.asarray(target_eligible),
        }
        condition_vectors[condition_id] = arrays
        eligible_count = int(np.sum(arrays["target_eligible"]))
        parseable_count = int(np.sum(arrays["both_parseable"]))
        condition_rows.append(
            {
                "condition_id": condition_id,
                "kind": plan["kind"],
                "target_step": target_step,
                "layer": int(plan["layer"]),
                "rows": len(case_ids),
                "baseline_and_intervention_parseable": parseable_count,
                "answer_flip_count": int(np.sum(arrays["answer_flip"])),
                "answer_flip_fraction_all_rows": float(
                    np.mean(arrays["answer_flip"])
                ),
                "answer_flip_fraction_parseable": (
                    float(np.sum(arrays["answer_flip"]) / parseable_count)
                    if parseable_count
                    else None
                ),
                "intervention_parse_failure_count": int(
                    np.sum(arrays["parse_failure"])
                ),
                "target_step_baseline_positive": eligible_count,
                "target_step_suppression_count": int(
                    np.sum(arrays["target_suppression"])
                ),
                "target_step_suppression_fraction": (
                    float(np.sum(arrays["target_suppression"]) / eligible_count)
                    if eligible_count
                    else None
                ),
                "mean_generated_token_difference": float(
                    np.mean(token_differences)
                ),
                "median_generated_token_difference": float(
                    np.median(token_differences)
                ),
            }
        )

    comparisons: list[dict[str, Any]] = []
    probe_plans = [row for row in plans.values() if row["kind"] == "probe_direction"]
    for comparison_index, probe_plan in enumerate(
        sorted(probe_plans, key=lambda row: str(row["target_step"]))
    ):
        probe_id = str(probe_plan["condition_id"])
        target_step = str(probe_plan["target_step"])
        random_ids = sorted(
            condition_id
            for condition_id, plan in plans.items()
            if plan["kind"] == "random_norm_matched"
            and str(plan["target_step"]) == target_step
        )
        if not random_ids:
            raise ValueError(f"No random controls found for {target_step}.")
        random_flip = np.mean(
            np.stack(
                [condition_vectors[condition_id]["answer_flip"] for condition_id in random_ids]
            ),
            axis=0,
        )
        random_suppression = np.mean(
            np.stack(
                [
                    condition_vectors[condition_id]["target_suppression"]
                    for condition_id in random_ids
                ]
            ),
            axis=0,
        )
        eligible = condition_vectors[probe_id]["target_eligible"] == 1.0
        answer_contrast = (
            condition_vectors[probe_id]["answer_flip"] - random_flip
        )
        comparison: dict[str, Any] = {
            "target_step": target_step,
            "probe_condition_id": probe_id,
            "random_condition_ids": random_ids,
            "random_controls": len(random_ids),
            "answer_flip_probe_minus_random": paired_bootstrap(
                answer_contrast,
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 10 * comparison_index,
            ),
        }
        if np.any(eligible):
            suppression_contrast = (
                condition_vectors[probe_id]["target_suppression"][eligible]
                - random_suppression[eligible]
            )
            comparison["target_suppression_probe_minus_random"] = paired_bootstrap(
                suppression_contrast,
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 10 * comparison_index + 1,
            )
            comparison["target_suppression_eligible_rows"] = int(np.sum(eligible))
        else:
            comparison["target_suppression_probe_minus_random"] = None
            comparison["target_suppression_eligible_rows"] = 0
        comparisons.append(comparison)

    baseline_parseable = sum(
        explicit_security_verdict(row.get("raw_output")) is not None
        for row in baseline.values()
    )
    summary = {
        "design": (
            "fresh no-intervention baseline paired by case with mean ablation "
            "of frozen probe directions and five norm-matched random controls"
        ),
        "baseline_rows": len(baseline),
        "baseline_parseable_verdicts": baseline_parseable,
        "intervention_conditions": len(plans),
        "comparisons": comparisons,
        "causal_interpretation_guardrail": (
            "A probe-over-random effect supports causal involvement of the fitted "
            "direction under mean ablation. A null remains inconclusive because "
            "the representation may be distributed or redundant."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "condition_summary.jsonl", condition_rows)
    write_jsonl(args.output_dir / "per_case_changes.jsonl", per_case_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
