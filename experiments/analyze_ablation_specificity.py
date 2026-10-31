"""Cross-step specificity: does each fitted direction suppress its own step more than the other two?"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _path in (EXPERIMENTS_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cot_oversight_case_studies.io import load_jsonl, sha256_file  # noqa: E402
from analyze_fresh_probe_direction_ablation import paired_bootstrap  # noqa: E402
from cot_oversight_case_studies.reasoning_probe import (  # noqa: E402
    REASONING_STEPS,
    normalize_reasoning_text,
    step_presence_labels,
)


EXPECTED_BASELINE_POSITIVES = {
    "identifies_vulnerability": 27,
    "traces_data_flow": 25,
    "assesses_exploitability": 29,
}


def bootstrap_case_statistic(
    *,
    case_count: int,
    statistic: Callable[[np.ndarray], float],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Use the existing case-resampling protocol for a non-linear paired statistic."""

    if case_count < 1:
        raise ValueError("At least one case is required.")
    rng = np.random.default_rng(seed)
    full_indices = np.arange(case_count)
    point = float(statistic(full_indices))
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = rng.integers(0, case_count, size=case_count)
        estimates[draw] = float(statistic(indices))
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


def ci_excludes_zero(result: dict[str, Any]) -> bool:
    lower, upper = result["ci95"]
    return bool(lower > 0.0 or upper < 0.0)


def ci_spans_zero(result: dict[str, Any]) -> bool:
    lower, upper = result["ci95"]
    return bool(lower <= 0.0 <= upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the frozen 3x3 probe-direction step-specificity analysis."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument("--intervention-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=59)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labeler_path = SRC_DIR / "cot_oversight_case_studies" / "reasoning_probe.py"
    labeler_hash = sha256_file(labeler_path)

    baseline_rows = [row for row in load_jsonl(args.baseline) if row.get("status") == "ok"]
    baseline = {str(row["id"]): row for row in baseline_rows}
    if len(baseline_rows) != args.expected_rows or len(baseline) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} unique baseline rows; found "
            f"{len(baseline_rows)} rows and {len(baseline)} IDs."
        )

    plans = {
        str(row["condition_id"]): row for row in load_jsonl(args.intervention_plan)
    }
    intervention_rows = [
        row
        for row in load_jsonl(args.interventions)
        if row.get("status") == "ok" and row.get("condition_id")
    ]
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in intervention_rows:
        condition_id = str(row["condition_id"])
        case_id = str(row["id"])
        if condition_id not in plans:
            raise ValueError(f"Unknown condition {condition_id}.")
        if case_id in by_condition[condition_id]:
            raise ValueError(f"Duplicate generation {condition_id}/{case_id}.")
        by_condition[condition_id][case_id] = row

    baseline_ids = set(baseline)
    missing_generations: list[dict[str, Any]] = []
    for condition_id in sorted(plans):
        condition_ids = set(by_condition.get(condition_id, {}))
        for case_id in sorted(baseline_ids - condition_ids):
            missing_generations.append(
                {"condition_id": condition_id, "id": case_id, "kind": "missing"}
            )
        for case_id in sorted(condition_ids - baseline_ids):
            missing_generations.append(
                {"condition_id": condition_id, "id": case_id, "kind": "extra"}
            )
    if missing_generations:
        raise ValueError(
            "Stored generations are incomplete; pairwise analysis stopped. "
            f"Details: {missing_generations}"
        )

    case_ids = sorted(baseline)
    baseline_labels = {
        case_id: step_presence_labels(normalize_reasoning_text(baseline[case_id]))
        for case_id in case_ids
    }
    condition_labels = {
        condition_id: {
            case_id: step_presence_labels(
                normalize_reasoning_text(by_condition[condition_id][case_id])
            )
            for case_id in case_ids
        }
        for condition_id in sorted(plans)
    }

    baseline_positive_counts = {
        step: sum(baseline_labels[case_id][step] for case_id in case_ids)
        for step in REASONING_STEPS
    }
    if baseline_positive_counts != EXPECTED_BASELINE_POSITIVES:
        raise RuntimeError(
            "Frozen-labeler sanity gate failed: expected "
            f"{EXPECTED_BASELINE_POSITIVES}, observed {baseline_positive_counts}."
        )

    probe_plans = {
        str(plan["target_step"]): plan
        for plan in plans.values()
        if plan["kind"] == "probe_direction"
    }
    if set(probe_plans) != set(REASONING_STEPS):
        raise ValueError(f"Unexpected probe families: {sorted(probe_plans)}")

    suppression_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    addition_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    family_step_contrasts: dict[str, dict[str, np.ndarray]] = {}
    family_step_masks: dict[str, dict[str, np.ndarray]] = {}
    random_ids_by_family: dict[str, list[str]] = {}

    for family_index, family in enumerate(REASONING_STEPS):
        probe_id = str(probe_plans[family]["condition_id"])
        random_ids = sorted(
            condition_id
            for condition_id, plan in plans.items()
            if plan["kind"] == "random_norm_matched"
            and str(plan["target_step"]) == family
        )
        if len(random_ids) != 5:
            raise ValueError(f"Expected five random controls for {family}; found {random_ids}.")
        random_ids_by_family[family] = random_ids
        suppression_matrix[family] = {}
        addition_matrix[family] = {}
        family_step_contrasts[family] = {}
        family_step_masks[family] = {}

        for step_index, step in enumerate(REASONING_STEPS):
            present = np.asarray(
                [baseline_labels[case_id][step] == 1 for case_id in case_ids],
                dtype=bool,
            )
            absent = ~present
            probe_values = np.asarray(
                [condition_labels[probe_id][case_id][step] for case_id in case_ids],
                dtype=int,
            )
            random_values = np.stack(
                [
                    np.asarray(
                        [condition_labels[random_id][case_id][step] for case_id in case_ids],
                        dtype=int,
                    )
                    for random_id in random_ids
                ]
            )

            probe_suppression = present & (probe_values == 0)
            random_suppression = np.mean(
                present[np.newaxis, :] & (random_values == 0), axis=0
            )
            suppression_contrast = probe_suppression.astype(float) - random_suppression
            family_step_contrasts[family][step] = suppression_contrast
            family_step_masks[family][step] = present
            suppression_result = paired_bootstrap(
                suppression_contrast[present],
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + family_index * 100 + step_index,
            )
            suppression_matrix[family][step] = {
                "eligible_n": int(np.sum(present)),
                "probe_suppression_rate": float(np.mean(probe_suppression[present])),
                "random_mean_suppression_rate": float(np.mean(random_suppression[present])),
                "probe_minus_random": suppression_result,
            }

            probe_addition = absent & (probe_values == 1)
            random_addition = np.mean(
                absent[np.newaxis, :] & (random_values == 1), axis=0
            )
            addition_contrast = probe_addition.astype(float) - random_addition
            addition_result = paired_bootstrap(
                addition_contrast[absent],
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 1000 + family_index * 100 + step_index,
            )
            addition_matrix[family][step] = {
                "eligible_n": int(np.sum(absent)),
                "probe_addition_rate": float(np.mean(probe_addition[absent])),
                "random_mean_addition_rate": float(np.mean(random_addition[absent])),
                "probe_minus_random": addition_result,
            }

    selectivity: dict[str, dict[str, Any]] = {}
    for family_index, family in enumerate(REASONING_STEPS):
        target = family
        non_targets = [step for step in REASONING_STEPS if step != target]

        def selectivity_stat(indices: np.ndarray) -> float:
            effects: dict[str, float] = {}
            for step in REASONING_STEPS:
                mask = family_step_masks[family][step][indices]
                if not np.any(mask):
                    raise RuntimeError(f"Bootstrap draw has no eligible rows for {step}.")
                effects[step] = float(
                    np.mean(family_step_contrasts[family][step][indices][mask])
                )
            return effects[target] - float(
                np.mean([effects[step] for step in non_targets])
            )

        selectivity[family] = bootstrap_case_statistic(
            case_count=len(case_ids),
            statistic=selectivity_stat,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed + 2000 + family_index,
        )

    identification_family = "identifies_vulnerability"
    identification_step = "identifies_vulnerability"
    identification_probe_id = str(
        probe_plans[identification_family]["condition_id"]
    )
    baseline_token_counts = np.asarray(
        [int(baseline[case_id]["generated_token_count"]) for case_id in case_ids]
    )
    probe_token_counts = np.asarray(
        [
            int(by_condition[identification_probe_id][case_id]["generated_token_count"])
            for case_id in case_ids
        ]
    )
    identification_present = family_step_masks[identification_family][identification_step]
    nonshrinking = probe_token_counts - baseline_token_counts >= 0
    length_mask = identification_present & nonshrinking
    length_contrast = family_step_contrasts[identification_family][identification_step]
    length_controlled = {
        "definition": (
            "baseline identification present and identification-probe output token "
            "count greater than or equal to baseline; random controls evaluated on "
            "the same paired cases"
        ),
        "eligible_n": int(np.sum(length_mask)),
        "probe_minus_random": paired_bootstrap(
            length_contrast[length_mask],
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed + 3000,
        ),
    }

    ident_diagonal = suppression_matrix[identification_family][identification_step][
        "probe_minus_random"
    ]
    ident_off_diagonal = [
        suppression_matrix[identification_family][step]["probe_minus_random"]
        for step in REASONING_STEPS
        if step != identification_step
    ]
    ident_selectivity = selectivity[identification_family]
    selective_rule = bool(
        ident_diagonal["point_estimate"] > 0.0
        and ci_excludes_zero(ident_diagonal)
        and all(ci_spans_zero(result) for result in ident_off_diagonal)
        and ident_selectivity["point_estimate"] > 0.0
        and ci_excludes_zero(ident_selectivity)
    )
    generic_degradation_rule = ci_spans_zero(ident_selectivity)
    if selective_rule:
        decision = "suggestive_selective_causal_involvement_uncorrected"
    elif generic_degradation_rule:
        decision = "direction_sensitive_generation_degradation_not_step_selective"
    else:
        decision = "mixed_specificity_evidence_no_prestated_category"

    result = {
        "analysis": "frozen direction-by-step specificity matrix",
        "input_hashes": {
            "baseline_sha256": sha256_file(args.baseline),
            "interventions_sha256": sha256_file(args.interventions),
            "intervention_plan_sha256": sha256_file(args.intervention_plan),
        },
        "labeler": {
            "path": str(labeler_path),
            "sha256": labeler_hash,
            "normalization": "normalize_reasoning_text",
            "rule": "step_presence_labels",
        },
        "rows": len(case_ids),
        "conditions": len(plans),
        "missing_generations": missing_generations,
        "baseline_positive_sanity_gate": {
            "expected": EXPECTED_BASELINE_POSITIVES,
            "observed": baseline_positive_counts,
            "passed": True,
        },
        "random_condition_ids": random_ids_by_family,
        "suppression_matrix": suppression_matrix,
        "selectivity": selectivity,
        "addition_matrix": addition_matrix,
        "identification_length_controlled_sensitivity": length_controlled,
        "pre_stated_decision_rule_result": {
            "decision": decision,
            "selective_rule_passed": selective_rule,
            "generic_degradation_rule_passed": generic_degradation_rule,
        },
        "multiplicity_guardrail": (
            "All intervals and p-values are exploratory and uncorrected; the "
            "six-comparison multiplicity caveat remains in force."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "specificity_matrix.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    readable = {
        "identifies_vulnerability": "Identification",
        "traces_data_flow": "Data flow",
        "assesses_exploitability": "Exploitability",
    }
    table_lines = [
        f"% labeler_sha256: {labeler_hash}",
        "% Estimates are probe-direction minus the mean of five norm-matched controls.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Direction family & Identification & Data flow & Exploitability & $S_f$ \\\\",
        "\\midrule",
    ]
    for family in REASONING_STEPS:
        cells: list[str] = []
        for step in REASONING_STEPS:
            estimate = suppression_matrix[family][step]["probe_minus_random"][
                "point_estimate"
            ]
            formatted = f"{estimate:+.3f}"
            cells.append(f"\\textbf{{{formatted}}}" if family == step else formatted)
        selectivity_estimate = selectivity[family]["point_estimate"]
        table_lines.append(
            f"{readable[family]} & " + " & ".join(cells) + f" & {selectivity_estimate:+.3f} \\\\"
        )
    table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "% All confidence intervals and eligible counts are in specificity_matrix.json.",
        ]
    )
    (args.output_dir / "specificity_matrix_table.tex").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
