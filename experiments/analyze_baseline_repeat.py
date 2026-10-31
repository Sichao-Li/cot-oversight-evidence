"""Compare repeated no-intervention baselines to measure run-to-run generation variation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _path in (EXPERIMENTS_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cot_oversight_case_studies.io import load_jsonl, sha256_file  # noqa: E402
from verdict_parser import explicit_security_verdict  # noqa: E402
from cot_oversight_case_studies.reasoning_probe import (  # noqa: E402
    REASONING_STEPS,
    normalize_reasoning_text,
    step_presence_labels,
)


def first_divergence(left: list[int], right: list[int]) -> int | None:
    """Return the zero-based first differing token, including prefix termination."""

    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def load_run(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows = [row for row in load_jsonl(path) if row.get("status") == "ok"]
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        case_id = str(row["id"])
        if case_id in by_id:
            duplicates.append(case_id)
        by_id[case_id] = row
    if duplicates:
        raise ValueError(f"Duplicate case IDs in {path}: {sorted(set(duplicates))}")
    return by_id, sorted(by_id)


def compare_runs(
    *,
    left_name: str,
    left: dict[str, dict[str, Any]],
    right_name: str,
    right: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    left_ids = set(left)
    right_ids = set(right)
    common_ids = sorted(left_ids & right_ids)
    missing = {
        f"missing_from_{left_name}": sorted(right_ids - left_ids),
        f"missing_from_{right_name}": sorted(left_ids - right_ids),
    }

    exact_matches = 0
    first_divergences: dict[str, int] = {}
    left_parseable = 0
    right_parseable = 0
    jointly_parseable = 0
    verdict_agreements = 0
    verdict_flips = 0
    step_agreements = {step: 0 for step in REASONING_STEPS}

    for case_id in common_ids:
        left_text = str(left[case_id].get("raw_output") or "")
        right_text = str(right[case_id].get("raw_output") or "")
        if left_text == right_text:
            exact_matches += 1
        else:
            left_tokens = tokenizer(left_text, add_special_tokens=False).input_ids
            right_tokens = tokenizer(right_text, add_special_tokens=False).input_ids
            divergence = first_divergence(left_tokens, right_tokens)
            if divergence is None:
                raise RuntimeError("Different text unexpectedly produced identical token IDs.")
            first_divergences[case_id] = divergence

        left_verdict = explicit_security_verdict(left_text)
        right_verdict = explicit_security_verdict(right_text)
        left_parseable += int(left_verdict is not None)
        right_parseable += int(right_verdict is not None)
        both_parseable = left_verdict is not None and right_verdict is not None
        jointly_parseable += int(both_parseable)
        if both_parseable:
            verdict_agreements += int(left_verdict == right_verdict)
            verdict_flips += int(left_verdict != right_verdict)

        left_labels = step_presence_labels(normalize_reasoning_text(left[case_id]))
        right_labels = step_presence_labels(normalize_reasoning_text(right[case_id]))
        for step in REASONING_STEPS:
            step_agreements[step] += int(left_labels[step] == right_labels[step])

    common_n = len(common_ids)
    return {
        "left_run": left_name,
        "right_run": right_name,
        "left_rows": len(left),
        "right_rows": len(right),
        "common_rows": common_n,
        "missing_rows": missing,
        "exact_text_identity_count": exact_matches,
        "exact_text_identity_rate": exact_matches / common_n if common_n else None,
        "divergent_text_count": common_n - exact_matches,
        "first_divergence_token_position_zero_based": first_divergences,
        "first_divergence_summary": {
            "minimum": min(first_divergences.values()) if first_divergences else None,
            "median": (
                sorted(first_divergences.values())[len(first_divergences) // 2]
                if first_divergences
                else None
            ),
            "maximum": max(first_divergences.values()) if first_divergences else None,
        },
        "verdict_parseability": {
            left_name: left_parseable,
            right_name: right_parseable,
            "jointly_parseable": jointly_parseable,
        },
        "verdict_agreement_among_jointly_parseable": {
            "count": verdict_agreements,
            "denominator": jointly_parseable,
            "rate": verdict_agreements / jointly_parseable if jointly_parseable else None,
        },
        "verdict_flip": {
            "count": verdict_flips,
            "denominator_all_common": common_n,
            "rate_all_common": verdict_flips / common_n if common_n else None,
            "denominator_jointly_parseable": jointly_parseable,
            "rate_jointly_parseable": (
                verdict_flips / jointly_parseable if jointly_parseable else None
            ),
            "definition": "flip only when both explicit verdicts are parseable",
        },
        "step_presence_agreement": {
            step: {
                "count": step_agreements[step],
                "denominator": common_n,
                "rate": step_agreements[step] / common_n if common_n else None,
            }
            for step in REASONING_STEPS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure repeat-baseline generation and verdict agreement."
    )
    parser.add_argument("--baseline-one", type=Path, required=True)
    parser.add_argument("--baseline-two", type=Path, required=True)
    parser.add_argument("--baseline-three", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--appendix-tex", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--random-condition-summary", type=Path)
    parser.add_argument("--expected-rows", type=int, default=59)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    labeler_path = SRC_DIR / "cot_oversight_case_studies" / "reasoning_probe.py"
    parser_path = EXPERIMENTS_DIR / "verdict_parser.py"

    run_paths: dict[str, Path] = {
        "baseline_1": args.baseline_one,
        "baseline_2": args.baseline_two,
    }
    if args.baseline_three:
        run_paths["baseline_3"] = args.baseline_three
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    for name, path in run_paths.items():
        run, _ids = load_run(path)
        runs[name] = run
        row_counts[name] = len(run)

    comparisons = [
        compare_runs(
            left_name="baseline_1",
            left=runs["baseline_1"],
            right_name="baseline_2",
            right=runs["baseline_2"],
            tokenizer=tokenizer,
        )
    ]
    if "baseline_3" in runs:
        comparisons.extend(
            [
                compare_runs(
                    left_name="baseline_1",
                    left=runs["baseline_1"],
                    right_name="baseline_3",
                    right=runs["baseline_3"],
                    tokenizer=tokenizer,
                ),
                compare_runs(
                    left_name="baseline_2",
                    left=runs["baseline_2"],
                    right_name="baseline_3",
                    right=runs["baseline_3"],
                    tokenizer=tokenizer,
                ),
            ]
        )

    primary = comparisons[0]
    requires_third = primary["divergent_text_count"] > 0 and "baseline_3" not in runs
    random_range: dict[str, Any] | None = None
    if args.random_condition_summary:
        random_rows = [
            row
            for row in load_jsonl(args.random_condition_summary)
            if row.get("kind") == "random_norm_matched"
            and row.get("target_step") == "identifies_vulnerability"
        ]
        rates = [float(row["answer_flip_fraction_all_rows"]) for row in random_rows]
        random_range = {
            "condition_count": len(rates),
            "minimum": min(rates) if rates else None,
            "maximum": max(rates) if rates else None,
        }

    if primary["verdict_flip"]["count"] <= 1:
        noise_decision = "near_zero"
    elif primary["verdict_flip"]["count"] >= 3:
        noise_decision = "material"
    else:
        noise_decision = "intermediate_two_of_59"

    result = {
        "analysis": "repeat no-intervention baseline run-to-run noise floor",
        "model": args.model,
        "model_revision": args.model_revision,
        "input_hashes": {name: sha256_file(path) for name, path in run_paths.items()},
        "labeler": {
            "path": str(labeler_path),
            "sha256": sha256_file(labeler_path),
            "normalization": "normalize_reasoning_text",
            "rule": "step_presence_labels",
        },
        "verdict_parser": {
            "path": str(parser_path),
            "sha256": sha256_file(parser_path),
            "rule": "explicit_security_verdict",
        },
        "expected_rows": args.expected_rows,
        "row_counts": row_counts,
        "comparisons": comparisons,
        "random_identification_direction_flip_range": random_range,
        "requires_third_baseline": requires_third,
        "pre_stated_noise_floor_decision": noise_decision,
        "noise_floor_rule": (
            "near zero if <=1/59 verdict flips; material if >=3/59; "
            "two flips is reported as intermediate"
        ),
        "no_imputation": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    step_parts: list[str] = []
    for step in REASONING_STEPS:
        latex_step = step.replace("_", "\\_")
        agreement = primary["step_presence_agreement"][step]["count"]
        step_parts.append(f"{latex_step} {agreement}/{primary['common_rows']}")
    step_text = ", ".join(step_parts)
    random_text = ""
    if random_range and random_range["minimum"] is not None:
        random_text = (
            " The corresponding identification-family random-direction flip "
            f"range was {100 * random_range['minimum']:.1f}--"
            f"{100 * random_range['maximum']:.1f}\\%."
        )
    tex = (
        f"% labeler_sha256: {result['labeler']['sha256']}\n"
        f"% verdict_parser_sha256: {result['verdict_parser']['sha256']}\n"
        "Between the first and second no-intervention baselines, the run-to-run "
        f"verdict flip rate was {primary['verdict_flip']['count']}/"
        f"{primary['common_rows']}; step-presence agreement was {step_text}."
        f"{random_text}\n"
    )
    args.appendix_tex.parent.mkdir(parents=True, exist_ok=True)
    args.appendix_tex.write_text(tex, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 3 if requires_third else 0


if __name__ == "__main__":
    raise SystemExit(main())
