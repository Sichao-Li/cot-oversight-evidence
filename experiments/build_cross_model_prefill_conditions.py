"""Build aligned own-trace and other-model-trace inputs for the cross-model prefill control."""
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

from cot_oversight_case_studies.io import load_jsonl, sha256_file, write_jsonl  # noqa: E402


def security_rows(path: Path, *, expected_rows: int) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("status") == "ok"
        and row.get("domain") == "domain_b"
        and row.get("variant") == "standard"
    ]
    by_id = {str(row.get("id") or ""): row for row in rows}
    if "" in by_id or len(by_id) != len(rows):
        raise ValueError(f"{path} contains missing or duplicate security IDs.")
    if len(by_id) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} security rows in {path}, found {len(by_id)}."
        )
    return by_id


def build_condition_row(
    task_row: dict[str, Any],
    trace_row: dict[str, Any],
    *,
    condition: str,
    prefill_model: str,
) -> dict[str, Any]:
    row = dict(task_row)
    for field in (
        "raw_output",
        "thinking_tokens",
        "answer_text",
        "answer_letter",
        "answer_security_label",
        "is_correct",
        "answer_split_mode",
        "answer_extraction_mode",
        "explicit_answer_found",
        "generated_token_count",
        "hit_token_cap",
        "token_counts",
    ):
        row[field] = trace_row.get(field)
    row["condition"] = condition
    row["prefill_model"] = prefill_model
    row["trace_source_model"] = trace_row.get("model")
    row["trace_source_record_id"] = trace_row.get("record_id")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build aligned self-trace and foreign-trace teacher-forcing inputs "
            "for the same frozen security tasks."
        )
    )
    parser.add_argument("--self-inference", type=Path, required=True)
    parser.add_argument("--foreign-inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefill-model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    )
    parser.add_argument("--expected-rows", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    self_rows = security_rows(args.self_inference, expected_rows=args.expected_rows)
    foreign_rows = security_rows(
        args.foreign_inference,
        expected_rows=args.expected_rows,
    )
    if set(self_rows) != set(foreign_rows):
        raise ValueError("Self and foreign sources do not contain identical task IDs.")

    prompt_mismatches: list[str] = []
    reference_mismatches: list[str] = []
    self_condition: list[dict[str, Any]] = []
    foreign_condition: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    for case_id in sorted(self_rows):
        self_row = self_rows[case_id]
        foreign_row = foreign_rows[case_id]
        if str(self_row.get("prompt") or "") != str(foreign_row.get("prompt") or ""):
            prompt_mismatches.append(case_id)
        if str(self_row.get("correct_answer") or "") != str(
            foreign_row.get("correct_answer") or ""
        ):
            reference_mismatches.append(case_id)
        self_condition.append(
            build_condition_row(
                self_row,
                self_row,
                condition="self_trace",
                prefill_model=args.prefill_model,
            )
        )
        foreign_condition.append(
            build_condition_row(
                self_row,
                foreign_row,
                condition="foreign_trace",
                prefill_model=args.prefill_model,
            )
        )
        pairing_rows.append(
            {
                "id": case_id,
                "record_id": self_row.get("record_id"),
                "self_model": self_row.get("model"),
                "foreign_model": foreign_row.get("model"),
                "self_is_correct": self_row.get("is_correct"),
                "foreign_is_correct": foreign_row.get("is_correct"),
            }
        )

    if prompt_mismatches or reference_mismatches:
        raise ValueError(
            "Task alignment failed: "
            f"prompt_mismatches={len(prompt_mismatches)}, "
            f"reference_mismatches={len(reference_mismatches)}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_path = args.output_dir / "self_trace_prefill.jsonl"
    foreign_path = args.output_dir / "foreign_trace_prefill.jsonl"
    write_jsonl(self_path, self_condition)
    write_jsonl(foreign_path, foreign_condition)
    write_jsonl(args.output_dir / "pairing_manifest.jsonl", pairing_rows)
    protocol = {
        "self_inference": str(args.self_inference.resolve()),
        "self_inference_sha256": sha256_file(args.self_inference),
        "foreign_inference": str(args.foreign_inference.resolve()),
        "foreign_inference_sha256": sha256_file(args.foreign_inference),
        "prefill_model": args.prefill_model,
        "rows_per_condition": len(self_condition),
        "task_ids_identical": True,
        "prompts_identical": True,
        "reference_answers_identical": True,
        "self_output_sha256": sha256_file(self_path),
        "foreign_output_sha256": sha256_file(foreign_path),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
