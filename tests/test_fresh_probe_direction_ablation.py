from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments import analyze_fresh_probe_direction_ablation as analysis


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_fresh_analysis_compares_probe_with_paired_random_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    labels_present = {
        "identifies_vulnerability": 1,
        "traces_data_flow": 0,
        "assesses_exploitability": 0,
    }
    labels_absent = {**labels_present, "identifies_vulnerability": 0}
    baseline = tmp_path / "baseline.jsonl"
    plan = tmp_path / "plan.jsonl"
    interventions = tmp_path / "interventions.jsonl"
    output_dir = tmp_path / "analysis"
    write_jsonl(
        baseline,
        [
            {
                "id": "case-1",
                "status": "ok",
                "raw_output": "A vulnerability exists.",
                "generated_step_presence_labels": labels_present,
                "generated_token_count": 100,
            },
            {
                "id": "case-2",
                "status": "ok",
                "raw_output": "A vulnerability exists.",
                "generated_step_presence_labels": labels_absent,
                "generated_token_count": 100,
            },
        ],
    )
    write_jsonl(
        plan,
        [
            {
                "condition_id": "probe__identifies_vulnerability",
                "kind": "probe_direction",
                "target_step": "identifies_vulnerability",
                "layer": 28,
            },
            {
                "condition_id": "random__identifies_vulnerability__00",
                "kind": "random_norm_matched",
                "target_step": "identifies_vulnerability",
                "layer": 28,
            },
        ],
    )
    write_jsonl(
        interventions,
        [
            {
                "id": "case-1",
                "status": "ok",
                "condition_id": "probe__identifies_vulnerability",
                "raw_output": "No vulnerability exists.",
                "generated_step_presence_labels": labels_absent,
                "generated_token_count": 90,
            },
            {
                "id": "case-2",
                "status": "ok",
                "condition_id": "probe__identifies_vulnerability",
                "raw_output": "A vulnerability exists.",
                "generated_step_presence_labels": labels_absent,
                "generated_token_count": 100,
            },
            {
                "id": "case-1",
                "status": "ok",
                "condition_id": "random__identifies_vulnerability__00",
                "raw_output": "A vulnerability exists.",
                "generated_step_presence_labels": labels_present,
                "generated_token_count": 101,
            },
            {
                "id": "case-2",
                "status": "ok",
                "condition_id": "random__identifies_vulnerability__00",
                "raw_output": "A vulnerability exists.",
                "generated_step_presence_labels": labels_absent,
                "generated_token_count": 99,
            },
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_fresh_probe_direction_ablation.py",
            "--baseline",
            str(baseline),
            "--interventions",
            str(interventions),
            "--intervention-plan",
            str(plan),
            "--output-dir",
            str(output_dir),
            "--expected-rows",
            "2",
            "--bootstrap-draws",
            "100",
        ],
    )

    assert analysis.main() == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = summary["comparisons"][0]
    assert comparison["answer_flip_probe_minus_random"]["point_estimate"] == 0.5
    assert (
        comparison["target_suppression_probe_minus_random"]["point_estimate"]
        == 1.0
    )
