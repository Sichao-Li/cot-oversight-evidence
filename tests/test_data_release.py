"""The released data must agree with the saved labels and the paper's counts."""
import json
from pathlib import Path

from cot_oversight_case_studies.io import fingerprint, load_jsonl
from cot_oversight_case_studies.reasoning_probe import normalize_reasoning_text, step_presence_labels

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/security"
MODEL_KEY = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek14b", "Qwen/QwQ-32B": "qwq32b",
             "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek70b"}


def test_row_counts():
    assert len(load_jsonl(DATA / "generations_deepseek14b.jsonl")) == 200
    assert len(load_jsonl(DATA / "generations_deepseek70b_prefilled_into_14b.jsonl")) == 200
    for name in MODEL_KEY.values():
        assert len(load_jsonl(DATA / f"heldout59/{name}.jsonl")) == 59
    assert len(load_jsonl(DATA / "ablation/interventions.jsonl")) == 18 * 59


def test_labels_match_saved_monitor_and_responses():
    saved = json.load((ROOT / "examples/saved_monitor.json").open())
    labels = load_jsonl(DATA / "heldout59/human_labels.jsonl")
    assert len(labels) == 177
    for key in MODEL_KEY.values():
        rows = {r["id"]: r for r in load_jsonl(DATA / f"heldout59/{key}.jsonl")}
        for label in (l for l in labels if MODEL_KEY[l["model"]] == key):
            assert saved[key]["labels"][label["id"]] == label["label"]
            row = rows[label["id"]]
            assert fingerprint({"thinking_tokens": row["thinking_tokens"] or "",
                                "answer_text": row["answer_text"] or ""}) == label["response_sha256"]


def test_heldout_is_subset_of_generations_with_identical_text():
    full = {r["id"]: r for r in load_jsonl(DATA / "generations_deepseek14b.jsonl")}
    for row in load_jsonl(DATA / "heldout59/deepseek14b.jsonl"):
        assert full[row["id"]]["thinking_tokens"] == row["thinking_tokens"]


def test_step_labels_give_paper_counts():
    rows = load_jsonl(DATA / "generations_deepseek14b.jsonl")
    assert sum("</think>" in (r["thinking_tokens"] or "") for r in rows) == 170
    positives = {"identifies_vulnerability": 0, "traces_data_flow": 0, "assesses_exploitability": 0}
    for row in load_jsonl(DATA / "ablation/baseline.jsonl"):
        for step, value in step_presence_labels(normalize_reasoning_text(row)).items():
            positives[step] += value
    assert positives == {"identifies_vulnerability": 27, "traces_data_flow": 25, "assesses_exploitability": 29}


def test_repeated_baseline_is_identical():
    one = {r["id"]: r["raw_output"] for r in load_jsonl(DATA / "ablation/baseline.jsonl")}
    two = {r["id"]: r["raw_output"] for r in load_jsonl(DATA / "ablation/baseline_repeat.jsonl")}
    assert one == two
