import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from cot_oversight_case_studies.monitorability import aggregate, evaluate, parse_vote, render_prompt
from cot_oversight_case_studies.monitorability.jury import FIELDS, overall_label, validate_template
from cot_oversight_case_studies.monitorability.api import request_vote, validate_parameters
from cot_oversight_case_studies.monitorability import __main__ as cli


def vote(existence="MATCH", kind="MATCH", exploit="NOT_APPLICABLE"):
    return dict(zip(FIELDS, (existence, kind, exploit)))


def test_field_aggregation_and_missingness():
    assert overall_label(vote("MISMATCH")) == "INCORRECT"
    assert overall_label(vote(kind="MISMATCH")) == "PARTIALLY_CORRECT"
    assert overall_label(vote(kind="NOT_APPLICABLE")) == "CORRECT"
    result = aggregate([vote(kind="MATCH"), vote(kind="MISMATCH"), vote(kind="NOT_APPLICABLE")], expected_jurors=3)
    assert result["fields"][FIELDS[1]] == "MATCH"
    assert result["label"] == "CORRECT"
    assert aggregate([vote(), None, vote()], expected_jurors=3)["label"] is None
    with pytest.raises(ValueError):
        parse_vote('{"label":"CORRECT"}')
    assert parse_vote("```json\n" + json.dumps(vote()) + "\n```") == vote()


def test_prompt_no_second_interpolation_or_annotation_leak():
    template = '{code_snippet}\n{thinking_tokens}\n{answer_text}\n{expected_behavior}\n{"output":0}'
    row = {"prompt": "code {expected_behavior}", "thinking_tokens": "trace", "answer_text": "",
           "reference_answer": "reference", "label": "DO_NOT_EXPOSE"}
    result = render_prompt(template, row)
    assert "code {expected_behavior}" in result
    assert '{"output":0}' in result
    assert "DO_NOT_EXPOSE" not in result


def test_metrics_and_errors():
    labels = {"a": "INCORRECT", "b": "CORRECT", "c": "CORRECT", "d": "UNCERTAIN"}
    preds = {"a": "PARTIALLY_CORRECT", "b": "CORRECT", "c": "INCORRECT"}
    result = evaluate(preds, labels)
    assert result["n"] == 3
    assert result["excluded_uncertain_ids"] == ["d"]
    assert result["metrics"]["exact"]["value"] == pytest.approx(1/3)
    assert result["metrics"]["partial"]["value"] == .5
    assert result["metrics"]["binary"]["value"] == pytest.approx(2/3)
    assert result["metrics"]["partial"]["baseline"] == pytest.approx(2/3)
    with pytest.raises(ValueError, match="Missing"):
        evaluate({}, labels)
    with pytest.raises(ValueError, match="Unknown"):
        evaluate(preds, {"a": "TYPO"})


def test_saved_real_decisions():
    saved = json.loads((Path(__file__).resolve().parents[1] / "examples/saved_monitor.json").read_text())
    for item in saved.values():
        result = evaluate(item["jury"], item["labels"])
        assert result["n"] == item["expected"]["n"]
        assert result["excluded_uncertain_ids"] == item["expected"]["excluded_uncertain_ids"]
        for name, metric in result["metrics"].items():
            assert metric["value"] == pytest.approx(item["expected"]["metrics"][name]["value"], abs=1e-12)
            if "baseline" in metric:
                assert metric["baseline"] == pytest.approx(item["expected"]["metrics"][name]["baseline"], abs=1e-12)


def test_real_example_labels_match_responses():
    root = Path(__file__).resolve().parents[1] / "examples"
    cases = {r["id"]: r for r in cli.load_rows(root / "security_cases.jsonl")}
    for label in cli.load_rows(root / "security_labels.jsonl"):
        assert label["response_sha256"] == cli.response_hash(cases[label["id"]])


def test_transport_leaves_defaults_unset():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="response", model="returned", usage=None,
                               choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(vote())))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = request_vote(client, "judge", "prompt", {})
    assert "temperature" not in calls[0]
    assert result["finish_reason"] == "stop"
    with pytest.raises(ValueError):
        validate_parameters({"api_key": "secret"})


@pytest.mark.parametrize("parameters", [
    {"temperature": "0"}, {"temperature": float("nan")}, {"temperature": -1},
    {"top_p": 1.5}, {"max_tokens": 0}, {"seed": True}, {"reasoning_effort": " "},
    {"max_tokens": 100, "max_completion_tokens": 100},
])
def test_invalid_decoding_settings(parameters):
    with pytest.raises(ValueError):
        validate_parameters(parameters)


def test_explicit_decoding_settings():
    settings = {"temperature": 0, "top_p": 1.0, "max_completion_tokens": 1024,
                "seed": 42, "reasoning_effort": "high"}
    assert validate_parameters(settings) == settings
    assert validate_parameters({}) == {}


def test_resume_and_changed_protocol(tmp_path, monkeypatch):
    data = tmp_path / "input.jsonl"
    data.write_text(json.dumps({"id": "one", "prompt": "code", "thinking_tokens": "trace", "answer_text": "answer", "reference_answer": "reference"}) + "\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{code_snippet} {thinking_tokens} {answer_text} {expected_behavior}")
    calls = []
    monkeypatch.setattr(cli, "create_client", lambda _: object())
    def request(*args):
        calls.append(args)
        return {"text": json.dumps(vote()), "finish_reason": "stop"}
    monkeypatch.setattr(cli, "request_vote", request)
    args = SimpleNamespace(input=data, prompt=prompt, parameters=None, fewshot=None,
                           provider="openrouter", judges="a,b,c", output=tmp_path / "out", retry_errors=False)
    assert cli.run(args) == 0
    assert len(calls) == 3
    assert cli.run(args) == 0
    assert len(calls) == 3
    annotations = tmp_path / "labels.jsonl"
    annotations.write_text(json.dumps({"id": "one", "label": "CORRECT", "response_sha256": "wrong"}) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        cli.score(SimpleNamespace(predictions=args.output / "predictions.json", labels=annotations,
                                  output=tmp_path / "metrics.json", bootstrap=False, seed_offset=0))
    args.judges = "different"
    with pytest.raises(ValueError, match="different inputs"):
        cli.run(args)


def test_truncated_votes_stay_incomplete(tmp_path, monkeypatch):
    data = tmp_path / "input.jsonl"
    data.write_text(json.dumps({"id": "one", "prompt": "code", "thinking_tokens": "trace", "answer_text": "", "reference_answer": "reference"}) + "\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{code_snippet} {thinking_tokens} {answer_text} {expected_behavior}")
    monkeypatch.setattr(cli, "create_client", lambda _: object())
    monkeypatch.setattr(cli, "request_vote", lambda *args: {"text": json.dumps(vote()), "finish_reason": "length"})
    args = SimpleNamespace(input=data, prompt=prompt, parameters=None, fewshot=None, provider="openrouter",
                           judges="a", output=tmp_path / "out", retry_errors=False)
    assert cli.run(args) == 1
    result = json.loads((args.output / "predictions.json").read_text())
    assert result[0]["label"] is None


def test_required_prompt_evidence_and_calibration():
    complete = "{code_snippet} {thinking_tokens} {answer_text} {expected_behavior}"
    validate_template(complete)
    with pytest.raises(ValueError, match="required evidence"):
        validate_template("{code_snippet}")
    with pytest.raises(ValueError, match="Calibration"):
        validate_template(complete, "development example")
    validate_template(complete + " {fewshot_examples}", "development example")
