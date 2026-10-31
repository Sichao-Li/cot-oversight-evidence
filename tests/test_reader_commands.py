import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_entry():
    spec = importlib.util.spec_from_file_location("reader_entry", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_and_forwarding(monkeypatch, capsys):
    entry = load_entry()
    entry.main(["--help"])
    assert "security-probes" in capsys.readouterr().out
    calls = []
    monkeypatch.setattr(entry.runpy, "run_module", lambda module, run_name: calls.append((module, list(sys.argv))))
    argv = sys.argv
    entry.main(["authority-patching", "--sites", "28:-1", "--output", "custom.json"])
    assert sys.argv is argv
    assert calls[0][0] == "cot_oversight_case_studies.localization"
    assert calls[0][1][-4:] == ["--sites", "28:-1", "--output", "custom.json"]
    entry.main(["security-monitor", "--provider", "openrouter", "--judges", "judge"])
    assert calls[1][1][1] == "judge"


def test_results_table_and_json(capsys):
    entry = load_entry()
    entry.main(["results"])
    output = capsys.readouterr().out
    assert "Exact / base" in output and "deepseek14b" in output
    assert "0.424 / 0.525" in output
    entry.main(["results", "--json"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 3
    assert [row["n"] for row in rows] == [59, 54, 58]


def test_security_evidence_command(capsys):
    entry = load_entry()
    entry.main(["security-evidence"])
    text = capsys.readouterr().out
    assert "15/59 changes vs 8.6/59 random" in text
    assert "suppression 13/27 vs 8.2/27" in text
    assert "0.424 / 0.525" in text
    assert "not a per-case" in text
    entry.main(["security-evidence", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert len(data["interventions"]) == 3
