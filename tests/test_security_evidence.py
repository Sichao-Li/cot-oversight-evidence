import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("security_evidence", ROOT / "demos/security_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads((ROOT / "examples/saved_security_evidence.json").read_text()), json.loads((ROOT / "examples/saved_monitor.json").read_text())


def test_all_three_evidence_types():
    module, data, monitors = load()
    report = module.summarize(data, monitors)
    assert len(report["representation"]) == len(report["interventions"]) == 3
    flow = next(r for r in report["representation"] if r["step_name"] == "traces_data_flow")
    assert f"{flow['fixed_activation_fold_auc_mean']:.3f}" == "0.838"
    ident = report["interventions"][0]
    assert (ident["layer"], ident["verdict_changes"], ident["jointly_parseable"]) == (28, 15, 36)
    assert (ident["suppressed"], ident["eligible_n"]) == (13, 27)
    assert ident["random_mean_verdict_changes"] == 8.6
    assert ident["random_mean_suppressed"] == 8.2
    assert all(r["selectivity"]["ci95"][0] < 0 < r["selectivity"]["ci95"][1] for r in report["interventions"])
    assert [r["n"] for r in report["monitoring"].values()] == [59, 54, 58]
    repeat = report["baseline_repeat"][0]
    assert repeat["exact_text_identity_count"] == 59
    assert repeat["verdict_parseability"]["baseline_1"] == 46


def test_reject_inconsistent_counts():
    module, data, monitors = load()
    modified = copy.deepcopy(data)
    row = next(r for r in modified["ablation"]["condition_summary"] if r["condition_id"] == "probe__identifies_vulnerability")
    row["answer_flip_count"] += 1
    with pytest.raises(ValueError, match="Inconsistent"):
        module.summarize(modified, monitors)


def test_reject_missing_random_control():
    module, data, monitors = load()
    data["ablation"]["condition_summary"] = [r for r in data["ablation"]["condition_summary"] if r["condition_id"] != "random__identifies_vulnerability__00"]
    with pytest.raises(ValueError, match="five distinct"):
        module.summarize(data, monitors)
