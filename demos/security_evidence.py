"""Check saved aggregate probe/intervention evidence and recompute jury scores."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from cot_oversight_case_studies.monitorability import evaluate

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("identifies_vulnerability", "traces_data_flow", "assesses_exploitability")


def close(actual, expected):
    if not math.isclose(actual, expected, abs_tol=1e-12):
        raise ValueError(f"Inconsistent saved aggregate: {actual} != {expected}")


def summarize(saved, monitors, bootstrap=False):
    probes, ablation = saved["probe_controls"], saved["ablation"]
    representation = [r for r in probes["condition_metrics"] if r["condition"] == "self"]
    if len(representation) != 3 or {r["step_name"] for r in representation} != set(STEPS):
        raise ValueError("Expected all three reasoning-step probes")
    interventions = []
    for step in STEPS:
        probe_rows = [r for r in ablation["condition_summary"] if r["condition_id"] == f"probe__{step}"]
        controls = [r for r in ablation["condition_summary"] if r["target_step"] == step and r["kind"] == "random_norm_matched"]
        if len(probe_rows) != 1 or len(controls) != 5 or len({r["condition_id"] for r in controls}) != 5:
            raise ValueError("Each step requires one probe condition and five distinct random controls")
        probe = probe_rows[0]
        if any(r["rows"] != 59 or r["layer"] != probe["layer"] for r in [probe, *controls]):
            raise ValueError("Mismatched intervention/control count or layer")
        changes = [r["answer_flip_probe_minus_random"] for r in ablation["answer_changes"] if r["target_step"] == step]
        if len(changes) != 1:
            raise ValueError("Expected one verdict-change contrast per step")
        change = changes[0]
        suppression = ablation["suppression"][step][step]
        eligible = suppression["eligible_n"]
        if any(r["target_step_baseline_positive"] != eligible for r in [probe, *controls]):
            raise ValueError("Mismatched baseline-positive denominator")
        random_changes = mean(r["answer_flip_count"] for r in controls)
        random_suppression = mean(r["target_step_suppression_count"] for r in controls)
        close(probe["answer_flip_count"]/59, probe["answer_flip_fraction_all_rows"])
        close((probe["answer_flip_count"]-random_changes)/59, change["point_estimate"])
        close(probe["target_step_suppression_count"]/eligible, suppression["probe_suppression_rate"])
        close(random_suppression/eligible, suppression["random_mean_suppression_rate"])
        close((probe["target_step_suppression_count"]-random_suppression)/eligible,
              suppression["probe_minus_random"]["point_estimate"])
        selectivity = ablation["selectivity"][step]
        off_diagonal = [ablation["suppression"][step][s]["probe_minus_random"]["point_estimate"] for s in STEPS if s != step]
        close(suppression["probe_minus_random"]["point_estimate"]-mean(off_diagonal), selectivity["point_estimate"])
        interventions.append({"step": step, "layer": probe["layer"], "n": 59,
                             "verdict_changes": probe["answer_flip_count"],
                             "jointly_parseable": probe["baseline_and_intervention_parseable"],
                             "random_mean_verdict_changes": random_changes, "verdict_contrast": change,
                             "suppressed": probe["target_step_suppression_count"], "eligible_n": eligible,
                             "random_mean_suppressed": random_suppression,
                             "suppression_contrast": suppression["probe_minus_random"], "selectivity": selectivity})
    monitoring = {}
    for model, item in monitors.items():
        result = evaluate(item["jury"], item["labels"], bootstrap=bootstrap, seed_offset=item["seed_offset"])
        if result["n"] != item["expected"]["n"]:
            raise ValueError("Monitor denominator changed")
        for metric, values in result["metrics"].items():
            close(values["value"], item["expected"]["metrics"][metric]["value"])
            if bootstrap and values != item["expected"]["metrics"][metric]:
                raise ValueError("Monitor bootstrap does not reproduce saved results")
        monitoring[model] = result
    return {"source_sha256": saved["source_sha256"], "representation": representation,
            "probe_control_comparisons": probes["paired_bootstrap"], "interventions": interventions,
            "baseline_repeat": ablation["repeat_baseline"], "monitoring": monitoring}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the checked evidence as JSON.")
    parser.add_argument("--bootstrap", action="store_true", help="Recompute monitor bootstrap intervals; intervention intervals remain saved estimates.")
    args = parser.parse_args()
    saved = json.loads((ROOT / "examples/saved_security_evidence.json").read_text())
    monitors = json.loads((ROOT / "examples/saved_monitor.json").read_text())
    result = summarize(saved, monitors, args.bootstrap)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print("Representation | causal intervention | held-out monitoring\n")
    print("Saved probe AUCs and intervention intervals; count contrasts checked below.")
    print("Monitor metrics recomputed from saved predictions and response-specific labels.")
    print("No generation, probe fitting, GPU use, or API calls.\n")
    print("Representation: activation / text-only mean fold AUROC")
    for r in result["representation"]:
        print(f"  {r['step_name']}: {r['fixed_activation_fold_auc_mean']:.3f} / {r['bow_fold_auc_mean']:.3f}")
    print("\nInterventions: probe vs mean of five random directions; all three targets")
    for r in result["interventions"]:
        print(f"  {r['step']} (layer {r['layer']}): {r['verdict_changes']}/59 changes vs {r['random_mean_verdict_changes']:.1f}/59 random; jointly parseable {r['jointly_parseable']}/59")
        print(f"    suppression {r['suppressed']}/{r['eligible_n']} vs {r['random_mean_suppressed']:.1f}/{r['eligible_n']} random")
        for field in ("verdict_contrast", "suppression_contrast", "selectivity"):
            stat = r[field]
            lo, hi = stat["ci95"]
            print(f"    {field}: {100*stat['point_estimate']:+.1f} pp [{100*lo:.1f}, {100*hi:.1f}]; p={stat['two_sided_p']:.3f}")
    repeat = result["baseline_repeat"][0]
    print(f"\nBaseline repetition: {repeat['exact_text_identity_count']}/{repeat['common_rows']} texts identical; {repeat['verdict_flip']['count']} verdict changes; {repeat['verdict_parseability']['baseline_1']}/59 baseline verdicts parseable.")
    print("\nMonitoring: exact accuracy / model-specific majority baseline")
    for model, r in result["monitoring"].items():
        m = r["metrics"]["exact"]
        print(f"  {model}, n={r['n']}: {m['value']:.3f} / {m['baseline']:.3f}")
    print("\nIntervention tests are exploratory and uncorrected; missing verdicts are not evidence of unchanged answers.")
    print("Cross-step selectivity and complete-trace faithfulness are not established. These are distinct evaluations, not a per-case faithfulness/monitorability test.")


if __name__ == "__main__":
    main()
