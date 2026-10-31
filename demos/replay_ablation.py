"""Recompute the direction-ablation results from the released generations and print a summary."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/security/ablation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--quiet", action="store_true", help="Suppress the analysis script's full JSON output.")
    args = parser.parse_args()
    script = ROOT / "experiments/analyze_fresh_probe_direction_ablation.py"
    sys.argv = [str(script), "--baseline", str(DATA / "baseline.jsonl"), "--interventions", str(DATA / "interventions.jsonl"),
                "--intervention-plan", str(DATA / "intervention_plan.jsonl"), "--output-dir", args.output_dir]
    sink = io.StringIO() if args.quiet else sys.stdout
    try:
        with contextlib.redirect_stdout(sink):
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    summary = json.load((Path(args.output_dir) / "summary.json").open())
    conditions = {row["condition_id"]: row for row in map(json.loads, (Path(args.output_dir) / "condition_summary.jsonl").read_text().splitlines()) if row}
    print(f"\nDirection ablation on {summary['baseline_rows']} held-out cases; "
          f"{summary['baseline_parseable_verdicts']} baseline verdicts parseable.")
    for comparison in summary["comparisons"]:
        step = comparison["target_step"]
        probe = conditions[comparison["probe_condition_id"]]
        randoms = [conditions[c] for c in comparison["random_condition_ids"]]
        flips = comparison["answer_flip_probe_minus_random"]
        supp = comparison["target_suppression_probe_minus_random"]
        random_flips = sum(r["answer_flip_count"] for r in randoms) / len(randoms)
        random_supp = sum(r["target_step_suppression_count"] for r in randoms) / len(randoms)
        print(f"  {step} (layer {probe['layer']})")
        print(f"    verdict changes: {probe['answer_flip_count']}/{probe['rows']} vs random mean {random_flips:.1f}; "
              f"jointly parseable {probe['baseline_and_intervention_parseable']}")
        print(f"    step removed:    {probe['target_step_suppression_count']}/{comparison['target_suppression_eligible_rows']} "
              f"vs random mean {random_supp:.1f}")
        print(f"    probe minus random: verdicts {100*flips['point_estimate']:+.1f} pp "
              f"[{100*flips['ci95'][0]:.1f}, {100*flips['ci95'][1]:.1f}], p={flips['two_sided_p']:.3f}; "
              f"suppression {100*supp['point_estimate']:+.1f} pp, p={supp['two_sided_p']:.3f}")
    print("Two-sided paired bootstrap p-values, uncorrected. Suppression intervals in the paper come from "
          "experiments/analyze_ablation_specificity.py (same inputs, different bootstrap seed).")
    print(f"Full output: {args.output_dir}/")


if __name__ == "__main__":
    main()
