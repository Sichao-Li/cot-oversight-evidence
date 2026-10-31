"""Recompute real, response-specific jury results without model/API calls."""
import argparse
import json
from pathlib import Path
from cot_oversight_case_studies.monitorability import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="Also reproduce 10,000-resample confidence intervals and p-values.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable records instead of a table.")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "examples/saved_monitor.json"
    saved = json.loads(path.read_text())
    if not args.json:
        print("Security analysis quality: saved LLM-jury predictions vs response-specific human labels")
        print("Bundled annotation version; no new model or API calls.\n")
        print(f"{'Model':<15} {'n':>3} {'Excluded':>8} {'Exact / base':>16} {'Partial / base':>16} {'Binary / base':>16}")
    for model, item in saved.items():
        result = evaluate(item["jury"], item["labels"], bootstrap=args.bootstrap, seed_offset=item["seed_offset"])
        expected = item["expected"]
        assert result["n"] == expected["n"]
        for metric, values in result["metrics"].items():
            assert abs(values["value"] - expected["metrics"][metric]["value"]) < 1e-12
            if args.bootstrap:
                assert values == expected["metrics"][metric]
        if args.json:
            print(json.dumps({"model": model, "n": result["n"], "excluded_uncertain": len(result["excluded_uncertain_ids"]),
                              "metrics": result["metrics"]}))
        else:
            metrics = result["metrics"]
            cells = [f"{metrics[name]['value']:.3f} / {metrics[name]['baseline']:.3f}" for name in ("exact", "partial", "binary")]
            print(f"{model:<15} {result['n']:>3} {len(result['excluded_uncertain_ids']):>8} " + " ".join(f"{cell:>16}" for cell in cells))
            if args.bootstrap:
                for name, values in metrics.items():
                    low, high = values["ci_95"]
                    p_text = f"; paired p={values['p_one_sided_gt_baseline']:.4f}" if "p_one_sided_gt_baseline" in values else ""
                    print(f"  {name}: 95% CI [{low:.3f}, {high:.3f}]{p_text}")
    if not args.json:
        print("\nExact: same three-way label. Partial: ordinal credit. Binary: CORRECT vs not-CORRECT.")
        print("Base: constant-label comparator under the same metric. Excluded: UNCERTAIN human labels.")
        print("Matches the bundled expected results. Metric definitions: METHODS.md.")


if __name__ == "__main__":
    main()
