"""Validate probes and patching on real cases using one shared model load."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from run_reasoning_span_probe import extract_reasoning_span_activations
from cot_oversight_case_studies.hf import load_model, runtime, blocks
from cot_oversight_case_studies.probes import extract_span, normalize_reasoning_text
from cot_oversight_case_studies.probes import step_presence_labels
from cot_oversight_case_studies.localization import evaluate_patch
from cot_oversight_case_studies.io import load_rows, write_new, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", default="examples/security_cases.jsonl")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", help="Write a validation report, not generated completions.")
    destination.add_argument("--no-save", action="store_true", help="Keep all computed values in memory; print only pass/fail.")
    parser.add_argument("--layers", default="8,12,16,20,24,28")
    parser.add_argument("--pairs", default="examples/authority_pairs.jsonl")
    parser.add_argument("--sites", default="24:-11")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        raise FileExistsError(args.output)
    rows = load_rows(args.input)
    layers = [int(value) for value in args.layers.split(",")]
    model, tokenizer = load_model(args.model, args.revision)
    checks = []
    step_labels = {}
    for row in rows:
        text = normalize_reasoning_text(row)
        step_labels[row["id"]] = step_presence_labels(text)
        current, n = extract_span(model, tokenizer, row["prompt"], text, layers)
        full_protocol, original_n = extract_reasoning_span_activations(model, tokenizer, prompt=row["prompt"],
            continuation=text, layers=layers, max_continuation_tokens=256, pooling="mean")
        for layer in layers:
            checks.append({"id": row["id"], "layer": layer, "tokens_equal": n == original_n,
                           "array_equal": bool(np.array_equal(current[layer], full_protocol[layer])),
                           "max_absolute_difference": float(np.max(np.abs(current[layer] - full_protocol[layer])))})
    passed = all(check["tokens_equal"] and check["array_equal"] for check in checks)
    sites = [tuple(map(int, value.split(":"))) for value in args.sites.split(",")]
    pairs = load_rows(args.pairs)
    patch_results = [{"id": row["id"], **evaluate_patch(model, tokenizer, row["clean_prompt"], row["hinted_prompt"],
                     row["correct_answer"], row["hint_answer"], sites)} for row in pairs]
    patch_checks = []
    for row in pairs:
        identity = evaluate_patch(model, tokenizer, row["clean_prompt"], row["clean_prompt"], row["correct_answer"], row["hint_answer"], sites)
        terminal = evaluate_patch(model, tokenizer, row["clean_prompt"], row["hinted_prompt"], row["correct_answer"], row["hint_answer"], [(len(blocks(model))-1, -1)])
        check = {"id": row["id"], "identity_margin_difference": identity["patched_margin"] - identity["clean_margin"],
                 "terminal_margin_difference": terminal["patched_margin"] - terminal["clean_margin"]}
        check["passed"] = abs(check["identity_margin_difference"]) <= 1e-6 and abs(check["terminal_margin_difference"]) <= 1e-6
        patch_checks.append(check)
    passed = passed and all(check["passed"] for check in patch_checks)
    report = {"passed": passed, "input_sha256": fingerprint(rows), "pairs_sha256": fingerprint(pairs),
                           "runtime": runtime(model), "checks": checks, "step_labels": step_labels, "patching": patch_results, "patch_checks": patch_checks,
                           "scope": "Functional validation on actual examples; not a research performance estimate."}
    if args.output:
        write_new(args.output, report)
    if not passed:
        detail = "Inspect the saved comparison." if args.output else "No report was saved; use --output to inspect differences."
        raise SystemExit(f"Extraction equivalence or patch sanity checks failed. {detail}")
    print(f"PASS: {len(checks)} real-case/layer activation comparisons match exactly; {len(patch_results)} actual task pair(s) patched.")


if __name__ == "__main__":
    main()
