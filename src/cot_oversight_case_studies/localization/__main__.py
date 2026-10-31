"""Evaluate a fixed site or site set without selecting on held-out outcomes."""
import argparse
from pathlib import Path
from . import evaluate_patch
from ..hf import load_model, runtime
from ..io import load_rows, fingerprint, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sites", required=True, help="Comma-separated layer:position pairs patched jointly, e.g. 24:-11")
    parser.add_argument("--split", choices=["demonstration", "discovery", "heldout"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    sites = [tuple(map(int, value.split(":"))) for value in args.sites.split(",")]
    if any(len(site) != 2 for site in sites):
        raise ValueError("Sites must be layer:position pairs.")
    rows = load_rows(args.input)
    if any(row.get("split") != args.split for row in rows):
        raise ValueError("All rows must have the requested split.")
    if args.output.exists():
        raise FileExistsError(args.output)
    model, tokenizer = load_model(args.model, args.revision, local_files_only=args.local_files_only)
    results = []
    for row in rows:
        result = evaluate_patch(model, tokenizer, row["clean_prompt"], row["hinted_prompt"], row["correct_answer"], row["hint_answer"], sites)
        results.append({"id": row["id"], "group": row.get("group"), **result})
    write_new(args.output, {"input_sha256": fingerprint(rows), "runtime": runtime(model), "requested_revision": args.revision,
                           "split": args.split, "results": results})


if __name__ == "__main__":
    main()
