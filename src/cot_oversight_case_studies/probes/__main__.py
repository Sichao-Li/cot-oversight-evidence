"""Extract real-model activations and evaluate reasoning-step probes."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from . import extract_span, normalize_reasoning_text, step_presence_labels, cross_validated_probe_predictions
from .. import reasoning_probe
from ..hf import load_model, runtime
from ..io import fingerprint, load_rows, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layers", default="8,12,16,20,24,28")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    layers = reasoning_probe.parse_layer_list(args.layers)
    rows = load_rows(args.input)
    texts = [normalize_reasoning_text(r) for r in rows]
    if any(not t for t in texts):
        raise ValueError("Every probe row needs a nonempty saved reasoning continuation.")
    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(args.model, args.revision, local_files_only=args.local_files_only)
    collected = {layer: [] for layer in layers}
    token_counts = []
    labels = [step_presence_labels(text) for text in texts]
    for row, text in zip(rows, texts):
        features, n = extract_span(model, tokenizer, row["prompt"], text, layers)
        for layer in layers:
            collected[layer].append(features[layer])
        token_counts.append(n)
    results = {}
    for layer, values in collected.items():
        x = np.stack(values)
        np.savez_compressed(args.output / f"layer_{layer}.npz", activations=x, ids=[r["id"] for r in rows])
        results[str(layer)] = {}
        for step in labels[0]:
            y = [r[step] for r in labels]
            if min(sum(y), len(y) - sum(y)) < 2:
                result = {"auc_mean": None, "reason": "Insufficient examples in each class for cross-validation."}
            else:
                result = cross_validated_probe_predictions(x, y)
                result["probabilities"] = result["probabilities"].tolist()
            result.update(positives=sum(y), negatives=len(y)-sum(y))
            results[str(layer)][step] = result
    write_new(args.output / "summary.json", {"input_sha256": fingerprint(rows), "runtime": runtime(model),
              "requested_revision": args.revision, "labeler_sha256": hashlib.sha256(Path(reasoning_probe.__file__).read_bytes()).hexdigest(),
              "ids": [r["id"] for r in rows], "labels": labels, "continuation_tokens": token_counts,
              "layers": results, "note": "Step presence, not correctness or causal faithfulness; no row-level confirmation in this smoke workflow."})


if __name__ == "__main__":
    main()
