"""Reject generation resumes with incompatible inputs, code or settings."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def guard_generation_run(args, source_file):
    if not re.fullmatch(r"[a-fA-F0-9]{40}", args.model_revision or ""):
        raise ValueError("Generation requires --model-revision with an immutable 40-character commit hash.")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive.")
    if hasattr(args, "batch_size") and args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    for index, count in (("shard_index", "num_shards"), ("condition_shard_index", "condition_shard_count")):
        if hasattr(args, index) and not 0 <= getattr(args, index) < getattr(args, count):
            raise ValueError("Invalid shard index/count.")
    source_file = Path(source_file)
    manifest = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    record_ids = [row.get("record_id") for row in manifest]
    if not manifest or any(not isinstance(key, str) or not key for key in record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("Generation manifest requires nonempty unique record_id values.")
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"] for row in manifest):
        raise ValueError("Generation manifest requires a nonempty prompt for every record.")
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in {"output", "local_files_only"}}
    inputs = {name: file_digest(getattr(args, name)) for name in ("manifest", "intervention_plan", "directions") if hasattr(args, name)}
    sources = {source_file.name: file_digest(source_file), "generation_protocol.py": file_digest(__file__)}
    if source_file.name == "run_probe_direction_ablation.py":
        sources["run_hf_inference.py"] = file_digest(source_file.with_name("run_hf_inference.py"))
        rules = source_file.parents[1] / "src/cot_oversight_case_studies/reasoning_probe.py"
        sources["reasoning_probe.py"] = file_digest(rules)
    versions = {}
    for package in ("torch", "transformers", "tokenizers", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {"schema": 1, "settings": settings, "input_sha256": inputs, "code_sha256": sources, "versions": versions}
    output = Path(args.output)
    sidecar = output.with_suffix(output.suffix + ".protocol.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != protocol:
            raise ValueError("Generation protocol mismatch: use a new output path for different inputs, code, runtime or settings.")
    else:
        metadata = output.with_suffix(output.suffix + ".metadata.jsonl")
        if output.exists() or metadata.exists():
            raise ValueError("Existing generation output has no verified protocol; use a new output path.")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, allow_nan=False)
            handle.write("\n")
