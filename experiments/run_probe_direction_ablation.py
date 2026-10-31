"""Generate responses while mean-ablating a fitted direction (or a random control)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for _path in (EXPERIMENTS_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cot_oversight_case_studies.io import append_jsonl, load_jsonl  # noqa: E402
from run_hf_inference import (  # noqa: E402
    build_record,
    render_prompt,
    resolve_stop_ids,
    trim_generated_token_ids,
)
from cot_oversight_case_studies.reasoning_probe import (  # noqa: E402
    normalize_reasoning_text,
    step_presence_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate under frozen probe-direction mean ablations."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intervention-plan", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--condition-shard-index", type=int, default=0)
    parser.add_argument("--condition-shard-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-gib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-rendering",
        choices=("auto_chat", "raw"),
        default="auto_chat",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "eager", "sdpa"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from generation_protocol import guard_generation_run
    guard_generation_run(args, __file__)
    if not 0 <= args.condition_shard_index < args.condition_shard_count:
        raise ValueError("Invalid condition shard.")
    jobs = load_jsonl(args.manifest)
    all_conditions = sorted(
        load_jsonl(args.intervention_plan),
        key=lambda row: (
            0 if row.get("kind") == "probe_direction" else 1,
            str(row.get("target_step") or ""),
            str(row.get("condition_id") or ""),
        ),
    )
    conditions = [
        row
        for index, row in enumerate(all_conditions)
        if index % args.condition_shard_count == args.condition_shard_index
    ]
    if not jobs or not conditions:
        raise ValueError("The selected jobs and intervention conditions must be non-empty.")
    with np.load(args.directions, allow_pickle=False) as payload:
        direction_arrays = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
        }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError("No CUDA device is visible.")
    max_memory = {
        device_index: f"{args.gpu_memory_gib}GiB"
        for device_index in range(visible_gpus)
    }
    model_kwargs = {
        "revision": args.model_revision,
        "torch_dtype": torch.bfloat16,
        "device_map": "balanced",
        "max_memory": max_memory,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.attention_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attention_implementation
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    stop_ids = resolve_stop_ids(tokenizer)
    existing = {
        str(row["instance_key"])
        for row in load_jsonl(args.output)
        if row.get("status") == "ok" and row.get("instance_key")
    } if args.output.exists() else set()

    metadata = {
        "record_type": "run_metadata",
        "model": args.model,
        "model_revision": args.model_revision,
        "manifest": str(args.manifest),
        "intervention_plan": str(args.intervention_plan),
        "directions": str(args.directions),
        "condition_shard_index": args.condition_shard_index,
        "condition_shard_count": args.condition_shard_count,
        "selected_conditions": [row["condition_id"] for row in conditions],
        "condition_order": "probe_directions_then_controls",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "do_sample": False,
        "prompt_rendering": args.prompt_rendering,
        "attention_implementation": args.attention_implementation,
        "ablation_span": "prompt_prefill_and_every_generated_step",
        "started_unix": time.time(),
    }
    append_jsonl(args.output.with_suffix(args.output.suffix + ".metadata.jsonl"), metadata)
    print(json.dumps(metadata, indent=2), flush=True)

    for condition in conditions:
        condition_id = str(condition["condition_id"])
        layer_index = int(condition["layer"])
        direction_cpu = direction_arrays[str(condition["vector_key"])]
        projection_mean = float(condition["projection_mean"])

        def ablation_hook(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            direction = torch.as_tensor(
                direction_cpu,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            mean = torch.as_tensor(
                projection_mean,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            projection = torch.einsum("...h,h->...", hidden, direction)
            transformed = hidden - (projection - mean).unsqueeze(-1) * direction
            if isinstance(output, tuple):
                return (transformed, *output[1:])
            return transformed

        handle = model.model.layers[layer_index].register_forward_hook(ablation_hook)
        try:
            for job_index, job in enumerate(jobs, start=1):
                instance_key = f"{condition_id}::{job['record_id']}"
                if instance_key in existing:
                    continue
                rendered_prompt = (
                    str(job["prompt"])
                    if args.prompt_rendering == "raw"
                    else render_prompt(tokenizer, str(job["prompt"]))
                )
                encoded = tokenizer(rendered_prompt, return_tensors="pt")
                encoded = {
                    key: value.to(input_device)
                    for key, value in encoded.items()
                }
                with torch.inference_mode():
                    output_ids = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=stop_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                input_width = encoded["input_ids"].shape[1]
                generated_ids = trim_generated_token_ids(
                    output_ids[0, input_width:],
                    stop_ids,
                )
                raw_output = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                ).strip()
                record = build_record(
                    job,
                    raw_output,
                    tokenizer,
                    model_id=args.model,
                    max_new_tokens=args.max_new_tokens,
                    shard_index=args.condition_shard_index,
                    num_shards=args.condition_shard_count,
                )
                reasoning_text = normalize_reasoning_text(record)
                record.update(
                    {
                        "instance_key": instance_key,
                        "condition_id": condition_id,
                        "intervention_kind": condition["kind"],
                        "intervention_target_step": condition["target_step"],
                        "intervention_layer": layer_index,
                        "intervention_projection_mean": projection_mean,
                        "generated_step_presence_labels": step_presence_labels(
                            reasoning_text
                        ),
                    }
                )
                append_jsonl(args.output, record)
                existing.add(instance_key)
                print(
                    f"condition={condition_id} jobs={job_index}/{len(jobs)} "
                    f"record_id={job['record_id']}",
                    flush=True,
                )
        finally:
            handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
