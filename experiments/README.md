# Experiment scripts

These scripts run the full-scale security analyses from the paper. Their inputs are in [`data/`](../data/README.md); the probe and generation scripts also need the pinned model on a GPU. `python run.py ablation` runs the ablation analysis from the released generations.

| Question | Scripts |
|---|---|
| Can reasoning steps be read from activations? | `run_reasoning_span_probe.py` |
| Do text features alone do as well? | `run_trace_only_probe_baselines.py` |
| Is the probe advantage specific to the model's own traces? | `build_cross_model_prefill_conditions.py`, `analyze_cross_model_prefill.py` |
| Does ablating a probe direction change the output? | `prepare_probe_direction_ablation.py`, `run_probe_direction_ablation.py`, `analyze_fresh_probe_direction_ablation.py` |
| Is the effect specific to the target step, and larger than run-to-run noise? | `analyze_ablation_specificity.py`, `analyze_baseline_repeat.py` |
| Generate task responses | `run_hf_inference.py` |

`verdict_parser.py` and `generation_protocol.py` are shared helpers. Each script's `--help` lists its inputs and options.

## Setup

Install a PyTorch build for your hardware, then:

```bash
pip install -e .
pip install -r experiments/requirements-gpu.txt
```

## Example: probes

```bash
python experiments/run_reasoning_span_probe.py \
  --inference data/security/generations_deepseek14b.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --model-revision 1df8507178afcc1bef68cd8c393f61a886323761 \
  --tokenizer-json /path/to/tokenizer.json \
  --primary-layers 8,12,16,20,24,28 \
  --output-dir outputs/probes
```

## Settings that matter

- Generation requires a 40-character `--model-revision`.
- `run_hf_inference.py` renders prompts with the chat template by default. The ablation study used `--prompt-rendering raw --attention-implementation eager --batch-size 1 --seed 42 --max-new-tokens 4096` for the baseline and the same settings for interventions.
- Generation outputs get a `.protocol.json` file recording input and code hashes, settings and library versions. Resuming with different inputs or settings is refused; use a new output path instead.
- The specificity analysis checks the study's baseline counts (27, 25 and 29 step-present cases) and expects five random directions per target. A new dataset needs these checks updated.
- Input rows use the study's field names: `domain="domain_b"` and `variant="standard"` mark the security analyses.
