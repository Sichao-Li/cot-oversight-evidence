# Reproducing and extending

## What runs from this repository alone

| Command | Reproduces | Needs |
|---|---|---|
| `python run.py results` | Monitor accuracy for the 14B, 32B and 70B models | CPU |
| `python run.py security-evidence` | Probe, ablation and monitor summary for the security study (saved values) | CPU |
| `python run.py ablation` | Direction-ablation results recomputed from the released generations: 15/59 verdict changes, 13/27 suppression, random controls, bootstrap contrasts | CPU |
| `python run.py security-probes` | Activation extraction on three examples | GPU |
| `python run.py authority-patching` | Activation patching on one example | GPU |
| `python run.py security-monitor` | A new LLM judge run on three examples | API key |

The monitor results use these model-specific human labels:

| Model | Cases | `UNCERTAIN` labels excluded | Evaluated |
|---|---:|---:|---:|
| DeepSeek-R1-Distill-Qwen-14B | 59 | 0 | 59 |
| QwQ-32B | 59 | 5 | 54 |
| DeepSeek-R1-Distill-Llama-70B | 59 | 1 | 58 |

Every bundled file is listed with its SHA-256 hash in `release_manifest.json`. After editing files, regenerate it with `python scripts/update_manifest.py`.

## Full experiments

The generations behind the security study are in `data/` ([details](../data/README.md)). With them, the scripts in `experiments/` ([guide](../experiments/README.md)) rerun each analysis:

| Analysis | Scripts | Still needed |
|---|---|---|
| Reasoning-step probes | `run_reasoning_span_probe.py` on `data/security/generations_deepseek14b.jsonl` | Pinned 14B model (GPU) |
| Text-only control | `run_trace_only_probe_baselines.py` on the same file | Pinned tokenizer only (CPU) |
| Other-model traces | `analyze_cross_model_prefill.py` on both generation files | Activations from the probe run |
| Direction fitting | `prepare_probe_direction_ablation.py` | Activations from the probe run; fitted directions are included in `data/security/ablation/` |
| Ablation generations | `run_hf_inference.py`, `run_probe_direction_ablation.py` | Pinned 14B model (GPU); the outputs are included |
| Ablation analysis | `analyze_fresh_probe_direction_ablation.py`, `analyze_ablation_specificity.py`, `analyze_baseline_repeat.py` | Nothing; `python run.py ablation` runs the first |

The authority-hint study's full case set and patching manifests are not included.

Activation and generation results only match when the model revision, tokenizer, text normalization, prompt rendering, dtype, batching and decoding all match. The GPU path was tested on one H200 with DeepSeek-R1-Distill-Qwen-14B revision `1df8507178afcc1bef68cd8c393f61a886323761`, PyTorch 2.12 (nightly build), Transformers 4.57.6, bfloat16, eager attention and batch size 1. To check your setup:

```bash
pip install -e '.[gpu]'
python demos/real_model_smoke.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --revision 1df8507178afcc1bef68cd8c393f61a886323761 --no-save
```

This compares the two activation extractors and checks patching on real cases.

## Input formats

All JSONL inputs need a unique, nonempty string `id` on every line.

**Security analyses** (probes and monitor), as in `examples/security_cases.jsonl`:

| Field | Meaning |
|---|---|
| `prompt` | Full task prompt, including code |
| `thinking_tokens` | Saved reasoning text |
| `answer_text` | Saved final answer; empty string if none |
| `reference_answer` | Reference given to the monitor (never the human label) |
| `model` | Model that produced the response |

**Patching pairs**, as in `examples/authority_pairs.jsonl`: `clean_prompt`, `hinted_prompt`, `correct_answer` and `hint_answer` (answer letters), `group` (`changed` or `control`), and `split` (`discovery`, `heldout` or `demonstration`).

**Human labels** for scoring: `id`, `label` (`INCORRECT`, `PARTIALLY_CORRECT`, `CORRECT` or `UNCERTAIN`), and `response_sha256`, which ties the label to one exact response:

```python
from cot_oversight_case_studies.io import fingerprint
response_sha256 = fingerprint({"thinking_tokens": case["thinking_tokens"], "answer_text": case["answer_text"]})
```

## LLM judge options

`python -m cot_oversight_case_studies.monitorability judge --help` lists all options.

| `--provider` | Key variable | Model IDs |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | e.g. `openai/gpt-5.4` |
| `openai` | `OPENAI_API_KEY` | Direct OpenAI IDs |
| `gemini` | `GEMINI_API_KEY` | Direct Gemini IDs |

- `--parameters decoding.json` sets any of `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `seed` and `reasoning_effort`. Unset parameters use provider defaults.
- `--fewshot` adds calibration examples. The study used six examples from a development split; they are not bundled, so a new run is not an exact replication.
- Custom prompts must contain `{code_snippet}`, `{thinking_tokens}`, `{answer_text}` and `{expected_behavior}`, plus `{fewshot_examples}` when `--fewshot` is used.

The output directory holds `protocol.json` (input and prompt hashes, settings), `votes.jsonl` (one record per API call) and `predictions.json`. Rerunning into the same directory reuses completed votes; `--retry-errors` retries failed ones. A case with a missing vote gets no prediction and the run exits with status 1.

To score predictions:

```bash
python -m cot_oversight_case_studies.monitorability score \
  --predictions outputs/security_monitor/predictions.json \
  --labels examples/security_labels.jsonl \
  --output outputs/security_monitor/metrics.json --bootstrap
```
