# Case study: security vulnerability reasoning

The paper evaluates three things separately for DeepSeek-R1-Distill-Qwen-14B on CyberSecEval tasks: whether reasoning steps can be read from activations (probes), whether intervening on a probe direction changes the output (ablation), and whether an LLM jury can judge the quality of the analysis (monitoring).

`examples/security_cases.jsonl` holds three of the model's analyses, each with the task and code, the reasoning, the final answer and the reference answer. They are examples for running the code, not an evaluation set.

## 1. The three pieces of evidence

```bash
python run.py security-evidence
```

This prints the saved probe AUCs, the ablation results, and the monitor accuracy, which is recomputed from saved predictions and human labels. Probe AUCs and ablation intervals are saved values; the ablation counts are re-checked.

The headline ablation result: removing the identification direction at layer 28 changes the verdict in 15 of 59 cases, against an average of 8.6 for five random directions, and removes identification content in 13 of 27 traces that had it, against 8.2 for random directions. The output also shows the other two directions, which have weaker or no effects, and a repeated baseline that reproduces all 59 outputs exactly.

## 2. Monitor results

```bash
python run.py results
python run.py results --bootstrap
```

These compare the saved jury predictions with human labels for each model's own responses (14B, 32B and 70B), and print exact, partial-credit and binary accuracy next to their baselines. [METHODS.md](../METHODS.md#monitor) defines each metric.

## 3. Activations and probes (GPU)

```bash
pip install -e '.[gpu]'
python run.py security-probes
```

This loads the pinned 14B model, teacher-forces each saved trace, and writes per-layer activations and step labels to `outputs/security_probes/`. With three cases there are too few examples for cross-validated AUC, so AUC is reported as unavailable. Pass `--input` and `--output` to run on your own cases. For the full protocol and the text-only and other-model controls, see [experiments/README.md](../experiments/README.md).

## 4. Direction ablation

```bash
python run.py ablation
```

This recomputes the ablation results on CPU from the released generations in `data/security/ablation/`: verdict changes and step suppression for each probe direction against its five random controls, with paired bootstrap contrasts. To regenerate the inputs on a GPU, the scripts run in this order:

1. `experiments/prepare_probe_direction_ablation.py`: fit the three directions on the 141 non-intervention cases, plus five random controls each.
2. `experiments/run_hf_inference.py`: generate a fresh baseline.
3. `experiments/run_probe_direction_ablation.py`: generate under each probe and random direction.
4. `experiments/analyze_fresh_probe_direction_ablation.py`: compare verdicts with the baseline.
5. `experiments/analyze_ablation_specificity.py` and `experiments/analyze_baseline_repeat.py`: cross-step effects and baseline repeatability.

[METHODS.md](../METHODS.md#direction-ablation) gives the settings; `data/security/ablation/protocol.json` records the ones used.

## 5. Run an LLM judge (paid API)

```bash
pip install -e '.[api]'
export OPENROUTER_API_KEY=...
python run.py security-monitor --provider openrouter --judges openai/gpt-5.4
python run.py score-monitor
```

This runs a new judge on the three examples and scores it against their human labels; it does not rerun the saved three-judge results. Pass several comma-separated model IDs for a jury. Provider options are in [REPRODUCTION.md](REPRODUCTION.md#llm-judge-options).
