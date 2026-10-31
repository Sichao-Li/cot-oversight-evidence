# CoT Oversight Case Studies

Code and data for the two case studies in

> **Position: Chain-of-Thought Oversight Requires Separate Evidence for Faithfulness and Monitorability**
> Sichao Li, Sai Ma, Xiyang Hu, Chudi Zhong, Tongliang Liu. NeurIPS 2026 (Position Paper Track).
> [OpenReview](https://openreview.net/forum?id=DLek3EUyds)

The paper argues that evidence about what a model represents, what causally drives its output, and what a monitor can detect should be reported separately. This repository contains the code behind its two case studies:

| Case study | What you can do here |
|---|---|
| Security vulnerability reasoning (Section 3) | Recompute the monitor results and the direction-ablation results from the released generations and labels; inspect the probe evidence; extract activations; run an LLM judge |
| Authority-hint factual reasoning (Section 4) | Patch activations from a clean prompt into a hinted prompt |

## Quick start

Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python run.py results              # monitor accuracy against human labels
python run.py security-evidence    # probe, intervention and monitor evidence together
python run.py ablation             # direction-ablation results recomputed from the released generations
```

All three run on CPU, the first two in seconds and the third in about a minute. No GPU or API key is needed. Add `--bootstrap` for confidence intervals and paired tests, or `--json` for machine-readable output. `python run.py --help` lists the GPU and API commands.

## Guides

- [Security vulnerability reasoning](docs/SECURITY_REASONING.md): probes, direction ablation and the LLM jury.
- [Authority-hint factual reasoning](docs/AUTHORITY_HINTS.md): clean-to-hinted activation patching.
- [Reproducing and extending](docs/REPRODUCTION.md): what is included, the full experiment workflows, and input formats for your own data.
- [Methods](METHODS.md): exact definitions of labels, metrics and interventions.

## Layout

| Path | Contents |
|---|---|
| `src/cot_oversight_case_studies/probes/` | Activation extraction and reasoning-step probes |
| `src/cot_oversight_case_studies/localization/` | Clean-to-hinted activation patching |
| `src/cot_oversight_case_studies/monitorability/` | LLM jury, voting and scoring |
| `experiments/` | Full-scale probe, control and intervention scripts ([guide](experiments/README.md)) |
| `data/` | The 200 DeepSeek-14B generations, the 59 held-out responses per model with human labels, and all ablation generations ([details](data/README.md)) |
| `examples/` | Three security cases, one authority-hint pair, and saved results ([details](examples/README.md)) |
| `prompts/` | Judge prompt |
| `demos/` | Scripts behind the `run.py` CPU commands, plus a GPU check |

## Tests

```bash
pip install -e '.[test]'
python -m pytest
```

## Citation

```bibtex
@inproceedings{li2026cotoversight,
  title     = {Position: Chain-of-Thought Oversight Requires Separate Evidence for Faithfulness and Monitorability},
  author    = {Li, Sichao and Ma, Sai and Hu, Xiyang and Zhong, Chudi and Liu, Tongliang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). Benchmark excerpts (CyberSecEval, MMLU) and model outputs in `data/` and `examples/` remain under their original terms. API commands incur provider charges; keep keys in environment variables.
