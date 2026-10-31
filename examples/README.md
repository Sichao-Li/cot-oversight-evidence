# Example data

| File | Contents |
|---|---|
| `security_cases.jsonl` | Three DeepSeek-R1-Distill-Qwen-14B security analyses with their task prompts and reference answers |
| `security_labels.jsonl` | Human quality labels for those three responses |
| `authority_pairs.jsonl` | One MMLU college-chemistry question, with and without the authority preamble |
| `saved_monitor.json` | Human labels, jury predictions and expected statistics for the 14B, 32B and 70B monitor results; the labelled responses themselves are in [`data/`](../data/README.md) |
| `saved_security_evidence.json` | Saved probe, control, ablation and baseline-repeat results for the security study |

The security cases are `cybersec_injection_instruct_v2_1144`, `cybersec_injection_instruct_v2_916` and `cybersec_memory_safety_instruct_166`. They were chosen as examples and are not a representative sample. Responses are stored as generated; where the model gave no separate final answer, `answer_text` is empty.

## Sources and terms

- Security tasks: [PurpleLlama / CyberSecEval](https://github.com/meta-llama/PurpleLlama). Each case keeps its source ID and CWE.
- Factual question: [MMLU](https://huggingface.co/datasets/cais/mmlu).
- Model outputs: [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B).

These excerpts and outputs remain under their original terms; the MIT license of this repository does not cover them. No personal data or credentials are included.
