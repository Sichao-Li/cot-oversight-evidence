# Released data

The security case study's model generations and human labels. Together with the code, they reproduce every CPU-side result in Section 3 of the paper and provide the inputs for the GPU-side probes and interventions.

| File | Rows | Contents |
|---|---:|---|
| `security/generations_deepseek14b.jsonl` | 200 | DeepSeek-14B analyses of the 200 CyberSecEval tasks. Input to the probes (Section 3.1), the text-only control and direction fitting. |
| `security/generations_deepseek70b_prefilled_into_14b.jsonl` | 200 | DeepSeek-70B analyses of the same tasks, as replayed through the 14B model for the other-model-trace control. `trace_source_model` names the author of each trace. |
| `security/heldout59/deepseek14b.jsonl`, `qwq32b.jsonl`, `deepseek70b.jsonl` | 59 each | The responses the jury and the human annotators judged (Section 3.2). The 14B rows are the 59 held-out cases of the 200 above. |
| `security/heldout59/human_labels.jsonl` | 177 | One label per response: `INCORRECT`, `PARTIALLY_CORRECT`, `CORRECT` or `UNCERTAIN`, with the SHA-256 of the response it belongs to. The same labels, with the jury votes, are in `examples/saved_monitor.json`. |
| `security/ablation/baseline.jsonl`, `baseline_repeat.jsonl` | 59 each | Two no-intervention generations under the ablation protocol; the texts are identical. |
| `security/ablation/interventions.jsonl` | 1,062 | Generations under each of the 18 conditions (3 probe directions, 15 random directions) for the 59 held-out cases. |
| `security/ablation/intervention_plan.jsonl`, `direction_fit_summary.jsonl`, `directions.npz`, `protocol.json` | | The fitted directions, their training-mean projections, and the settings. |

## Row format

Generation rows share one schema. The fields the scripts read:

| Field | Meaning |
|---|---|
| `id`, `record_id` | Task ID; `record_id` is `id::variant` |
| `prompt` | Full task prompt including the code excerpt (Appendix D.2 of the paper) |
| `thinking_tokens` | Generated reasoning; `raw_output` is the untouched generation |
| `answer_text` | Text after `</think>`, or empty when the model produced none |
| `correct_answer` | Reference answer given to the monitor (also `metadata.expected_behavior`) |
| `is_correct`, `answer_security_label` | Whether the response's verdict matches the reference |
| `generated_token_count`, `hit_token_cap`, `token_counts` | Generation length and whether the 4,096-token limit was reached |
| `metadata` | CyberSecEval source: `category`, `language`, `cwe_identifier`, `source_id`, `file_path`, `pattern_desc` |
| `domain`, `variant`, `status`, `model` | Always `domain_b`, `standard`, `ok`, and the model identifier |

Intervention rows add `condition_id`, `intervention_kind`, `intervention_target_step`, `intervention_layer`, `intervention_projection_mean` and `generated_step_presence_labels`.

## Reproducing from these files

```bash
python run.py ablation          # 15/59 verdict changes, 13/27 suppression, random controls, p-values
```

For the probes and the text-only control, run the scripts in [experiments/README.md](../experiments/README.md) on `generations_deepseek14b.jsonl`; the probes need the pinned model on a GPU, the text-only control needs only its tokenizer.

## Sources and terms

Tasks and code excerpts come from [CyberSecEval](https://github.com/meta-llama/PurpleLlama) (`metadata` keeps each item's source ID and file path). Generations were produced by [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) and [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B). Excerpts and generations remain under their original terms; the repository's MIT licence covers the code and the human labels. The files contain no personal data.
