# Judge prompt

`security_jury_prompt.txt` is the structured security rubric used by each juror (the prompt in the paper's appendix). It receives the code, the model's reasoning, its final answer and the reference answer, never the human label.

The study prepended six calibration examples from a development split. They are not bundled, so a new run with this prompt is not an exact replication. To use it:

```bash
python run.py security-monitor --prompt prompts/security_jury_prompt.txt
```

Provider options are in [REPRODUCTION.md](../docs/REPRODUCTION.md#llm-judge-options).
