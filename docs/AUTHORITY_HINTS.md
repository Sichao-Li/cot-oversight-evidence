# Case study: authority-hint factual reasoning

The paper asks whether a behavior that a monitor can detect, a model following an authority cue to a wrong answer, can also be located with activation patching.

`examples/authority_pairs.jsonl` holds one MMLU college-chemistry question. The correct answer is B; the authority cue endorses A. The clean prompt is the hinted prompt with the authority preamble removed.

## Run the patch (GPU)

```bash
pip install -e '.[gpu]'
python run.py authority-patching
```

This loads the pinned 14B model and:

1. runs the clean prompt and records the activation at the chosen site;
2. runs the hinted prompt;
3. runs the hinted prompt again with that activation replaced by the clean one;
4. reports the correct-minus-hinted answer-letter logit margin for each run.

Results go to `outputs/authority_patching.json`. The default site is layer 24, 11 tokens before the end of the input. To try another site:

```bash
python run.py authority-patching --sites 28:-1 --output outputs/authority_other_site.json
```

Comma-separated sites are patched together.

## Scope

This example shows the procedure on one question. The paper's localization result rests on many cases, matched controls, and sites chosen on discovery cases before testing on held-out cases. To run that design on your own data, give each pair a `group` and a `split` (see [input formats](REPRODUCTION.md#input-formats)). The authority monitor and verbalization analyses from the paper are not included.
