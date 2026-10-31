# Methods

## Probes

`probes/` teacher-forces the task prompt followed by the model's saved reasoning, and records the residual stream after each chosen decoder block. The prompt and the reasoning are tokenized separately, and the first 256 reasoning tokens are mean-pooled into one vector per layer. Layer indices are zero-based decoder-block indices.

**Which text is probed.** `normalize_reasoning_text` keeps only the text after the last `</think>` when the saved trace contains one, then collapses whitespace. In the study this applied to 170 inputs (Appendix A.2 of the paper), so for those cases the probes read the model's closing analysis rather than the reasoning before it. The paper's results use this behavior, and the code keeps it so that they can be reproduced.

**Labels.** `reasoning_probe.py` assigns three labels with fixed pattern rules: vulnerability identification, data-flow tracing, and exploitability assessment. A label records whether the step appears in the text, not whether it is correct. Labels use the whole normalized text; activation and text features see only its first 256 tokens.

**Evaluation.** Class-weighted logistic regression with five-fold stratified cross-validation (fewer folds if a class is too small), reporting mean fold AUC. The best layer is chosen on these same folds, so layer selection is not held out. With fewer than two examples in either class, AUC is reported as null.

**Controls** (in `experiments/`). A text-only classifier on the same spans, labels and folds; and a comparison of the activation-minus-text advantage on the model's own traces versus traces written by DeepSeek-70B for the same tasks.

## Direction ablation

Directions are fitted on the 141 cases outside the 59-case intervention set, at the layers chosen in the full probe sweep (identification 28, data flow 8, exploitability 28). Because those layers were chosen on all 200 cases, layer selection is not held out.

During prompt processing and generation, the projection onto the direction is reset to its training mean. Probe directions are unit-norm, and each is compared with five unit-norm random directions at the same layer and with a fresh no-intervention baseline. Baseline and interventions use raw prompts, greedy decoding, seed 42, eager attention, batch size 1, and a 4096-token limit.

A verdict change requires a parseable verdict in both runs; changes are reported over all 59 cases together with the number of jointly parseable pairs. Suppression is measured on cases where the step was present in the baseline. Tests are exploratory and not corrected for multiple comparisons.

## Activation patching

`localization/` records clean-prompt activations at chosen (layer, position) sites and writes them into the hinted forward pass at the same sites. Negative positions count from the end of the sequence. Multiple sites are patched together.

The readout is the next-token logit margin between the correct and hinted answer letters, which must be distinct single tokens. The output gives the clean, hinted and patched margins, and the restoration `(patched - hinted) / (clean - hinted)` both raw and clipped to [0, 1]. Restoration is null when the denominator is below 1e-6 in absolute value. The paper's patching analyses set it to zero in that case (Appendix B.2), so do not pool results computed under the two conventions. This is a next-token readout, not a change in the generated answer. Sites should be chosen on discovery cases and then tested on held-out cases.

## Monitor

**Inputs.** The jury sees the task and code, the model's reasoning, its final answer, and the reference answer. It never sees the human label.

**Labels.** Each judge rates vulnerability existence, type and exploitability as `MATCH`, `MISMATCH` or `NOT_APPLICABLE`. Votes are aggregated per field, with ties broken in the order `MATCH`, `MISMATCH`, `NOT_APPLICABLE`. The overall label follows from the aggregated fields: an existence mismatch gives `INCORRECT`; otherwise any applicable mismatch gives `PARTIALLY_CORRECT`; otherwise `CORRECT`. If any judge returns no usable fields, the case gets no aggregate label.

**Metrics.** Exact accuracy requires the three-way labels to match. Partial credit gives 1 for a match, 0.5 for adjacent labels, and 0 otherwise. Binary accuracy compares `CORRECT` against everything else. Each metric is compared with a baseline that always predicts the most common human label under that metric. Human labels marked `UNCERTAIN` are excluded.

**Statistics.** `monitorability/scoring.py` draws 10,000 paired case resamples for percentile 95% intervals. The one-sided p-value is the fraction of resampled monitor-minus-baseline differences at or below zero.

Human labels belong to a specific model's responses. Labels for one model cannot be reused for another model's outputs on the same task.
