"""Patch the same post-decoder-block site in clean and hinted forward passes."""
from ..hf import encode, hidden, hooks, replace_hidden
from math import isfinite


def position_index(position, length):
    if not -length <= position < length:
        raise ValueError(f"Position {position} outside sequence length {length}.")
    return position % length


def evaluate_patch(model, tokenizer, clean_prompt, hinted_prompt, correct_answer, hint_answer, sites):
    import torch
    if not sites or len(sites) != len(set(sites)):
        raise ValueError("Specify unique (zero-based layer, token position) sites.")
    candidates = [tokenizer(" " + answer, add_special_tokens=False)["input_ids"] for answer in (correct_answer, hint_answer)]
    if any(len(ids) != 1 for ids in candidates) or candidates[0] == candidates[1]:
        raise ValueError("Readout requires distinct single-token answer letters with a leading space.")
    correct_id, hint_id = (ids[0] for ids in candidates)
    clean_ids, hinted_ids = (encode(model, tokenizer, p) for p in (clean_prompt, hinted_prompt))
    positions = {}
    for layer, position in sites:
        clean_pos = position_index(position, clean_ids.shape[1])
        hinted_pos = position_index(position, hinted_ids.shape[1])
        positions.setdefault(layer, []).append((clean_pos, hinted_pos))
    saved = {}

    def capture(layer):
        def callback(_module, _args, output):
            saved[layer] = [hidden(output)[:, p, :].detach().clone() for p, _ in positions[layer]]
        return callback

    def patch(layer):
        def callback(_module, _args, output):
            tensor = hidden(output).clone()
            for value, (_, p) in zip(saved[layer], positions[layer]):
                tensor[:, p, :] = value.to(tensor)
            return replace_hidden(output, tensor)
        return callback

    def margin():
        logits = model(input_ids=hinted_ids, use_cache=False).logits[0, -1].float()
        return float((logits[correct_id] - logits[hint_id]).item())

    with torch.inference_mode():
        with hooks(model, {layer: capture(layer) for layer in positions}):
            logits = model(input_ids=clean_ids, use_cache=False).logits[0, -1].float()
            clean = float((logits[correct_id] - logits[hint_id]).item())
        hinted = margin()
        with hooks(model, {layer: patch(layer) for layer in positions}):
            patched = margin()
    denominator = clean - hinted
    if not all(isfinite(value) for value in (clean, hinted, patched, denominator)):
        raise ValueError("Nonfinite logits/margins; restoration is not defined for this forward pass.")
    raw = (patched - hinted) / denominator if abs(denominator) >= 1e-6 else None
    return {"clean_margin": clean, "hinted_margin": hinted, "patched_margin": patched,
            "margin_change": patched - hinted, "raw_restoration": raw,
            "clipped_restoration": min(1., max(0., raw)) if raw is not None else None,
            "denominator_valid": raw is not None, "sites": sites,
            "readout": "next-token correct-minus-hint logit margin; not a generated-answer flip"}
