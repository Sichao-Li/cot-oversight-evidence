"""Mean-pool post-block residual activations from a teacher-forced continuation."""
from ..hf import encode, hidden, hooks


def extract_span(model, tokenizer, prompt, continuation, layers, max_tokens=256):
    import torch
    if max_tokens < 1 or not layers:
        raise ValueError("Require layers and a positive continuation token limit.")
    prompt_ids = encode(model, tokenizer, prompt)
    continuation_ids = encode(model, tokenizer, continuation)[:, :max_tokens]
    start = prompt_ids.shape[1]
    inputs = torch.cat([prompt_ids, continuation_ids], dim=1)
    features = {}

    def capture(layer):
        def callback(_module, _args, output):
            features[layer] = hidden(output)[0, start:, :].mean(0).detach().float().cpu().numpy()
        return callback

    with torch.inference_mode(), hooks(model, {layer: capture(layer) for layer in layers}):
        model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, return_dict=True)
    return features, int(continuation_ids.shape[1])
