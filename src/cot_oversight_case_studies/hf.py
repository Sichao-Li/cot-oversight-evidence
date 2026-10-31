"""Hugging Face decoder-block access for Llama/Qwen-style causal models."""
from contextlib import contextmanager
import importlib.metadata
import re


def load_model(name, revision, *, local_files_only=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision or ""):
        raise ValueError("Specify an immutable model revision.")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, local_files_only=local_files_only,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", attn_implementation="eager",
    ).eval()
    return model, tokenizer


def blocks(model):
    try:
        return model.model.layers
    except AttributeError as exc:
        raise ValueError("Expected a Llama/Qwen-style model.model.layers architecture.") from exc


def encode(model, tokenizer, text):
    import torch
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Cannot evaluate an empty token sequence.")
    return torch.tensor([ids], device=model.get_input_embeddings().weight.device)


def hidden(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, tensor):
    return (tensor, *output[1:]) if isinstance(output, tuple) else tensor


@contextmanager
def hooks(model, callbacks):
    handles = []
    layers = blocks(model)
    try:
        for layer, callback in callbacks.items():
            if not 0 <= layer < len(layers):
                raise ValueError(f"Layer {layer} outside model depth {len(layers)}.")
            handles.append(layers[layer].register_forward_hook(callback))
        yield
    finally:
        for handle in handles:
            handle.remove()


def runtime(model):
    import torch
    return {"model": model.config._name_or_path, "resolved_revision": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "dtype": str(model.dtype), "attention": getattr(model.config, "_attn_implementation", None),
            "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
            "prompt_rendering": "raw", "batch_size": 1}
