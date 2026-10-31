"""Small CPU fixtures test hooks; real demonstrations never use these fixtures."""
from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from cot_oversight_case_studies.probes import extract_span
from cot_oversight_case_studies.localization import evaluate_patch
from cot_oversight_case_studies.hf import hooks


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 64 for c in text.strip()]}


class Block(torch.nn.Module):
    def forward(self, x):
        return (x * 1.1,)


class SmallDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.model = DecoderBody(self.embedding)
        self.head = torch.nn.Linear(8, 64)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, use_cache=False):
        x = self.model(input_ids)
        return SimpleNamespace(logits=self.head(x))


class DecoderBody(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.layers = torch.nn.ModuleList([Block(), Block()])

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)[0]
        return x


def test_span_and_final_site():
    torch.manual_seed(42)
    model, tokenizer = SmallDecoder().eval(), Tokenizer()
    features, n = extract_span(model, tokenizer, "prompt", "abcd", [0, 1], max_tokens=2)
    expected = model.embedding(torch.tensor([ord(c) % 64 for c in "ab"])).mean(0).detach().numpy() * 1.1
    np.testing.assert_allclose(features[0], expected, atol=1e-6)
    assert n == 2
    result = evaluate_patch(model, tokenizer, "a", "b", "A", "B", [(1, -1)])
    assert result["patched_margin"] == pytest.approx(result["clean_margin"], abs=1e-6)
    assert result["clipped_restoration"] == pytest.approx(1.)
    assert all(not layer._forward_hooks for layer in model.model.layers)
    undefined = evaluate_patch(model, tokenizer, "a", "a", "A", "B", [(0, -1)])
    assert undefined["raw_restoration"] is None


def test_cleanup_and_bounds():
    model, tokenizer = SmallDecoder().eval(), Tokenizer()
    def fail(*args):
        raise RuntimeError("intentional test failure")
    with pytest.raises(RuntimeError):
        with hooks(model, {0: fail}):
            model(torch.tensor([[1, 2]]))
    assert not model.model.layers[0]._forward_hooks
    with pytest.raises(ValueError):
        evaluate_patch(model, tokenizer, "a", "b", "A", "B", [(0, -2)])
    with pytest.raises(ValueError):
        evaluate_patch(model, tokenizer, "a", "b", "AA", "B", [(0, -1)])
    with pytest.raises(ValueError):
        with hooks(model, {0: lambda *args: None, 10: lambda *args: None}):
            pass
    assert not model.model.layers[0]._forward_hooks


def test_nonfinite_readout_not_reported_as_restoration():
    model, tokenizer = SmallDecoder().eval(), Tokenizer()
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate_patch(model, tokenizer, "a", "b", "A", "B", [(1, -1)])
    assert all(not layer._forward_hooks for layer in model.model.layers)
