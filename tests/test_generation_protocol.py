from argparse import Namespace
from pathlib import Path
import pytest
from experiments.generation_protocol import guard_generation_run


def setup_args(tmp_path):
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"record_id":"example","prompt":"task"}\n')
    return Namespace(model_revision="a" * 40, model="example", max_new_tokens=64, batch_size=1,
                     manifest=manifest, output=tmp_path / "output.jsonl", local_files_only=True,
                     shard_index=0, num_shards=1)


def test_matching_resume_and_changed_config(tmp_path):
    args = setup_args(tmp_path)
    guard_generation_run(args, __file__)
    args.output.write_text('{"record_id":"example","status":"ok"}\n')
    guard_generation_run(args, __file__)
    args.max_new_tokens = 128
    with pytest.raises(ValueError, match="protocol mismatch"):
        guard_generation_run(args, __file__)


def test_input_content_changes_detected(tmp_path):
    args = setup_args(tmp_path)
    guard_generation_run(args, __file__)
    args.manifest.write_text('{"record_id":"example","prompt":"DIFFERENT"}\n')
    with pytest.raises(ValueError, match="protocol mismatch"):
        guard_generation_run(args, __file__)


def test_unverified_output_and_unpinned_revision(tmp_path):
    args = setup_args(tmp_path)
    args.output.write_text("existing output")
    with pytest.raises(ValueError, match="no verified protocol"):
        guard_generation_run(args, __file__)
    args.model_revision = "main"
    with pytest.raises(ValueError, match="immutable"):
        guard_generation_run(args, __file__)


def test_duplicate_inputs_rejected_before_writing_protocol(tmp_path):
    args = setup_args(tmp_path)
    args.manifest.write_text(args.manifest.read_text() * 2)
    with pytest.raises(ValueError, match="unique record_id"):
        guard_generation_run(args, __file__)
    assert not args.output.with_suffix(".jsonl.protocol.json").exists()


def test_preflight_present_in_both_generation_entrypoints():
    root = Path(__file__).resolve().parents[1] / "experiments"
    for name in ("run_hf_inference.py", "run_probe_direction_ablation.py"):
        code = (root / name).read_text()
        assert code.index("guard_generation_run(args, __file__)") < code.index("model = AutoModelForCausalLM.from_pretrained")
