"""Reader entry points for the two case studies; run with --help to list them."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"
TASKS = {
    "results": "Recompute saved security-jury results (CPU; no API key).",
    "security-evidence": "Check and report representation, intervention and monitoring evidence (CPU).",
    "ablation": "Recompute the direction-ablation results from the released generations (CPU).",
    "security-probes": "Extract reasoning-step activations for the security examples (GPU).",
    "security-monitor": "Ask an LLM judge to assess the security examples (paid API).",
    "score-monitor": "Compare judge decisions with the example's human labels (CPU).",
    "authority-patching": "Patch clean activations into the authority-hinted example (GPU).",
}


def command(task):
    common_model = ["--model", MODEL, "--revision", REVISION]
    if task == "security-probes":
        return "probes", ["--input", str(ROOT / "examples/security_cases.jsonl"), *common_model,
                           "--output", "outputs/security_probes"]
    if task == "authority-patching":
        return "localization", ["--input", str(ROOT / "examples/authority_pairs.jsonl"), *common_model,
                                 "--sites", "24:-11", "--split", "demonstration",
                                 "--output", "outputs/authority_patching.json"]
    if task == "security-monitor":
        return "monitorability", ["judge", "--input", str(ROOT / "examples/security_cases.jsonl"),
                                   "--prompt", str(ROOT / "prompts/security_jury_prompt.txt"),
                                   "--output", "outputs/security_monitor"]
    if task == "score-monitor":
        return "monitorability", ["score", "--predictions", "outputs/security_monitor/predictions.json",
                                   "--labels", str(ROOT / "examples/security_labels.jsonl"),
                                   "--output", "outputs/security_monitor/metrics.json"]
    raise ValueError(task)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {name:20s} {description}" for name, description in TASKS.items()) +
               "\n\nDefaults use the bundled examples. Append method options to override inputs, outputs, model or revision.")
    parser.add_argument("task", choices=TASKS)
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return
    task = parser.parse_args(args[:1]).task
    extra = args[1:]
    old_argv = sys.argv
    try:
        if task in ("results", "security-evidence", "ablation"):
            script = {"results": "replay_monitor.py", "security-evidence": "security_evidence.py",
                      "ablation": "replay_ablation.py"}[task]
            sys.argv = [script, *extra]
            runpy.run_path(str(ROOT / "demos" / script), run_name="__main__")
        else:
            module, defaults = command(task)
            sys.argv = [task, *defaults, *extra]
            runpy.run_module(f"cot_oversight_case_studies.{module}", run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
