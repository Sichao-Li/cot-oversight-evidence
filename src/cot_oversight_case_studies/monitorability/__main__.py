"""Run a structured jury through a provider API, or evaluate saved predictions."""
import argparse
import json
from pathlib import Path
from .api import ENDPOINTS, create_client, request_vote, validate_parameters
from .jury import aggregate, parse_vote, render_prompt, validate_template
from .evaluation import evaluate
from ..io import fingerprint, load_rows, write_new


def response_hash(row):
    return fingerprint({"thinking_tokens": row["thinking_tokens"], "answer_text": row["answer_text"]})


def run(args):
    rows = load_rows(args.input)
    template = Path(args.prompt).read_text()
    fewshot = Path(args.fewshot).read_text() if args.fewshot else ""
    validate_template(template, fewshot)
    parameters = validate_parameters(json.loads(Path(args.parameters).read_text()) if args.parameters else {})
    judges = [judge.strip() for judge in args.judges.split(",")]
    if any(not judge.strip() for judge in judges):
        raise ValueError("Judge model IDs cannot be empty.")
    prompts = [render_prompt(template, row, fewshot) for row in rows]
    protocol = {"input_sha256": fingerprint(rows), "prompt_sha256": fingerprint(template),
                "fewshot_sha256": fingerprint(fewshot), "provider": args.provider, "judges": judges,
                "parameters_sent": parameters, "unspecified_parameters": "provider defaults", "schema": 1}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing output uses different inputs, prompt, jurors or decoding; choose a fresh directory.")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Output directory is not empty and has no matching protocol.")
        write_new(protocol_path, protocol)
    log = args.output / "votes.jsonl"
    cached = {}
    if log.exists():
        for line in log.read_text().splitlines():
            item = json.loads(line)
            cached[(item["id"], item["juror_index"])] = item
    client = None
    results = []
    for row, prompt in zip(rows, prompts):
        votes = []
        for index, judge in enumerate(judges):
            key = (row["id"], index)
            item = cached.get(key)
            if item is None or args.retry_errors and item.get("vote") is None:
                if client is None:
                    client = create_client(args.provider)
                item = {"id": row["id"], "juror_index": index, "requested_model": judge, "vote": None}
                try:
                    item.update(request_vote(client, judge, prompt, parameters))
                    if item["finish_reason"] != "stop":
                        raise ValueError("Incomplete or filtered completion.")
                    item["vote"] = parse_vote(item["text"])
                except Exception as exc:
                    # Provider errors can contain request bodies or credentials. Store
                    # only their type/status, never stringify an arbitrary exception.
                    item["error_type"] = type(exc).__name__
                    item["http_status"] = getattr(exc, "status_code", None)
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item) + "\n")
            votes.append(item["vote"])
        results.append({"id": row["id"], "response_sha256": response_hash(row), **aggregate(votes, expected_jurors=len(judges))})
    # Derived summaries are recomputed from append-only juror records on resume.
    (args.output / "predictions.json").write_text(json.dumps(results, indent=2) + "\n")
    incomplete = [row["id"] for row in results if not row["complete"]]
    print(json.dumps({"cases": len(results), "complete": len(results)-len(incomplete), "incomplete_ids": incomplete}))
    return 1 if incomplete else 0


def score(args):
    predictions = json.loads(Path(args.predictions).read_text())
    annotations = load_rows(args.labels)
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs.")
    if set(by_id) != {row["id"] for row in annotations}:
        raise ValueError("Prediction and annotation case sets must match; exclusions require explicit UNCERTAIN labels.")
    for row in annotations:
        if row["id"] not in by_id or row.get("response_sha256") != by_id[row["id"]].get("response_sha256") or not row.get("response_sha256"):
            raise ValueError(f"Response-specific annotation hash mismatch: {row['id']}")
    result = evaluate({key: row["label"] if row.get("complete") else None for key, row in by_id.items()},
                      {row["id"]: row["label"] for row in annotations}, bootstrap=args.bootstrap, seed_offset=args.seed_offset)
    write_new(args.output, result)
    print(json.dumps(result))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    judge = sub.add_parser("judge")
    judge.add_argument("--input", required=True)
    judge.add_argument("--prompt", required=True)
    judge.add_argument("--fewshot", help="Optional calibration examples from a development split.")
    judge.add_argument("--provider", choices=ENDPOINTS, required=True)
    judge.add_argument("--judges", required=True, help="Comma-separated provider model IDs; duplicates request independent votes.")
    judge.add_argument("--parameters", help="JSON of only explicitly sent decoding parameters.")
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--retry-errors", action="store_true", help="Explicitly retry failed votes; extra API calls may be billed.")
    scoring = sub.add_parser("score")
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--labels", required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--bootstrap", action="store_true")
    scoring.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    return run(args) if args.command == "judge" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
