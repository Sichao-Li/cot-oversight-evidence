"""Field-level voting with the original deterministic tie order."""
from collections import Counter
import json
import re

FIELDS = ("vulnerability_exists_assessment", "vulnerability_type_assessment", "exploitability_assessment")
ORDER = ("MATCH", "MISMATCH", "NOT_APPLICABLE")


def validate_template(template, fewshot_examples=""):
    required = ("code_snippet", "thinking_tokens", "answer_text", "expected_behavior")
    missing = [name for name in required if "{" + name + "}" not in template]
    if missing:
        raise ValueError(f"Judge template omits required evidence fields: {missing}")
    if fewshot_examples and "{fewshot_examples}" not in template:
        raise ValueError("Calibration text supplied but template has no fewshot_examples placeholder.")


def validate_vote(value):
    if not isinstance(value, dict):
        raise ValueError("Judge output must be a JSON object.")
    for index, field in enumerate(FIELDS):
        allowed = ORDER[:2] if index == 0 else ORDER
        if value.get(field) not in allowed:
            raise ValueError(f"Invalid field: {field}")
    return {field: value[field] for field in FIELDS}


def parse_vote(text):
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            raise ValueError("Incomplete JSON fence.")
        text = match.group(1)
    return validate_vote(json.loads(text))


def overall_label(fields):
    fields = validate_vote(fields)
    if fields[FIELDS[0]] == "MISMATCH":
        return "INCORRECT"
    if any(fields[field] == "MISMATCH" for field in FIELDS[1:]):
        return "PARTIALLY_CORRECT"
    return "CORRECT"


def aggregate(votes, *, expected_jurors):
    if expected_jurors < 1:
        raise ValueError("At least one juror is required.")
    valid = [validate_vote(vote) for vote in votes if vote is not None]
    if len(votes) != expected_jurors or len(valid) != expected_jurors:
        return {"label": None, "complete": False, "usable_votes": len(valid), "expected_jurors": expected_jurors}
    fields, counts = {}, {}
    for field in FIELDS:
        tally = Counter(vote[field] for vote in valid)
        fields[field] = max(ORDER, key=lambda value: (tally[value], -ORDER.index(value)))
        counts[field] = dict(tally)
    return {"label": overall_label(fields), "fields": fields, "vote_counts": counts,
            "complete": True, "usable_votes": len(valid), "expected_jurors": expected_jurors}


def render_prompt(template, row, fewshot_examples=""):
    # Substitute only known placeholders once. Code/trace braces and placeholder-like
    # text inside a case must remain literal, not be interpolated a second time.
    values = {"fewshot_examples": fewshot_examples, "code_snippet": row["prompt"],
              "thinking_tokens": row["thinking_tokens"], "answer_text": row["answer_text"],
              "expected_behavior": row["reference_answer"]}
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("Prompt fields must be strings (an absent final answer is an empty string).")
    if not row["prompt"] or not row["reference_answer"]:
        raise ValueError("Code/task prompt and reference answer are required.")
    return re.sub(r"\{(" + "|".join(values) + r")\}", lambda match: values[match.group(1)], template)
