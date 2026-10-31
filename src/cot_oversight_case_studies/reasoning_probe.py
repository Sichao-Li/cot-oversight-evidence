"""Reasoning-step labels and the linear-probe evaluation used for the security case study."""
from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np


REASONING_STEPS = (
    "identifies_vulnerability",
    "traces_data_flow",
    "assesses_exploitability",
)

NEGATED_VULNERABILITY_PATTERNS = (
    re.compile(r"\b(no|none|without) (?:security )?vulnerabilit(?:y|ies)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:does not|doesn't|did not|didn't) (?:contain|have|show|exhibit) "
        r"(?:(?:a|an|any) )?(?:security )?vulnerabilit(?:y|ies)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bnot vulnerable\b", re.IGNORECASE),
)

VULNERABILITY_PATTERNS = (
    re.compile(r"security vulnerability existence|whether a security vulnerability exists", re.IGNORECASE),
    re.compile(
        r"\b(vulnerability exists|the code contains(?: a)? vulnerability|there (?:is|are) "
        r"(?:a )?(?:potential )?(?:security )?vulnerability)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(type of vulnerability|cwe-\d+)\b", re.IGNORECASE),
    re.compile(
        r"\b(command injection|sql injection|xpath injection|path traversal|directory traversal|path injection|"
        r"memory corruption|use[- ]after[- ]free|double free|buffer overflow|null pointer dereference|"
        r"race condition|time of check to time of use|toctou|insecure deserialization|"
        r"deserialization vulnerability|code injection)\b",
        re.IGNORECASE,
    ),
)

SOURCE_PATTERNS = (
    re.compile(
        r"\b(user(?:[- ](?:supplied|provided|controlled))?|unsanitized input|untrusted input|"
        r"attacker(?:[- ]controlled)? input)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(response data|external input|remote input|request parameter|query parameter|headers?|body)\b", re.IGNORECASE),
    re.compile(
        r"\b(remotehost|userName|passWord|group|name|hostname|url|path|pid|queryString|comm|condition|"
        r"values|record|id|file_name|filename|extension_dir|pkgName|rpmPkgName|rpmPkgPath|version|"
        r"arch|topDir|llvm_config|ops->mod_name|mod_name|intString2|port)\b",
        re.IGNORECASE,
    ),
    re.compile(r"`[^`]+`", re.IGNORECASE),
)

SINK_PATTERNS = (
    re.compile(
        r"\b(Runtime\.exec|Runtime\.getRuntime\(\)\.exec|ProcessBuilder|Process\.Start|child_process\.exec|"
        r"os\.system|os\.popen|system\(|exec\(|eval\(|xpath\.compile|json\.loads|ExecuteNonQuery|"
        r"ExecuteScalar|cu\.execute|Topics\.Find|Command::new|new File|sys\.setdlopenflags|sprintf|"
        r"snprintf|dprintf|scanf\(|strcmp|memcmp)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:system|exec|eval|scanf)\s*\(", re.IGNORECASE),
    re.compile(r"\b(?:child_process\.exec|os\.popen|Process\.Start)\s*\(", re.IGNORECASE),
    re.compile(r"\b(sql query|xpath query|shell command|command string|file path)\b", re.IGNORECASE),
    re.compile(r"\b(strcpy|strcat|memcpy|lstrcatW|malloc|calloc|realloc|O_TRUNC)\b", re.IGNORECASE),
    re.compile(r"\b(compare|comparison|mismatch|break on the first mismatch)\b", re.IGNORECASE),
)

FLOW_PATTERNS = (
    re.compile(
        r"\b(concatenat(?:e|es|ed|ing)|construct(?:s|ed|ing)|directly embed(?:s|ded)?|passes? .* to|"
        r"used in|compiled into|reaches?|flows? to|propagat(?:e|es|ed|ion)|cop(?:y|ies|ied) into|"
        r"formatted into|inserted into|looked up by|used directly to construct|feeds? into|leads? to|"
        r"causes?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwithout proper (?:escaping|saniti[sz]ation|validation|locking|synchronization|"
        r"bounds checking|null checking)\b",
        re.IGNORECASE,
    ),
)

SPECIFIC_LINE_PATTERNS = (
    re.compile(
        r"\b(specific line\(s\) or condition\(s\)|specific lines? causing the vulnerability|"
        r"specific lines? or conditions?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe vulnerability (?:exists|is present) in the following lines?\b", re.IGNORECASE),
    re.compile(r"\bline where\b", re.IGNORECASE),
    re.compile(r"\bline \d+\b", re.IGNORECASE),
)

CONDITION_MECHANISM_PATTERNS = (
    re.compile(
        r"\b(null pointer dereference|returns? null|use[- ]after[- ]free|double free|buffer overflow|"
        r"out[- ]of[- ]bounds)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(without proper locking|without synchronization|filename collision|race condition|O_TRUNC)\b", re.IGNORECASE),
    re.compile(
        r"\b(unsafely constructs? .* command|directly concatenat(?:e|es|ed|ing) .* "
        r"(?:query|xpath|command|string|path))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(failed malloc|malloc (?:returns?|returned) null|embedded nul|undersized (?:size|allocation)|"
        r"size calculation|ineffective truncation|read[- ]only buffer|c_str\(\)|raw[- ]byte(?:s)?|"
        r"incorrect flag combination|dynamic library loading|early loop exit|first mismatch|time taken|"
        r"variable[- ]time|timing(?:[- ]side[- ]channel| depends| variation| leak| attacks?)|"
        r"unsynchronized filename generation|long [a-z_]+|oversized [a-z_]+|unchecked input|"
        r"unvalidated input)\b",
        re.IGNORECASE,
    ),
)

DIRECT_TRACE_PATTERNS = (
    re.compile(
        r"\b(?:unsanitized|untrusted|user(?:[- ](?:supplied|provided|controlled))?|"
        r"attacker(?:[- ]controlled)?)\b.*\b(?:concatenat(?:e|es|ed|ing)|construct(?:s|ed|ing)|"
        r"used in|passed to|compiled into|reaches?)\b.*\b(?:Runtime\.exec|ProcessBuilder|Process\.Start|"
        r"os\.system|os\.popen|system\(|exec\(|xpath\.compile|json\.loads|sql query|xpath query|"
        r"command string|file path)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:group|name|remoteHost|userName|passWord|hostname|url|path|pid|response data)\b.*\b"
        r"(?:concatenat(?:e|es|ed|ing)|construct(?:s|ed|ing)|used in|passed to|compiled into|"
        r"deseriali[sz]ed by)\b.*\b(?:xpath\.compile|json\.loads|Runtime\.exec|Process\.Start|"
        r"os\.popen|command string|file path)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"`[^`]+`.*\b(?:into|to|used directly to construct|passed to|formatted into|copied into|"
        r"inserted into|looked up by)\b.*\b(?:Runtime\.exec|Runtime\.getRuntime\(\)\.exec|"
        r"child_process\.exec|os\.popen|eval\(|xpath\.compile|json\.loads|ExecuteNonQuery|"
        r"ExecuteScalar|cu\.execute|Topics\.Find|Command::new|new File|sys\.setdlopenflags|sprintf|"
        r"snprintf|dprintf|scanf\(|strcpy|strcat|memcpy|lstrcatW|malloc)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hostname|path|pid|id|queryString|comm|condition|values|record|version|arch|pkgName|"
        r"rpmPkgName|rpmPkgPath|group|name|port|filename|file_name)\b.*\b(?:into|to|"
        r"used directly to construct|passed to|formatted into|copied into|inserted into|looked up by|"
        r"appended to)\b.*\b(?:child_process\.exec|os\.popen|Process\.Start|Runtime\.exec|"
        r"Runtime\.getRuntime\(\)\.exec|eval|ExecuteNonQuery|ExecuteScalar|cu\.execute|Topics\.Find|"
        r"Command::new|new File|sprintf|snprintf|strcpy|strcat|lstrcatW)\b",
        re.IGNORECASE,
    ),
)

NEGATED_EXPLOITABILITY_PATTERNS = (
    re.compile(r"\bnot (?:reachable|exploitable)\b", re.IGNORECASE),
    re.compile(r"\bunlikely to be exploitable\b", re.IGNORECASE),
)

EXPLOITABILITY_PATTERNS = (
    re.compile(r"\b(reachability and exploitability|under normal execution conditions)\b", re.IGNORECASE),
    re.compile(
        r"\b(exploitable|reachable|trigger(?:ed)?|attacker could|arbitrary code execution|"
        r"system compromise)\b",
        re.IGNORECASE,
    ),
)


def normalize_reasoning_text(row: dict[str, Any]) -> str:
    """Return the text the probes read, as used for the paper's results.

    If the saved trace contains ``</think>``, only the text after the last
    ``</think>`` is kept, so the probe reads the model's closing analysis rather
    than the reasoning before it. Whitespace is collapsed.
    """

    text = str(row.get("thinking_tokens") or row.get("answer_text") or row.get("raw_output") or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return re.sub(r"\s+", " ", text).strip()


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text or "") for pattern in patterns)


def step_presence_labels(text: str) -> dict[str, int]:
    """Label which reasoning steps appear in the text, using fixed pattern rules."""

    has_positive_vulnerability = _matches_any(text, VULNERABILITY_PATTERNS)
    has_negative_vulnerability = _matches_any(text, NEGATED_VULNERABILITY_PATTERNS)
    identifies_vulnerability = int(has_positive_vulnerability and not has_negative_vulnerability)

    has_source = _matches_any(text, SOURCE_PATTERNS)
    has_sink = _matches_any(text, SINK_PATTERNS)
    has_flow = _matches_any(text, FLOW_PATTERNS)
    has_specific_line = _matches_any(text, SPECIFIC_LINE_PATTERNS)
    has_condition_mechanism = _matches_any(text, CONDITION_MECHANISM_PATTERNS)
    has_direct_trace = _matches_any(text, DIRECT_TRACE_PATTERNS)
    if has_negative_vulnerability and not identifies_vulnerability:
        traces_data_flow = 0
    else:
        traces_data_flow = int(
            has_direct_trace
            or (has_source and has_sink and (has_flow or has_specific_line or has_condition_mechanism))
            or (has_sink and has_condition_mechanism and (has_flow or has_specific_line or has_source))
            or (has_specific_line and has_condition_mechanism and identifies_vulnerability)
        )

    has_positive_exploitability = _matches_any(text, EXPLOITABILITY_PATTERNS)
    has_negative_exploitability = _matches_any(text, NEGATED_EXPLOITABILITY_PATTERNS)
    assesses_exploitability = int(has_positive_exploitability and not has_negative_exploitability)

    return {
        "identifies_vulnerability": identifies_vulnerability,
        "traces_data_flow": traces_data_flow,
        "assesses_exploitability": assesses_exploitability,
    }


def parse_layer_list(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers or min(layers) < 0:
        raise ValueError("Layer lists must contain non-negative integers.")
    return layers


def relative_layer_sweep(
    source_layers: Iterable[int],
    *,
    source_depth: int,
    target_depth: int,
) -> list[int]:
    """Map zero-based source layers by fractional model depth."""

    if source_depth <= 0 or target_depth <= 0:
        raise ValueError("Model depths must be positive.")
    return sorted(
        {
            min(target_depth - 1, max(0, int(round(layer * target_depth / source_depth))))
            for layer in source_layers
        }
    )


def sample_confirmation_indices(rows: list[dict[str, Any]], *, n: int, seed: int) -> list[int]:
    candidates = [index for index, row in enumerate(rows) if row.get("is_correct") is True]
    if len(candidates) <= n:
        return sorted(candidates, key=lambda index: str(rows[index]["record_id"]))
    rng = np.random.default_rng(seed)
    sampled_offsets = sorted(rng.choice(len(candidates), size=n, replace=False).tolist())
    return [candidates[offset] for offset in sampled_offsets]


def cross_validated_probe_predictions(
    activations: np.ndarray,
    labels: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Reproduce the five-fold class-weighted logistic-probe evaluation."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

    x = np.asarray(activations)
    y = np.asarray(labels, dtype=int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    cv_folds = min(5, positives, negatives)
    if cv_folds < 2:
        return {
            "probabilities": np.full(len(y), 0.5),
            "auc_mean": 0.0,
            "auc_std": 0.0,
            "ap_mean": 0.0,
            "ap_std": 0.0,
            "cv_folds": cv_folds,
        }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    probabilities = cross_val_predict(classifier, x, y, cv=cv, method="predict_proba")[:, 1]
    auc_scores = cross_val_score(classifier, x, y, cv=cv, scoring="roc_auc")
    ap_scores = cross_val_score(classifier, x, y, cv=cv, scoring="average_precision")
    return {
        "probabilities": probabilities,
        "auc_mean": float(auc_scores.mean()),
        "auc_std": float(auc_scores.std()),
        "ap_mean": float(ap_scores.mean()),
        "ap_std": float(ap_scores.std()),
        "cv_folds": cv_folds,
    }
