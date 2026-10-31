"""Last-explicit-verdict parser used by the paired intervention analyses."""

from __future__ import annotations

import re

EXPLICIT_NEGATIVE_PATTERNS = (
    re.compile(
        r"\bno (?:security )?vulnerabilit(?:y|ies)(?: exists?| (?:is|are) present)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does not|doesn't) (?:contain|have|show) (?:a |any )?"
        r"(?:security )?vulnerabilit(?:y|ies)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bnot vulnerable\b", re.IGNORECASE),
)

EXPLICIT_POSITIVE_PATTERNS = (
    re.compile(r"\bvulnerabilit(?:y|ies) (?:exists?|is present|are present)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:contains?|has) (?:a |an |one or more )?(?:security )?"
        r"vulnerabilit(?:y|ies)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:the )?code is vulnerable\b", re.IGNORECASE),
    re.compile(
        r"\b(?:vulnerability type|type of vulnerability)\s*:\s*"
        r"(?!none|not applicable|n/?a)",
        re.IGNORECASE,
    ),
)

def explicit_security_verdict(text: str | None) -> bool | None:
    """Return the last explicit existence verdict, avoiding broad cue words."""

    if not text:
        return None
    matches: list[tuple[int, bool]] = []
    for pattern in EXPLICIT_NEGATIVE_PATTERNS:
        matches.extend((match.end(), False) for match in pattern.finditer(text))
    for pattern in EXPLICIT_POSITIVE_PATTERNS:
        matches.extend((match.end(), True) for match in pattern.finditer(text))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]
