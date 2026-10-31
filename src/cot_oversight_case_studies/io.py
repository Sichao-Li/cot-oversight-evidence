"""Input/output and hashing helpers shared by the package and the experiment scripts."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def load_rows(path):
    rows, ids = [], set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on input line {line_number}.") from exc
        case_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Input line {line_number} needs a nonempty string case ID.")
        if case_id in ids:
            raise ValueError(f"Duplicate case ID on input line {line_number}.")
        ids.add(case_id)
        rows.append(row)
    if not rows:
        raise ValueError("Input must contain at least one case.")
    return rows


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


# JSONL and hashing helpers used by the experiment scripts.

def load_jsonl(path, *, missing_ok=False):
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(cache_dir, record_id):
    """Activation-cache file for one case, keyed by a hash of its ID."""
    return Path(cache_dir) / f"{hashlib.sha256(record_id.encode('utf-8')).hexdigest()[:20]}.npz"
