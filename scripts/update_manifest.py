"""Regenerate release_manifest.json: one SHA-256 hash per tracked release file."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.json"
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "build", "dist"}
SKIP_FILES = {MANIFEST.name, ".DS_Store"}
SCOPE = ("Probe, patching and monitor code; example cases; saved probe, ablation "
         "and model-specific monitor results.")


def release_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if (path.is_file() and path.name not in SKIP_FILES
                and not SKIP_DIRS.intersection(relative.parts)
                and not any(part.endswith(".egg-info") for part in relative.parts)):
            yield relative.as_posix(), path


def main():
    files = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in release_files()}
    MANIFEST.write_text(json.dumps({"schema": 1, "scope": SCOPE, "files": files}, indent=2) + "\n")
    print(f"Wrote {len(files)} hashes to {MANIFEST.name}.")


if __name__ == "__main__":
    main()
