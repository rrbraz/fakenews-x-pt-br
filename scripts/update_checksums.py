"""Regenerate SHA-256 checksums for released data and result artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "metadata" / "artifact_checksums.sha256"


def main() -> None:
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and path.suffix in {".csv", ".parquet", ".json"}
        and "results/rerun" not in path.as_posix()
        and "data/generated" not in path.as_posix()
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}"
        for path in paths
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUTPUT.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
