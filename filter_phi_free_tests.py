#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

REPO_ROOT = Path.cwd()

SRC_DIRS = [
    REPO_ROOT / "llvm" / "test" / "CodeGen",
    REPO_ROOT / "llvm" / "test" / "Transforms",
]

OUT_ROOT = REPO_ROOT / "497"

def get_unique_dest(base_dir: Path, filename: str) -> Path:
    """
    Return a path under base_dir with filename, and if it already exists,
    append _1, _2, ... before the extension.
    """
    dest = base_dir / filename
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    i = 1
    while True:
        candidate = base_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    phi_re = re.compile(r"\bphi\b")

    for src_dir in SRC_DIRS:
        if not src_dir.is_dir():
            print(f"Warning: {src_dir} does not exist, skipping")
            continue

        for path in src_dir.rglob("*.ll"):
            try:
                with path.open("r", encoding="utf 8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                print(f"Could not read {path}: {e}")
                continue

            # Skip files that contain any phi instruction
            if phi_re.search(text):
                continue

            # Flatten into OUT_ROOT, handle name collisions
            dest = get_unique_dest(OUT_ROOT, path.name)
            shutil.copy2(path, dest)

if __name__ == "__main__":
    main()
