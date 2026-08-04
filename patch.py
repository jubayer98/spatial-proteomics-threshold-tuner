#!/usr/bin/env python3
"""
patch.py — Fix spatialproteomics read-only buffer error

Applies a one-line fix to spatialproteomics/la/label.py so that
relabel_sequential receives a writable copy of the labels array.
Works natively on macOS, Linux, and Windows.

Usage: python patch.py
"""

import sys
from pathlib import Path


def patch_spatialproteomics():
    # 1. Locate package directory safely
    try:
        import spatialproteomics.la
    except ImportError:
        sys.stderr.write(
            "ERROR: spatialproteomics is not installed in the current Python environment.\n"
        )
        sys.exit(1)

    target = Path(spatialproteomics.la.__file__).parent / "label.py"

    # 2. Guardrail check
    if not target.is_file():
        sys.stderr.write(f"ERROR: {target} not found.\n")
        sys.exit(1)

    # 3. Read content
    content = target.read_text(encoding="utf-8")

    old_line = "_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values)"
    new_line = "_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values.copy())"

    # 4. Check status & apply patch
    if new_line in content:
        print(f"Already patched: {target}")
        sys.exit(0)

    if old_line not in content:
        sys.stderr.write(
            f"ERROR: Expected line not found in {target}. The library may have changed.\n"
        )
        sys.exit(1)

    patched_content = content.replace(old_line, new_line)
    target.write_text(patched_content, encoding="utf-8")
    print(f"Patched: {target}")


if __name__ == "__main__":
    patch_spatialproteomics()
