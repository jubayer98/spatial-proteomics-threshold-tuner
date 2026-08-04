"""
patch.py — Fix spatialproteomics read-only buffer error

Applies a one-line fix to spatialproteomics/la/label.py so that
relabel_sequential receives a writable copy of the labels array.
Works on macOS, Linux, and Windows.

Usage:  python patch.py
"""
import importlib.util
import pathlib
import sys


def main():
    spec = importlib.util.find_spec("spatialproteomics.la")
    if spec is None or spec.origin is None:
        print("ERROR: spatialproteomics is not installed in the current Python environment.")
        sys.exit(1)

    target = pathlib.Path(spec.origin)
    if not target.exists():
        print(f"ERROR: {target} not found.")
        sys.exit(1)

    old = "_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values)"
    new = "_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values.copy())"

    content = target.read_text()

    if new in content:
        print(f"Already patched: {target}")
        return

    if old not in content:
        print(f"ERROR: Expected line not found in {target}. The library may have changed.")
        sys.exit(1)

    patched = content.replace(old, new)
    target.write_text(patched)
    print(f"Patched: {target}")


if __name__ == "__main__":
    main()
