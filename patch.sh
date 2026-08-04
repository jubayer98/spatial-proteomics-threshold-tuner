#!/usr/bin/env bash
#
# patch.sh — Fix spatialproteomics read-only buffer error
#
# Applies a one-line fix to spatialproteomics/la/label.py so that
# relabel_sequential receives a writable copy of the labels array.
# Works on macOS, Linux, and Windows (Git Bash / WSL).
#
# Usage:  bash patch.sh
#

set -euo pipefail

# Find the spatialproteomics package directory via Python
PKG_DIR=$(python -c "import spatialproteomics.la; from pathlib import Path; print(Path(spatialproteomics.la.__file__).parent)" 2>/dev/null) || {
    echo "ERROR: spatialproteomics is not installed in the current Python environment." >&2
    exit 1
}

TARGET="${PKG_DIR}/label.py"

# --- guardrail ---
if [ ! -f "$TARGET" ]; then
    echo "ERROR: $TARGET not found." >&2
    exit 1
fi

# --- apply patch ---
OLD='_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values)'
NEW='_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values.copy())'

if grep -qF "$NEW" "$TARGET"; then
    echo "Already patched: $TARGET"
    exit 0
fi

if ! grep -qF "$OLD" "$TARGET"; then
    echo "ERROR: Expected line not found in $TARGET. The library may have changed." >&2
    exit 1
fi

# Use Python for reliable cross-platform inline replacement
python -c "
import pathlib
p = pathlib.Path('${TARGET}')
content = p.read_text()
content = content.replace(
    '_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values)',
    '_, fw, _ = relabel_sequential(self._obj.coords[Dims.LABELS].values.copy())'
)
p.write_text(content)
" && echo "Patched: $TARGET" || {
    echo "ERROR: Failed to patch $TARGET" >&2
    exit 1
}
