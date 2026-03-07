#!/usr/bin/env bash
# Format all C++ headers in include/ and backends/ using clang-format.
# Usage: bash scripts/format_code.sh

set -euo pipefail

CLANG_FORMAT="${CLANG_FORMAT:-}"

# Auto-detect clang-format binary
if [ -z "$CLANG_FORMAT" ]; then
    if command -v clang-format-18 &>/dev/null; then
        CLANG_FORMAT="clang-format-18"
    elif command -v clang-format &>/dev/null; then
        CLANG_FORMAT="clang-format"
    else
        echo "Error: clang-format not found. Install clang-format-18 or set CLANG_FORMAT." >&2
        exit 1
    fi
fi

echo "Using: $($CLANG_FORMAT --version)"

find include/ \( -name "*.hpp" -o -name "*.h" \) -print0 | xargs -0 -r "$CLANG_FORMAT" -i

if [ -d "backends/" ]; then
    find backends/ \( -name "*.hpp" -o -name "*.h" \) -print0 | xargs -0 -r "$CLANG_FORMAT" -i
fi

echo "Formatting complete."
