#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
# Run DTL unit tests
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="build"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_tests.sh [build_dir] [-- <ctest-args...>]"
    exit 0
fi

if [[ $# -gt 0 && "${1:-}" != "--" && ! "${1:-}" =~ ^- ]]; then
    BUILD_DIR="$1"
    shift
fi

if [[ "${1:-}" == "--" ]]; then
    shift
fi

cd "${ROOT_DIR}"
# Build the configured test executables before invoking CTest.
cmake --build "${BUILD_DIR}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure "$@"
