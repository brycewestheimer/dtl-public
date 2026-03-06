#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
# Run C binding regression tests (single-rank + MPI multi-rank + RMA).
#
# Usage:
#   bash scripts/run_binding_regression.sh [build_dir]
#
# Examples:
#   bash scripts/run_binding_regression.sh
#   bash scripts/run_binding_regression.sh build/dev
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="build"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_binding_regression.sh [build_dir]"
    exit 0
fi

if [[ $# -gt 0 && ! "${1:-}" =~ ^- ]]; then
    BUILD_DIR="$1"
    shift
fi

cd "${ROOT_DIR}"

# Build before testing
cmake --build "${BUILD_DIR}"

TEST_BIN="${BUILD_DIR}/tests/bindings/c/test_c_bindings"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: C binding test binary not found at ${TEST_BIN}" >&2
    echo "Build with: cmake --build ${BUILD_DIR} --target test_c_bindings" >&2
    exit 1
fi

FAIL=0

# --- Single-rank C binding tests ---
echo ""
echo "=============================================="
echo "  C Binding Tests: single-rank"
echo "=============================================="
if ! "$TEST_BIN" --gtest_brief=1; then
    FAIL=1
fi

# --- MPI multi-rank tests (if mpirun available) ---
read -r -a MPIRUN_CMD <<<"${MPIRUN:-mpirun}"
read -r -a MPI_FLAG_ARR <<<"${MPI_FLAGS:---oversubscribe --bind-to none}"

if command -v "${MPIRUN_CMD[0]}" &>/dev/null; then
    echo ""
    echo "=============================================="
    echo "  C Binding Tests: 2 MPI ranks"
    echo "=============================================="
    if ! "${MPIRUN_CMD[@]}" "${MPI_FLAG_ARR[@]}" -np 2 "$TEST_BIN" --gtest_brief=1; then
        FAIL=1
    fi

    echo ""
    echo "=============================================="
    echo "  C Binding Tests: 2 MPI ranks (RMA filter)"
    echo "=============================================="
    if ! "${MPIRUN_CMD[@]}" "${MPI_FLAG_ARR[@]}" -np 2 "$TEST_BIN" --gtest_brief=1 --gtest_filter='*RMA*:*Rma*:*rma*'; then
        FAIL=1
    fi
else
    echo "SKIP: mpirun not found, skipping MPI multi-rank tests"
fi

echo ""
if [[ $FAIL -ne 0 ]]; then
    echo "=============================================="
    echo "  FAILED: C binding regression"
    echo "=============================================="
    exit 1
fi

echo "=============================================="
echo "  All C binding regression tests passed"
echo "=============================================="
