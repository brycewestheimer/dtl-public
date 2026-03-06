#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
# Pre-release parity gate: runs all test suites and rejects on any failure.
#
# Usage:
#   bash scripts/run_parity_gate.sh [build_dir]
#
# Examples:
#   bash scripts/run_parity_gate.sh build/dev-cuda
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="build"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash scripts/run_parity_gate.sh [build_dir]"
    exit 0
fi

if [[ $# -gt 0 && ! "${1:-}" =~ ^- ]]; then
    BUILD_DIR="$1"
    shift
fi

cd "${ROOT_DIR}"

FAIL=0

run_step() {
    local name="$1"
    shift
    echo ""
    echo "=============================================="
    echo "  PARITY GATE: ${name}"
    echo "=============================================="
    if ! "$@"; then
        echo "  FAILED: ${name}"
        FAIL=1
    fi
}

# Build
run_step "Build" cmake --build "${BUILD_DIR}"

# Unit tests
UNIT_BIN="${BUILD_DIR}/tests/dtl_unit_tests"
if [ -x "$UNIT_BIN" ]; then
    run_step "Unit tests" "$UNIT_BIN" --gtest_brief=1
fi

# C binding tests (single-rank)
C_BIN="${BUILD_DIR}/tests/bindings/c/test_c_bindings"
if [ -x "$C_BIN" ]; then
    run_step "C binding tests (single-rank)" "$C_BIN" --gtest_brief=1
fi

# MPI multi-rank tests
read -r -a MPIRUN_CMD <<<"${MPIRUN:-mpirun}"
read -r -a MPI_FLAG_ARR <<<"${MPI_FLAGS:---oversubscribe --bind-to none}"

if command -v "${MPIRUN_CMD[0]}" &>/dev/null; then
    if [ -x "$UNIT_BIN" ]; then
        run_step "MPI tests (2 ranks)" \
            "${MPIRUN_CMD[@]}" "${MPI_FLAG_ARR[@]}" -np 2 "$UNIT_BIN" --gtest_brief=1
        run_step "MPI tests (4 ranks)" \
            "${MPIRUN_CMD[@]}" "${MPI_FLAG_ARR[@]}" -np 4 "$UNIT_BIN" --gtest_brief=1
    fi

    if [ -x "$C_BIN" ]; then
        run_step "C binding MPI (2 ranks)" \
            "${MPIRUN_CMD[@]}" "${MPI_FLAG_ARR[@]}" -np 2 "$C_BIN" --gtest_brief=1
    fi
fi

# CUDA placement tests
if [ -x "$C_BIN" ]; then
    run_step "CUDA placement tests" \
        "$C_BIN" --gtest_brief=1 --gtest_filter='*Device*:*Cuda*:*CUDA*:*Unified*'
fi

# NCCL integration tests via CTest label
if ctest --test-dir "${BUILD_DIR}" -N -L nccl 2>/dev/null | grep -q 'Test #'; then
    run_step "NCCL integration tests" \
        ctest --test-dir "${BUILD_DIR}" -L nccl --output-on-failure
fi

echo ""
echo "=============================================="
if [[ $FAIL -ne 0 ]]; then
    echo "  PARITY GATE: FAILED"
    echo "=============================================="
    exit 1
fi
echo "  PARITY GATE: PASSED"
echo "=============================================="
