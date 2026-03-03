#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
# Run DTL MPI multi-rank tests at various rank counts.
#
# Usage:
#   bash scripts/run_mpi_tests.sh [build_dir] [rank... ] [-- <test-binary-args...>]
#
# Examples:
#   bash scripts/run_mpi_tests.sh
#   bash scripts/run_mpi_tests.sh build-alpha
#   bash scripts/run_mpi_tests.sh 2 4 8
#   bash scripts/run_mpi_tests.sh build-alpha 2 4 -- --gtest_filter=Distributed*
#
# If no explicit rank list is provided, the script runs tests at rank counts
# 2, 3 (odd-rank smoke), and 4. An optional higher rank count of 8 is
# attempted if the machine has enough cores.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="build"
RANKS=()
TEST_ARGS=(--gtest_brief=1)

usage() {
    cat <<'EOF'
Usage: bash scripts/run_mpi_tests.sh [build_dir] [rank... ] [-- <test-binary-args...>]

Examples:
  bash scripts/run_mpi_tests.sh
  bash scripts/run_mpi_tests.sh build-alpha
  bash scripts/run_mpi_tests.sh 2 4 8
  bash scripts/run_mpi_tests.sh build-alpha 2 4 -- --gtest_filter=Distributed*
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 && "${1:-}" != "--" && ! "${1:-}" =~ ^[0-9]+$ ]]; then
    BUILD_DIR="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            if [[ $# -gt 0 ]]; then
                TEST_ARGS=("$@")
            fi
            break
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                RANKS+=("$1")
                shift
            else
                echo "ERROR: unrecognized argument '$1'" >&2
                usage >&2
                exit 2
            fi
            ;;
    esac
done

cd "${ROOT_DIR}"

TEST_BIN="${BUILD_DIR}/tests/dtl_unit_tests"

if [ ! -x "$TEST_BIN" ]; then
    echo "ERROR: Test binary not found at ${TEST_BIN}" >&2
    echo "Build with: cmake --build ${BUILD_DIR} --target dtl_unit_tests" >&2
    exit 1
fi

read -r -a MPIRUN_CMD <<<"${MPIRUN:-mpirun}"
read -r -a MPI_FLAG_ARR <<<"${MPI_FLAGS:---oversubscribe --bind-to none}"

if [[ ${#RANKS[@]} -eq 0 ]]; then
    RANKS=(2 3 4)

    CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    if [[ "${CORES}" -ge 8 ]]; then
        RANKS+=(8)
    fi
fi

run_mpi_test() {
    local np=$1
    echo ""
    echo "=============================================="
    echo "  MPI Tests: ${np} ranks"
    echo "=============================================="
    "${MPIRUN_CMD[@]}" "${MPI_FLAG_ARR[@]}" -np "$np" "$TEST_BIN" "${TEST_ARGS[@]}"
}

for np in "${RANKS[@]}"; do
    run_mpi_test "${np}"
done

echo ""
echo "=============================================="
echo "  All MPI tests passed"
echo "=============================================="
