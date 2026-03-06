#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
#
# Run DTL scaling benchmarks across multiple rank counts.
#
# Usage:
#   bash scripts/run_scaling_benchmarks.sh <build_dir> [rank_counts...]
#
# Examples:
#   bash scripts/run_scaling_benchmarks.sh build 1 2 4
#   bash scripts/run_scaling_benchmarks.sh build 1 2 4 8 16 32
#
# Output:
#   <build_dir>/scaling_results/scaling_<N>ranks.json per sweep

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <build_dir> <rank_count> [rank_count...]"
    echo "Example: $0 build 1 2 4 8"
    exit 1
fi

BUILD_DIR="$1"
shift
RANK_COUNTS=("$@")

BENCH_EXE="${BUILD_DIR}/benchmarks/scaling/bench_scaling"
if [ ! -x "$BENCH_EXE" ]; then
    echo "Error: benchmark executable not found at ${BENCH_EXE}"
    echo "Build with: cmake -DDTL_BUILD_BENCHMARKS=ON -DDTL_ENABLE_MPI=ON && cmake --build ${BUILD_DIR}"
    exit 1
fi

RESULTS_DIR="${BUILD_DIR}/scaling_results"
mkdir -p "$RESULTS_DIR"

echo "=== DTL Scaling Benchmarks ==="
echo "Executable: ${BENCH_EXE}"
echo "Rank counts: ${RANK_COUNTS[*]}"
echo "Results dir: ${RESULTS_DIR}"
echo ""

for N in "${RANK_COUNTS[@]}"; do
    OUTFILE="${RESULTS_DIR}/scaling_${N}ranks.json"
    echo "--- Running with ${N} rank(s) ---"

    if [ "$N" -eq 1 ]; then
        "${BENCH_EXE}" \
            --benchmark_out="${OUTFILE}" \
            --benchmark_out_format=json \
            --benchmark_repetitions=1
    else
        mpirun --oversubscribe -np "$N" "${BENCH_EXE}" \
            --benchmark_out="${OUTFILE}" \
            --benchmark_out_format=json \
            --benchmark_repetitions=1
    fi

    echo "  -> ${OUTFILE}"
done

echo ""
echo "=== Done. Results in ${RESULTS_DIR}/ ==="
echo "Generate report with: python3 scripts/plot_scaling.py ${RESULTS_DIR}"
