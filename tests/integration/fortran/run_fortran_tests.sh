#!/bin/bash
# =============================================================================
# Fortran MPI Integration Test & Benchmark Runner
# =============================================================================
# Run from the build directory where test_fortran_mpi and bench_fortran_api
# are located (e.g. build/bin/tests/integration/fortran/).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPIRUN="${MPIRUN:-/usr/bin/mpirun}"
MPIRUN_FLAGS="${MPIRUN_FLAGS:---allow-run-as-root}"

echo "=== Fortran MPI Integration Tests ==="
echo ""

echo "--- Single-rank ---"
"${SCRIPT_DIR}/test_fortran_mpi"
echo ""

echo "--- Multi-rank (np=2) ---"
${MPIRUN} ${MPIRUN_FLAGS} -np 2 "${SCRIPT_DIR}/test_fortran_mpi"
echo ""

echo "--- Multi-rank (np=4) ---"
${MPIRUN} ${MPIRUN_FLAGS} -np 4 "${SCRIPT_DIR}/test_fortran_mpi"
echo ""

echo "=== Fortran API Benchmarks ==="
echo ""

echo "--- Single-rank benchmark ---"
"${SCRIPT_DIR}/bench_fortran_api" || true  # benchmark failure is non-fatal
echo ""

echo "--- Multi-rank benchmark (np=4) ---"
${MPIRUN} ${MPIRUN_FLAGS} -np 4 "${SCRIPT_DIR}/bench_fortran_api" || true  # benchmark failure is non-fatal
echo ""

echo "=== Done ==="
