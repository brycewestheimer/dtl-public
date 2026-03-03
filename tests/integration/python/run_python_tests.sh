#!/bin/bash
# Run DTL Python integration tests and benchmarks with MPI
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "  DTL Python MPI Integration Tests"
echo "=============================================="

echo ""
echo "--- Single-rank ---"
python3 -m pytest "$SCRIPT_DIR/test_python_mpi_integration.py" -v --tb=short 2>&1

echo ""
echo "--- Multi-rank (np=2) ---"
mpirun --allow-run-as-root --oversubscribe -np 2 \
    python3 -m pytest "$SCRIPT_DIR/test_python_mpi_integration.py" -v --tb=short 2>&1

echo ""
echo "--- Multi-rank (np=4) ---"
mpirun --allow-run-as-root --oversubscribe -np 4 \
    python3 -m pytest "$SCRIPT_DIR/test_python_mpi_integration.py" -v --tb=short 2>&1

echo ""
echo "=============================================="
echo "  DTL Python API Benchmarks"
echo "=============================================="

echo ""
echo "--- Single-rank ---"
python3 "$SCRIPT_DIR/bench_python_api.py" 2>&1

echo ""
echo "--- Multi-rank (np=4) ---"
mpirun --allow-run-as-root --oversubscribe -np 4 \
    python3 "$SCRIPT_DIR/bench_python_api.py" 2>&1

echo ""
echo "=============================================="
echo "  Done"
echo "=============================================="
