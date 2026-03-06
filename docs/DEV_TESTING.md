# Developer Testing Guide

Quick-reference commands for building and running DTL tests.

## Build Presets

```bash
cmake --preset dev              # Debug, MPI + tests + examples
cmake --preset dev-no-mpi       # Debug, no MPI
cmake --preset ci               # Warnings-as-errors
cmake --preset asan             # Address sanitizer
cmake --preset tsan             # Thread sanitizer
cmake --preset bindings         # C/Python/Fortran bindings
```

## Unit Tests

```bash
# Full suite
cmake --build --preset dev && ctest --preset dev

# Single test by CTest regex
ctest --test-dir build/dev -R DistributedVector --output-on-failure

# Single test by GTest filter
./build/dev/tests/dtl_unit_tests --gtest_filter='DistributedVector*'
```

## C Binding Tests

```bash
# Single-rank
./build/dev/tests/bindings/c/test_c_bindings

# Full regression (single-rank + 2-rank MPI + RMA)
bash scripts/run_binding_regression.sh build/dev
```

## MPI Tests

```bash
# Default rank counts (2, 3, 4)
bash scripts/run_mpi_tests.sh build/dev

# Specific ranks with GTest filter
bash scripts/run_mpi_tests.sh build/dev 2 4 -- --gtest_filter=Distributed*

# Direct mpirun
mpirun --oversubscribe --bind-to none -np 2 ./build/dev/tests/dtl_unit_tests
```

## CUDA Tests

```bash
# Build with CUDA preset
cmake --preset dev-cuda
cmake --build --preset dev-cuda

# Run CUDA-specific tests
ctest --test-dir build/dev-cuda -L cuda --output-on-failure

# CUDA placement filter
./build/dev-cuda/tests/bindings/c/test_c_bindings --gtest_filter='*Device*:*Cuda*:*CUDA*:*Unified*'
```

## NCCL Tests

```bash
# NCCL integration tests (requires CUDA + NCCL + MPI)
ctest --test-dir build/dev-cuda -L nccl --output-on-failure

# Direct execution
mpirun --oversubscribe --bind-to none -np 2 ./build/dev-cuda/tests/dtl_mpi_tests --gtest_filter='*NCCL*:*Nccl*'
```

## Parity Gate (Full Suite)

```bash
bash scripts/run_parity_gate.sh build/dev-cuda
```
