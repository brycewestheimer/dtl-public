# 2. Installation and Build Workflows

## Build prerequisites

Minimum baseline:

- CMake 3.20+
- C++20 compiler

Optional dependencies by feature:

- MPI implementation for multi-rank execution
- CUDA/HIP/NCCL for GPU-accelerated paths
- Python and pybind11 for Python bindings
- Fortran compiler for Fortran module

## Standard CMake configuration

From repo root:

```bash
cmake -S . -B build \
  -DDTL_ENABLE_MPI=OFF \
  -DDTL_BUILD_TESTS=ON
cmake --build build -j6
```

## Common build profiles

### Core C++ usage

```bash
cmake -S . -B build-core \
  -DDTL_ENABLE_MPI=OFF \
  -DDTL_BUILD_TESTS=ON
cmake --build build-core -j6
```

### MPI-enabled profile

```bash
cmake -S . -B build-mpi \
  -DDTL_ENABLE_MPI=ON \
  -DDTL_BUILD_TESTS=ON
cmake --build build-mpi -j6
```

### Full binding profile (C + Python + Fortran)

```bash
cmake -S . -B build-bindings \
  -DDTL_ENABLE_MPI=OFF \
  -DDTL_BUILD_C_BINDINGS=ON \
  -DDTL_BUILD_PYTHON=ON \
  -DDTL_BUILD_FORTRAN=ON \
  -DDTL_BUILD_TESTS=ON
cmake --build build-bindings -j6
```

## Feature flags to know

- `DTL_ENABLE_MPI`
- `DTL_ENABLE_CUDA`
- `DTL_ENABLE_HIP`
- `DTL_ENABLE_NCCL`
- `DTL_BUILD_C_BINDINGS`
- `DTL_BUILD_PYTHON`
- `DTL_BUILD_FORTRAN`
- `DTL_BUILD_DOCS`

## Installation and packaging notes

- use CMake install/export if packaging into other projects
- use `spack repo add ./spack` to expose the bundled Spack package
- align runtime/backend flags with deployment environment
- avoid enabling optional backends you cannot provision at runtime

## Verification checklist

1. configure succeeds with expected backend summary
2. build succeeds with `-j6`
3. run at least smoke tests before adopting the build output

## Next step

Proceed to [Chapter 3](03-environment-context-and-backends.md) for runtime initialization and backend handling.
