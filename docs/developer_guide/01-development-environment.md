# 1. Development Environment

## Prerequisites

Required baseline:

- CMake 3.20+
- C++20-capable compiler
- Python 3.8+

Optional components by feature area:

- MPI for multi-rank execution
- CUDA/HIP for GPU backends
- Fortran compiler for Fortran bindings

## Repository setup

Clone and configure from repo root:

```bash
git clone <repo-url>
cd dtl
cmake -S . -B build-dev -DDTL_ENABLE_MPI=OFF -DDTL_BUILD_TESTS=ON
cmake --build build-dev -j6
```

## Local build profiles

Core C++ only:

```bash
cmake -S . -B build-core \
  -DDTL_ENABLE_MPI=OFF \
  -DDTL_BUILD_TESTS=ON
cmake --build build-core -j6
```

C + Python + Fortran bindings:

```bash
cmake -S . -B build-bindings \
  -DDTL_ENABLE_MPI=OFF \
  -DDTL_BUILD_C_BINDINGS=ON \
  -DDTL_BUILD_PYTHON=ON \
  -DDTL_BUILD_FORTRAN=ON \
  -DDTL_BUILD_TESTS=ON
cmake --build build-bindings -j6
```

Docs-enabled profile:

```bash
cmake -S . -B build-docs \
  -DDTL_ENABLE_MPI=OFF \
  -DDTL_BUILD_DOCS=ON
cmake --build build-docs --target docs -j6
```

## Build safety policy

Do not exceed `-j6` for any build or test command in this environment.

## Recommended day-to-day loop

1. Configure once per profile.
2. Build incrementally with `-j6`.
3. Run focused tests first, then broader sweeps.
4. Regenerate docs when touching public APIs.
