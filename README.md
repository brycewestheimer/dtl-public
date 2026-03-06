# Distributed Template Library (DTL)

[![CI](https://github.com/brycewestheimer/dtl-public/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/brycewestheimer/dtl-public/actions/workflows/ci.yml)
[![MPI Tests](https://github.com/brycewestheimer/dtl-public/actions/workflows/mpi-tests.yml/badge.svg?branch=main)](https://github.com/brycewestheimer/dtl-public/actions/workflows/mpi-tests.yml)
[![Format Check](https://github.com/brycewestheimer/dtl-public/actions/workflows/format-check.yml/badge.svg?branch=main)](https://github.com/brycewestheimer/dtl-public/actions/workflows/format-check.yml)
[![Coverage](https://github.com/brycewestheimer/dtl-public/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/brycewestheimer/dtl-public/actions/workflows/coverage.yml)

DTL is a C++20 distributed-computing library with STL-inspired container and algorithm interfaces, explicit data distribution semantics, and policy-based execution control.

Version: **0.1.0-alpha.1**

## Table of Contents

- [What DTL Is](#what-dtl-is)
- [Release Note](#release-note)
- [Current Support Status](#current-support-status)
- [Requirements](#requirements)
- [Build and Install](#build-and-install)
- [Quick Start](#quick-start)
- [Language Bindings](#language-bindings)
- [Documentation](#documentation)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [How to Cite](#how-to-cite)
- [License](#license)

## What DTL Is

DTL is designed for developers who want:

- Familiar local semantics for distributed containers (`local_view()` as an STL-compatible contiguous view)
- Explicit global/remote semantics where communication cost matters
- A context-driven model for single-rank and multi-rank execution
- A policy system for partitioning, placement, execution, consistency, and error handling

Core modules include:

- Distributed containers (`distributed_vector`, `distributed_array`, `distributed_tensor`, `distributed_map`, `distributed_span`)
- Views (`local_view`, `global_view`, `segmented_view`, `remote_ref`)
- Algorithms and collectives
- Backend abstraction (CPU, MPI, optional accelerator backends)

## Release Note

This is the `v0.1.0-alpha.1` alpha release of the Distributed Template Library.
It is intended for early adopters evaluating API shape, build paths, and core
distributed container semantics. Experimental backends and language bindings
should be treated as pre-release surfaces.

### Alpha Preview Notice

This release is an **API preview** — not a production-ready package. Its primary
purpose is to expose the API shape, core container semantics, and basic usage
patterns for early feedback. The complete DTL package, including full backend
coverage, production testing, and performance validation, will be available in a
future public release under the MIT license. This alpha release is licensed under
BSD-3-Clause (see `LICENSE`).

## Current Support Status

### Backends

| Backend | Status |
|---|---|
| CPU | Supported |
| MPI | Supported |
| CUDA | Experimental |
| HIP | Experimental |
| NCCL | Experimental |
| SHMEM/OpenSHMEM | Experimental |

### Bindings

| Binding | Status |
|---|---|
| C++ | Supported |
| C ABI | Supported |
| Python | Experimental |
| Fortran | Experimental |

## Requirements

- CMake 3.20+
- C++20 compiler
  - GCC 11+
  - Clang 15+
  - MSVC 19.29+
- Optional: MPI implementation (OpenMPI/MPICH) for multi-rank execution
- Optional: CUDA/HIP/NCCL/OpenSHMEM toolchains for experimental backend paths

## Build and Install

### Configure and Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
```

### Install

```bash
cmake --install build
```

### Common Build Options

| Option | Default | Purpose |
|---|---|---|
| `DTL_ENABLE_MPI` | `ON` | Enable MPI backend support |
| `DTL_ENABLE_CUDA` | `OFF` | Enable CUDA backend support |
| `DTL_ENABLE_HIP` | `OFF` | Enable HIP backend support |
| `DTL_ENABLE_NCCL` | `OFF` | Enable NCCL support |
| `DTL_ENABLE_SHMEM` | `OFF` | Enable OpenSHMEM support |
| `DTL_BUILD_TESTS` | `ON` | Build unit tests |
| `DTL_BUILD_INTEGRATION_TESTS` | `OFF` | Build integration tests |
| `DTL_BUILD_EXAMPLES` | `ON` | Build examples |
| `DTL_BUILD_BENCHMARKS` | `OFF` | Build benchmarks |
| `DTL_BUILD_DOCS` | `OFF` | Build docs targets |
| `DTL_BUILD_C_BINDINGS` | `OFF` | Build C ABI library |
| `DTL_BUILD_PYTHON` | `OFF` | Build Python bindings |
| `DTL_BUILD_FORTRAN` | `OFF` | Build Fortran bindings |

When `DTL_BUILD_FORTRAN=ON`, the supported Fortran examples are built from
`bindings/fortran/examples`. The legacy `examples/fortran` sources are not part
of the default top-level example build.

### CMake Consumer Integration

After installation:

```cmake
find_package(DTL REQUIRED)
target_link_libraries(my_app PRIVATE DTL::dtl)
```

Or vendored:

```cmake
add_subdirectory(path/to/dtl)
target_link_libraries(my_app PRIVATE DTL::dtl)
```

### Spack

DTL includes a local Spack repository under [`spack/`](spack/README.md).

```bash
spack repo add ./spack
spack install dtl
spack install dtl +tests
spack install dtl +python +c_bindings
spack install dtl +docs
```

## Quick Start

These examples assume multi-rank execution (for example, `mpirun -np 4 ...`) and use
the same flow: `environment -> context -> distributed_vector`.

### C++

```cpp
#include <dtl/dtl.hpp>

#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    dtl::distributed_vector<double> vec(1024, ctx);

    // Local view: local partition only (no communication)
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), static_cast<double>(vec.global_offset()));

    // Global view: explicit remote_ref access
    auto global = vec.global_view();
    constexpr dtl::index_t probe_idx = 17;
    auto ref = global[probe_idx];  // remote_ref<double>
    double probe_local = 0.0;
    if (ref.is_local()) {
        auto v = ref.get();
        if (v) {
            probe_local = v.value();
        }
    }
    double probe_value = comm.allreduce_sum_value(probe_local);

    // Local + collective reduction
    double local_sum = dtl::local_reduce(vec, 0.0, std::plus<>{});
    double global_sum = comm.allreduce_sum_value(local_sum);

    if (ctx.rank() == 0) {
        std::cout << "global[" << probe_idx << "] = " << probe_value << "\n";
        std::cout << "global sum = " << global_sum << "\n";
    }
    return 0;
}
```

### C

```c
#include <dtl/bindings/c/dtl.h>

#include <stdio.h>

#define CHECK_DTL(call) do { \
    dtl_status _s = (call); \
    if (!dtl_status_ok(_s)) { \
        fprintf(stderr, "%s\n", dtl_status_message(_s)); \
        return 1; \
    } \
} while (0)

int main(int argc, char** argv) {
    dtl_environment_t env = NULL;
    dtl_context_t ctx = NULL;
    dtl_vector_t vec = NULL;

    CHECK_DTL(dtl_environment_create_with_args(&env, &argc, &argv));
    CHECK_DTL(dtl_environment_make_world_context(env, &ctx));
    CHECK_DTL(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 1024, &vec));

    // Local view equivalent
    double* local = (double*)dtl_vector_local_data_mut(vec);
    dtl_size_t local_size = dtl_vector_local_size(vec);
    dtl_index_t local_offset = dtl_vector_local_offset(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        local[i] = (double)(local_offset + (dtl_index_t)i);
    }

    // Global-index access equivalent to C++ global_view/remote_ref
    dtl_index_t probe_idx = 17;
    double probe_local = 0.0;
    if (dtl_vector_is_local(vec, probe_idx)) {
        dtl_index_t local_idx = dtl_vector_to_local(vec, probe_idx);
        CHECK_DTL(dtl_vector_get_local(vec, (dtl_size_t)local_idx, &probe_local));
    }
    double probe_value = 0.0;
    CHECK_DTL(dtl_allreduce(ctx, &probe_local, &probe_value, 1, DTL_DTYPE_FLOAT64, DTL_OP_SUM));

    // Local + collective reduction
    double local_sum = 0.0;
    for (dtl_size_t i = 0; i < local_size; ++i) {
        local_sum += local[i];
    }
    double global_sum = 0.0;
    CHECK_DTL(dtl_allreduce(ctx, &local_sum, &global_sum, 1, DTL_DTYPE_FLOAT64, DTL_OP_SUM));

    if (dtl_context_rank(ctx) == 0) {
        printf("global[%ld] = %.1f\n", (long)probe_idx, probe_value);
        printf("global sum = %.1f\n", global_sum);
    }

    dtl_vector_destroy(vec);
    dtl_context_destroy(ctx);
    dtl_environment_destroy(env);
    return 0;
}
```

### Fortran

```fortran
program quickstart_fortran
    use, intrinsic :: iso_c_binding
    use dtl
    implicit none

    type(c_ptr) :: ctx, vec, data_ptr
    real(c_double), pointer :: local(:)
    integer(c_int64_t) :: local_size, local_offset, i
    integer(c_int64_t), parameter :: probe_idx = 17_c_int64_t
    integer(c_int) :: status
    real(c_double), target :: probe_local, probe_value, local_sum, global_sum

    ! Fortran bindings currently create a context directly.
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) stop 'dtl_context_create_default failed'
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 1024_c_int64_t, vec)
    if (status /= DTL_SUCCESS) stop 'dtl_vector_create failed'

    ! Local view equivalent
    local_size = dtl_vector_local_size(vec)
    local_offset = dtl_vector_local_offset(vec)
    data_ptr = dtl_vector_local_data_mut(vec)
    call c_f_pointer(data_ptr, local, [local_size])
    do i = 1, local_size
        local(i) = real(local_offset + i - 1, c_double)
    end do

    ! Global-index access equivalent to C++ global_view/remote_ref
    probe_local = 0.0_c_double
    if (probe_idx >= local_offset .and. probe_idx < local_offset + local_size) then
        probe_local = local(probe_idx - local_offset + 1)
    end if
    status = dtl_allreduce(ctx, c_loc(probe_local), c_loc(probe_value), 1_c_int64_t, &
                           DTL_DTYPE_FLOAT64, DTL_OP_SUM)
    if (status /= DTL_SUCCESS) stop 'dtl_allreduce probe failed'

    ! Local + collective reduction
    local_sum = sum(local)
    status = dtl_allreduce(ctx, c_loc(local_sum), c_loc(global_sum), 1_c_int64_t, &
                           DTL_DTYPE_FLOAT64, DTL_OP_SUM)
    if (status /= DTL_SUCCESS) stop 'dtl_allreduce sum failed'

    if (dtl_context_rank(ctx) == 0) then
        print '(A,I0,A,F8.1)', 'global[', probe_idx, '] = ', probe_value
        print '(A,F12.1)', 'global sum = ', global_sum
    end if

    call dtl_vector_destroy(vec)
    call dtl_context_destroy(ctx)
end program quickstart_fortran
```

### Python

```python
import numpy as np
import dtl

with dtl.Environment() as env:
    ctx = env.make_world_context()
    vec = dtl.DistributedVector(ctx, size=1024, dtype=np.float64)

    # Local view: rank-local NumPy view (zero-copy)
    local = vec.local_view()
    local[:] = np.arange(vec.local_offset, vec.local_offset + vec.local_size, dtype=np.float64)

    # Global-index access equivalent (Python bindings do not expose C++ remote_ref)
    probe_idx = 17
    if vec.local_offset <= probe_idx < (vec.local_offset + vec.local_size):
        probe_local = float(local[probe_idx - vec.local_offset])
    else:
        probe_local = 0.0
    probe_value = dtl.allreduce(ctx, probe_local, op=dtl.SUM)

    # Local + collective reduction
    local_sum = float(np.sum(local))
    global_sum = dtl.allreduce(ctx, local_sum, op=dtl.SUM)

    if ctx.rank == 0:
        print(f"global[{probe_idx}] = {probe_value}")
        print(f"global sum = {global_sum}")
```

See `examples/basics/`, `examples/c/`, `examples/fortran/`, and `examples/python/` for
buildable end-to-end programs.

## Language Bindings

- C bindings: enable with `-DDTL_BUILD_C_BINDINGS=ON`
- Python bindings: enable with `-DDTL_BUILD_PYTHON=ON`
- Fortran bindings: enable with `-DDTL_BUILD_FORTRAN=ON`

Examples:

- C examples: `examples/c/`
- Python examples: `examples/python/`
- Fortran examples: `examples/fortran/`

## Documentation

Primary documentation lives in `docs/`.

- **C++ API Quick Reference** (no build needed): [`docs/api_reference/cpp_quick_reference.md`](docs/api_reference/cpp_quick_reference.md)
- Docs home: `docs/index.md`
- Getting started: `docs/getting_started.md`
- User guide: `docs/user_guide/`
- API reference: `docs/api_reference/`

Build full docs locally (Doxygen + Sphinx + Breathe + RTD theme):

```bash
bash scripts/generate_docs.sh build-docs -- -j6
```

## Testing

Build tests (default):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
```

Run tests:

```bash
bash scripts/run_tests.sh build -- -j6
bash scripts/run_mpi_tests.sh build 2 4
```

Integration tests are opt-in via `-DDTL_BUILD_INTEGRATION_TESTS=ON`.

CI coverage is produced by the `Coverage` GitHub Actions workflow (`.github/workflows/coverage.yml`).
MPI CI jobs use runner-constrained, single-node oversubscribed ranks (default 2) to remain compatible with standard GitHub-hosted runners.

## Known Limitations

Current C ABI support includes explicit stub behavior for selected multi-rank paths
(`dtl_remote_invoke*` cross-rank and `dtl_vector_redistribute` in multi-rank mode).

## Contributing

See `CONTRIBUTING.md` for workflow, coding standards, and review requirements.

## How to Cite

Citation metadata is maintained in `CITATION.cff`.

Until the manuscript is published, cite the software record in `CITATION.cff`:

```text
Westheimer, B. M. (2026). Distributed Template Library (DTL) (Version 0.1.0-alpha.1) [Computer software].
https://github.com/brycewestheimer/dtl-public
```

## License

DTL is licensed under BSD-3-Clause.

See `LICENSE` for full text.
