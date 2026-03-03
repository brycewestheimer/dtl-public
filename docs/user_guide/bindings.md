# Legacy Deep-Dive: Language Bindings

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [08-language-bindings-overview.md](08-language-bindings-overview.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


DTL provides language bindings beyond the native C++ API, enabling use from C, Python, Fortran, and other languages. This guide covers binding architecture, usage patterns, and language-specific considerations.

---

## Table of Contents

- [Overview](#overview)
- [C Bindings](#c-bindings)
- [Python Bindings](#python-bindings)
- [Fortran Bindings](#fortran-bindings)
- [Other Languages](#other-languages)
- [Performance Considerations](#performance-considerations)
- [References](#references)

---

## Overview

### Binding Architecture

DTL's language bindings follow a layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    User Applications                     │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│  Python  │  Fortran │   Julia  │    R     │   Other    │
├──────────┴──────────┴──────────┴──────────┴────────────┤
│                      C ABI Layer                        │
│              (libdtl_c.so / dtl_c.dll)                  │
├─────────────────────────────────────────────────────────┤
│                    C++ Core Library                     │
│                   (header-only + MPI)                   │
└─────────────────────────────────────────────────────────┘
```

**Key Design Principles:**

1. **C ABI as Universal Interface**: The C bindings provide a stable ABI that any language with C FFI can use
2. **Python with Native Feel**: The Python bindings wrap the C ABI with Pythonic idioms and NumPy integration
3. **Zero-Copy Where Possible**: Data views share memory with the underlying C++ containers
4. **Non-owning span semantics**: C/Python/Fortran expose first-class distributed span APIs that remain explicitly non-owning

### Available Bindings

| Language | Status | Interface | Documentation |
|----------|--------|-----------|---------------|
| C++ | Native | Header-only | Full API reference |
| C | Complete | libdtl_c | [C Bindings Guide](../bindings/c_bindings.md) |
| Python | Complete | dtl module | [Python Bindings Guide](../bindings/python_bindings.md) |
| Fortran | Complete | Native `dtl` module (ISO_C_BINDING over C ABI) | [Fortran Bindings Guide](../bindings/fortran_bindings.md) |

### Feature Matrix

| Feature | C++ | C | Python |
|---------|-----|---|--------|
| **Containers** | | | |
| distributed_vector | Native | `dtl_vector_*` | `DistributedVector` |
| distributed_array | Native | `dtl_array_*` | `DistributedArray` |
| distributed_span | Native (`distributed_span`) | `dtl_span_*` (`dtl_span_t`) | `DistributedSpan` |
| distributed_tensor | Native | `dtl_tensor_*` | `DistributedTensor` |
| **Collective Operations** | | | |
| allreduce | Native | `dtl_allreduce` | `dtl.allreduce()` |
| broadcast | Native | `dtl_broadcast` | `dtl.broadcast()` |
| gather/scatter | Native | `dtl_gather/scatter` | `dtl.gather/scatter()` |
| reduce | Native | `dtl_reduce` | `dtl.reduce()` |
| allgather | Native | `dtl_allgather` | `dtl.allgather()` |
| **Algorithms** | | | |
| for_each/transform | Native | `dtl_for_each/transform` | `dtl.for_each/transform()` |
| fill/copy | Native | `dtl_fill/copy` | `dtl.fill/copy()` |
| find/count | Native | `dtl_find/count` | `dtl.find/count()` |
| reduce | Native | `dtl_vector_reduce` | `dtl.reduce()` |
| sort | Native | `dtl_sort_*` | `dtl.sort()` |
| **Policies** | | | |
| Partition (block/cyclic) | Template params | `dtl_container_options` | Constructor kwargs |
| Placement (host/device) | Template params | `dtl_placement_policy` | Constructor kwargs |
| **RMA Operations** | | | |
| Window create/destroy | Native | `dtl_window_*` | `dtl.Window` |
| put/get | Native | `dtl_rma_put/get` | `dtl.rma_put/get()` |
| Atomic accumulate | Native | `dtl_rma_accumulate` | `dtl.rma_accumulate()` |
| fetch_and_op | Native | `dtl_rma_fetch_and_op` | `dtl.rma_fetch_and_add()` |
| compare_and_swap | Native | `dtl_rma_compare_and_swap` | `dtl.rma_compare_and_swap()` |
| Synchronization | Native | `dtl_window_fence/lock/unlock` | `Window.fence/lock/unlock()` |

---

## C Bindings

The C bindings provide a stable, language-agnostic interface to DTL. They are designed for:

- Integration with C codebases
- Building bindings for other languages
- Systems requiring ABI stability

### Quick Example

```c
#include <dtl/bindings/c/dtl.h>
#include <stdio.h>

int main() {
    dtl_context_t ctx;
    dtl_status status = dtl_context_create_default(&ctx);
    if (status != DTL_SUCCESS) {
        fprintf(stderr, "Error: %s\n", dtl_status_message(status));
        return 1;
    }

    printf("Rank %d of %d\n", dtl_context_rank(ctx), dtl_context_size(ctx));

    // Create distributed vector
    dtl_vector_t vec;
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 1000, &vec);
    if (status != DTL_SUCCESS) {
        dtl_context_destroy(ctx);
        return 1;
    }

    // Access local data
    double* data = dtl_vector_local_data_mut(vec);
    for (size_t i = 0; i < dtl_vector_local_size(vec); i++) {
        data[i] = (double)i;
    }

    // Barrier synchronization
    dtl_context_barrier(ctx);

    // Cleanup
    dtl_vector_destroy(vec);
    dtl_context_destroy(ctx);
    return 0;
}
```

### Building with C Bindings

```bash
# Compile
gcc -std=c99 -o my_program my_program.c -ldtl_c -lmpi

# Run with MPI
mpirun -np 4 ./my_program
```

See the [complete C bindings guide](../bindings/c_bindings.md) for detailed API reference.

---

## Python Bindings

The Python bindings provide a Pythonic interface with seamless NumPy integration.

### Installation

```bash
# From source (in DTL build directory)
cd build
make python_install

# Or using pip (when package is published)
pip install dtl
```

### Quick Example

```python
import dtl
import numpy as np

# Create context (uses MPI_COMM_WORLD by default)
with dtl.Context() as ctx:
    print(f"Rank {ctx.rank} of {ctx.size}")

    # Create distributed vector
    vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)

    # Access local data as NumPy array (zero-copy)
    local = vec.local_view()
    local[:] = np.arange(len(local)) + vec.local_offset

    # Collective operations
    local_sum = np.sum(local)
    global_sum = dtl.allreduce(ctx, local_sum, op=dtl.SUM)

    if ctx.is_root:
        print(f"Global sum: {global_sum}")
```

### Running with MPI

```bash
mpirun -np 4 python my_program.py
```

### Key Features

- **Zero-Copy NumPy Views**: `local_view()` returns a NumPy array that shares memory with DTL
- **mpi4py Interoperability**: Pass mpi4py communicators to `Context()`
- **Type Annotations**: Full PEP 484 type hints for IDE support
- **Collective Operations**: `allreduce`, `broadcast`, `gather`, `scatter`, `reduce`, `allgather`

See the [complete Python bindings guide](../bindings/python_bindings.md) for detailed documentation.

---

## Fortran Bindings

Fortran programs can use DTL through the C bindings via `ISO_C_BINDING`.

### Quick Example

```fortran
program dtl_example
    use, intrinsic :: iso_c_binding
    implicit none

    ! DTL interface declarations
    interface
        function dtl_context_create_default(ctx) bind(c, name='dtl_context_create_default')
            import :: c_ptr, c_int
            type(c_ptr), intent(out) :: ctx
            integer(c_int) :: dtl_context_create_default
        end function

        function dtl_context_rank(ctx) bind(c, name='dtl_context_rank')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_rank
        end function

        subroutine dtl_context_destroy(ctx) bind(c, name='dtl_context_destroy')
            import :: c_ptr
            type(c_ptr), value :: ctx
        end subroutine
    end interface

    type(c_ptr) :: ctx
    integer(c_int) :: status, rank

    ! Create context
    status = dtl_context_create_default(ctx)
    if (status /= 0) stop 'Failed to create context'

    rank = dtl_context_rank(ctx)
    print *, 'Rank:', rank

    call dtl_context_destroy(ctx)
end program
```

See the [complete Fortran bindings guide](../bindings/fortran_bindings.md) for module templates and advanced usage.

---

## Other Languages

Any language with C FFI support can use DTL through the C bindings:

### Julia

```julia
using Libdl

const libdtl = dlopen("libdtl_c")

dtl_context_create_default = dlsym(libdtl, :dtl_context_create_default)
dtl_context_rank = dlsym(libdtl, :dtl_context_rank)

ctx = Ref{Ptr{Nothing}}()
status = ccall(dtl_context_create_default, Cint, (Ref{Ptr{Nothing}},), ctx)
rank = ccall(dtl_context_rank, Cint, (Ptr{Nothing},), ctx[])
println("Rank: $rank")
```

### Rust

```rust
use std::ffi::c_void;

extern "C" {
    fn dtl_context_create_default(ctx: *mut *mut c_void) -> i32;
    fn dtl_context_rank(ctx: *mut c_void) -> i32;
    fn dtl_context_destroy(ctx: *mut c_void);
}

fn main() {
    let mut ctx: *mut c_void = std::ptr::null_mut();
    unsafe {
        let status = dtl_context_create_default(&mut ctx);
        if status == 0 {
            let rank = dtl_context_rank(ctx);
            println!("Rank: {}", rank);
            dtl_context_destroy(ctx);
        }
    }
}
```

---

## Performance Considerations

### Binding Overhead

| Binding | Overhead | Notes |
|---------|----------|-------|
| C++ | None | Native implementation |
| C | Minimal | Single function call indirection |
| Python | Low | pybind11 optimized; NumPy views are zero-copy |
| Fortran | Minimal | Direct C call via ISO_C_BINDING |

### Best Practices

1. **Minimize boundary crossings**: Do bulk operations rather than element-by-element access
2. **Use zero-copy views**: Python's `local_view()` shares memory with C++
3. **Prefer collective operations**: `allreduce`, `broadcast`, etc. are optimized
4. **Keep data on native side**: Avoid copying between language runtimes

### Example: Efficient Python Usage

```python
# GOOD: Bulk operation on NumPy array
local = vec.local_view()
local[:] = np.sin(local)  # Vectorized NumPy operation

# BAD: Element-by-element in Python loop
for i in range(len(local)):
    local[i] = np.sin(local[i])  # Python loop overhead
```

---

## References

- [C Bindings Guide](../bindings/c_bindings.md)
- [Python Bindings Guide](../bindings/python_bindings.md)
- [Fortran Bindings Guide](../bindings/fortran_bindings.md)
