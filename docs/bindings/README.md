# DTL Language Bindings

This directory contains documentation for DTL's language bindings, enabling use of the library from C, Python, Fortran, and other languages.

## Overview

DTL provides multi-language support through a layered binding architecture:

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

The **C ABI** serves as the universal interface layer, providing a stable binary interface that any language with C FFI capabilities can use. Higher-level bindings (like Python) wrap this C layer to provide native language idioms.

## Available Bindings

| Language | Guide | Status | Build Target |
|----------|-------|--------|--------------|
| C | [c_bindings.md](c_bindings.md) | Complete (includes mode-aware NCCL `_ex` APIs) | `libdtl_c` |
| Python | [python_bindings.md](python_bindings.md) | Complete core + mode-aware NCCL context APIs | `_dtl` module |
| Fortran | [fortran_bindings.md](fortran_bindings.md) | Documented + explicit NCCL device collective C interop | Via C ABI |

## Quick Start

### C

```c
#include <dtl/bindings/c/dtl.h>

dtl_context_t ctx;
dtl_context_create_default(&ctx);
printf("Rank %d of %d\n", dtl_context_rank(ctx), dtl_context_size(ctx));
dtl_context_destroy(ctx);
```

### Python

```python
import dtl
import numpy as np

with dtl.Context() as ctx:
    vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)
    local = vec.local_view()  # Zero-copy NumPy array
    local[:] = np.arange(len(local))
```

### Fortran

```fortran
use, intrinsic :: iso_c_binding
use dtl_bindings

type(c_ptr) :: ctx
integer(c_int) :: status

status = dtl_context_create_default(ctx)
print *, 'Rank:', dtl_context_rank(ctx)
call dtl_context_destroy(ctx)
```

## Building Bindings

### C Bindings (libdtl_c)

```bash
cmake .. -DDTL_BUILD_C_BINDINGS=ON
make dtl_c
```

### Python Bindings

```bash
cmake .. -DDTL_BUILD_PYTHON=ON
make _dtl
make python_install
```

## Why C as the Base Layer?

The C ABI provides several advantages as the universal binding layer:

1. **ABI Stability**: C's ABI is standardized per platform and rarely changes
2. **Universal FFI**: Every major language has C Foreign Function Interface support
3. **No Name Mangling**: C symbols are predictable (unlike C++)
4. **Simple Memory Model**: Explicit ownership via naming conventions (`_create`/`_destroy`)
5. **Cross-Platform**: Works identically on Linux, macOS, and Windows

Languages can then build idiomatic wrappers on top of this stable foundation, as DTL does with Python bindings that add NumPy integration and Pythonic patterns.

## Related Documentation

- [User Guide: Bindings](../user_guide/bindings.md) - User-oriented binding overview
