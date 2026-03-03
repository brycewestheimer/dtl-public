# DTL Intentional Cross-Language Differences

This document explains deliberate divergences between the C++, C, Python, and
Fortran bindings. These are not gaps to be closed but design decisions driven by
language idioms and user ergonomics.

---

## Error Handling

| Language | Mechanism | Rationale |
|----------|-----------|-----------|
| C++ | `result<T>` monadic type | Zero-overhead, composable, no exceptions |
| C | `dtl_status` integer codes | ABI-stable, no C++ runtime dependency |
| Python | `DTLError` exception hierarchy | Pythonic; integrates with `try/except` |
| Fortran | `integer(c_int)` status codes | No exception support in Fortran; matches MPI convention |

All languages map to the same underlying status code taxonomy (categories,
specific codes). The representation varies but the semantics are identical.

## Generics / Type Dispatch

| Language | Mechanism | Rationale |
|----------|-----------|-----------|
| C++ | Templates (`distributed_vector<T>`) | Zero-cost abstraction, compile-time dispatch |
| C | `void*` + `dtl_dtype` enum | Type erasure for ABI stability |
| Python | Factory functions + NumPy dtype | Duck typing; dtype inferred from NumPy array |
| Fortran | Type-specific convenience wrappers | No generics; `dtl_broadcast_double()` avoids `c_loc` boilerplate |

The Fortran helpers in `dtl_helpers.f90` wrap the generic C functions with
typed alternatives (e.g., `dtl_allreduce_sum_double`) to avoid forcing users to
construct `c_ptr` / `c_loc` calls for every operation.

## Resource Lifecycle (RAII vs Manual)

| Language | Mechanism | Rationale |
|----------|-----------|-----------|
| C++ | RAII constructors/destructors | Automatic cleanup, exception-safe |
| C | `dtl_*_create()` / `dtl_*_destroy()` | Explicit lifecycle, no C++ runtime |
| Python | Context managers + garbage collector | `with` blocks for deterministic cleanup; GC as safety net |
| Fortran | `dtl_*_create()` / `dtl_*_destroy()` | Mirrors C pattern; Fortran lacks destructors for `bind(c)` types |

Python objects release their C handles in `__del__` (or `__exit__` if used as
context managers). Fortran users must call destroy explicitly.

## Callback / Closure Support

| Language | Mechanism | Rationale |
|----------|-----------|-----------|
| C++ | Lambdas, `std::function` | First-class closures with state |
| C | Function pointer + `void* user_data` | Classic C callback pattern |
| Python | Python callables (GIL-aware) | Native closures; GIL released during C calls, reacquired in callback |
| Fortran | `abstract interface` + `c_funloc` | ISO_C_BINDING function pointer interop |

Fortran callbacks must be `bind(c)` procedures. The `user_data` pointer
(`type(c_ptr)`) allows passing Fortran-side state to callbacks.

## Reduce Operations

| Language | Representation | Rationale |
|----------|---------------|-----------|
| C++ | Enum class `reduce_op` | Type-safe, compile-time checked |
| C | Integer enum `dtl_reduce_op` | ABI-stable integers |
| Python | String (`"sum"`, `"max"`, `"min"`, `"prod"`) | Pythonic; readable in scripts and notebooks |
| Fortran | Integer constants (`DTL_OP_SUM`, etc.) | Matches MPI Fortran conventions |

Python internally maps strings to the C enum values. Module-level constants
(`dtl.SUM`, `dtl.MAX`) are also available for programmatic use.

## Data Views

| Language | Mechanism | Rationale |
|----------|-----------|-----------|
| C++ | `local_view<T>`, `global_view` | Type-safe span with bounds checking |
| C | Raw `void*` from `dtl_*_local_data[_mut]()` | Minimal abstraction, user casts |
| Python | `local_view()` returns zero-copy NumPy array; `to_numpy()` returns copy | NumPy integration; choice of view vs copy |
| Fortran | `c_f_pointer()` on `c_ptr` from local_data | Standard Fortran interop pattern |

Python's `local_view()` shares memory with the container (modifications visible
immediately). `to_numpy()` returns an independent copy, useful when the tensor
may be redistributed or when the user needs a contiguous host-side buffer.

## Policy Expression

| Language | Mechanism | Rationale |
|----------|-----------|-----------|
| C++ | Tag types (`dtl::block{}`, `dtl::par{}`) | Compile-time dispatch, zero overhead |
| C | Integer enums (`DTL_PARTITION_BLOCK`) | ABI-stable |
| Python | Integer constants + enum classes | `dtl.PARTITION_BLOCK` or `dtl.PartitionPolicy.BLOCK` |
| Fortran | Integer constants (`DTL_PARTITION_BLOCK`) | Same values as C; passed directly to C functions |

All languages share the same underlying integer values for policy enums.

## Non-Blocking Communication

| Language | Available | Rationale |
|----------|-----------|-----------|
| C++ | Full (isend, irecv, wait, test) | Native async support |
| C | Full | Complete ABI |
| Python | Blocking only (send, recv, sendrecv) | GIL complexity; users can use mpi4py for async patterns |
| Fortran | Full (isend, irecv, wait, waitall, test) | Matches MPI Fortran idiom |

Python omits `isend`/`irecv` because the GIL interaction with non-blocking MPI
creates subtle correctness issues. For advanced async patterns, Python users
should use mpi4py directly alongside DTL contexts.

## Scan/Exscan Operations

| Language | Available | Rationale |
|----------|-----------|-----------|
| C++ | Full | Native support |
| C | Full | Complete ABI |
| Python | Not exposed | Low demand in Python HPC workflows; users can use mpi4py |
| Fortran | Full | Matches MPI Fortran patterns |

## Window Passive-Target Synchronization

| Language | Available | Rationale |
|----------|-----------|-----------|
| C++ | Full (lock, unlock, flush variants) | Native RMA support |
| C | Full | Complete ABI |
| Python | Fence-based only | Passive-target RMA is rarely used from Python |
| Fortran | Full | Matches MPI-3 RMA Fortran patterns |

## String Handling

| Language | Approach | Rationale |
|----------|----------|-----------|
| C++ | `std::string` / `std::string_view` | Native C++ strings |
| C | `const char*` (null-terminated) | Standard C strings |
| Python | Python `str` (automatic conversion) | pybind11 handles encoding |
| Fortran | `character(kind=c_char)` + `dtl_c_to_f_string()` helper | Fortran fixed-length strings need explicit conversion from C |

The `dtl_c_to_f_string()` helper in `dtl_helpers.f90` converts C null-terminated
strings to Fortran allocatable strings, handling the length calculation
automatically.

---

## Functions Intentionally Omitted

### Fortran

| C API Function | Reason |
|----------------|--------|
| `dtl_vector_copy_to_host` / `_from_host` | Fortran users access data via `c_f_pointer` on `local_data_mut`; no device-host copy abstraction needed in CPU-only Fortran codes |
| `dtl_context_set_error_handler` | Fortran error handling uses status codes; callback-based error handlers would require complex interop |

### Python

| C API Function | Reason |
|----------------|--------|
| `dtl_isend` / `dtl_irecv` / `dtl_wait` / `dtl_waitall` / `dtl_test` | GIL complexity; use mpi4py for non-blocking patterns |
| `dtl_alltoall` | Rarely needed; use `alltoallv` for variable-count |
| `dtl_scan` / `dtl_exscan` | Low demand; use mpi4py or NumPy cumsum |
| `dtl_context_determinism_mode` and policy queries | Internal scheduling; not useful from Python |
| `dtl_environment_create_with_args` | Python contexts handle initialization transparently |
| Policy name getters | Python has `str()` on enum values |
| Container policy query functions (execution, consistency, error) | Exposed via properties on container objects instead |
