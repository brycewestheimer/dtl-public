# 8. Language Bindings Overview

## Supported bindings

- C ABI
- Python
- Fortran (ISO C binding-based)

## Choosing a binding

- C ABI for stable interoperability with C and foreign-language bridges
- Python for interactive/data-science and orchestration workflows
- Fortran for HPC codes integrating with C ABI handles

## C ABI usage model

- explicit handle lifecycle (`create`/`destroy`)
- explicit `dtl_status` error handling
- backend capability checks and availability codes

See: `docs/bindings/c_bindings.md`

## Python usage model

- native extension + high-level Pythonic wrappers
- NumPy integration for local data views/copies
- async and callback APIs require careful lifecycle awareness

See: `docs/bindings/python_bindings.md`

## Fortran usage model

- `iso_c_binding` interop wrappers around C ABI
- handle-oriented lifecycle with explicit destroy calls
- status-code-driven failure handling

See: `docs/bindings/fortran_bindings.md`

## Cross-language parity guidance

When switching layers (C++ to C/Python/Fortran):

- verify equivalent status/error semantics
- verify ownership rules are preserved
- verify backend-unavailable behavior is consistent

For a detailed function-by-function comparison, see:

- [API Parity Matrix](../parity/api-parity-matrix.md) — full coverage table for every C API function across all bindings
- [Intentional Differences](../parity/intentional-differences.md) — rationale for deliberate cross-language divergences

## `distributed_span` mapping across bindings

All language bindings now expose first-class distributed span surfaces:

- C ABI: `dtl_span_t` (`dtl_span_from_vector/array/tensor`, subspan and local access APIs)
- Python: `dtl.DistributedSpan(...)` factory wrapping typed native span handles
- Fortran: `dtl_span_*` interfaces in `dtl.f90` (including creation, subspan, and local data access)

Across all bindings, distributed spans remain non-owning. The underlying owner (container/context/runtime handle) must outlive all spans and borrowed local views.

## Binding test expectations

- C ABI tests pass
- Fortran smoke tests pass
- Python non-MPI/non-CUDA suite passes (plus MPI/CUDA marked suites when available)

## Next step

Continue to [Chapter 9](09-error-handling-and-reliability.md).

## Deep-dive reference

- [Legacy Deep-Dive: Language Bindings](bindings.md)
- [Runtime and Handle Model](13-runtime-and-handle-model.md)
