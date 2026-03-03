# Known Issues

**Last Updated:** 2026-02-27

## Applies To

DTL v0.1.0-alpha.1

## Known Issues

### Futures API Experimental Status

The futures-based asynchronous API (`dtl::future`, `dtl::promise`) is experimental
in v0.1.0-alpha.1. The API surface may change in minor releases. Users should pin to
specific versions if depending on async return types.

### Async Algorithm Stability Caveats

Async execution policy (`DTL_EXEC_ASYNC`) for distributed algorithms may exhibit
non-deterministic ordering of completion callbacks when used with MPI backends
under high contention. Use `DTL_EXEC_SEQ` or `DTL_EXEC_PAR` for deterministic
behavior in production workloads.

### GPU Backend Experimental Status

CUDA, HIP, and NCCL backends are experimental. While functional for common
operations (vector fill, reduce, broadcast), edge cases in multi-GPU topologies
and unified memory coherence across MPI ranks are not fully validated.

### Fortran Bindings Experimental

Fortran bindings are experimental and require `DTL_BUILD_C_BINDINGS=ON`. The
Fortran module API (`use dtl`) provides coverage (~90% of the C API)
across submodules including context, vector, array, span, tensor, communication,
collectives, algorithms, RMA, MPMD, futures, policies, and topology.
Futures and async algorithm submodules are considered experimental and
may change in minor releases.

### Python Bindings Experimental

Python bindings are experimental. The custom exception hierarchy
(`dtl.CommunicationError`, `dtl.BackendError`, etc.) is available but some
edge-case status codes may fall through to the base `dtl.DTLError`.

### Distributed Map and Remote/RPC Bindings Removed

The distributed map C/Fortran binding surface and remote/RPC C/Fortran/Python
binding surface have been removed in v0.1.0-alpha.1. The C++ core
`distributed_map` and `dtl::remote` are retained. Binding surfaces will be
re-added when the underlying implementations are production-ready.
