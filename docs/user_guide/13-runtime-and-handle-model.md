# 13. Runtime and Handle Model

## Why this chapter exists

DTL relies on explicit runtime capability checks and explicit handle lifetimes across C++, C ABI, Python, and Fortran. Understanding this model prevents most integration and lifecycle bugs.

## Runtime model

DTL runtime state includes backend availability and initialized domains.

Typical runtime capabilities:

- MPI availability and thread level
- CUDA/HIP/NCCL/SHMEM availability
- context/domain support for operations

Use runtime and environment queries early at startup to choose safe execution paths.

## Handle model overview

A handle represents an externally visible, lifetime-managed runtime object.

Common handle classes by API layer:

- C++: RAII objects and internal resource handles
- C ABI: opaque pointers (context, container, request, action, window, etc.)
- Python: wrapper objects over native handles
- Fortran: `type(c_ptr)` wrappers around C handles
- Non-owning views (for example `distributed_span`, NumPy local views, Fortran `c_f_pointer` projections) borrow from owning handles and do not extend owner lifetime

## C ABI handle contract

1. Create/init function returns valid handle on success.
2. Callers must destroy/free when done.
3. Invalid/null handles return deterministic error codes.
4. Backend-unavailable paths return explicit availability statuses.

## Python and Fortran handle behavior

### Python

- wrappers own native handles
- async/request handles must be explicitly waited/tested/destroyed (or deterministically cleaned by wrapper)
- object lifetime must outlive active operations using native resources

### Fortran

- handle ownership remains in C ABI
- Fortran callers pass and retain `c_ptr` handles
- explicit destroy calls are required for created resources

## Lifetime and ordering rules

- do not destroy context/window/communicator while requests using them are active
- avoid creating hidden shared ownership cycles
- ensure cleanup on all failure/early-return paths
- do not keep non-owning span/view projections alive across owner destruction or structural owner mutations

## Practical checklist

Before integrating runtime-dependent code:

- [ ] capability checks gate backend-specific code paths
- [ ] handle create/destroy pairs are explicit and tested
- [ ] async/request lifetimes are bounded and observable
- [ ] errors distinguish invalid handle vs unavailable backend vs operation failure

## Related references

- [Environment and backend chapter](03-environment-context-and-backends.md)
- [Bindings chapter](08-language-bindings-overview.md)
- [Error handling chapter](09-error-handling-and-reliability.md)
- [Troubleshooting chapter](11-troubleshooting-and-diagnostics.md)
