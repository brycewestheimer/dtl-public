# ADR-002: C++20 Concepts over CRTP for Backend Abstraction

**Status:** Accepted
**Date:** 2026-01-15

## Context

DTL backends (MPI, CUDA, NCCL, HIP, SHMEM, shared memory) need a common
interface for communication, execution, and memory management. Traditional
C++ approaches use either virtual functions or CRTP (Curiously Recurring
Template Pattern).

## Decision

DTL uses C++20 concepts for all backend abstractions. Backend types satisfy
concepts like `Communicator`, `Executor`, `MemorySpace`, and
`RMACommunicator` via structural typing — no base classes, no vtables.

```cpp
template <typename T>
concept Communicator = requires(T& comm, rank_t dest) {
    { comm.rank() } -> std::convertible_to<rank_t>;
    { comm.size() } -> std::convertible_to<rank_t>;
    // ...
};
```

## Consequences

**Positive:**
- Zero-cost abstraction: no vtable overhead, full inlining
- Better error messages than SFINAE (concept subsumption)
- Concepts serve as executable documentation of type requirements
- New backends can be added without modifying any existing code
- `static_assert(Communicator<my_backend>)` provides instant conformance checking

**Negative:**
- Requires C++20 (limits compiler support to GCC 10+, Clang 12+, MSVC 19.29+)
- Cannot store heterogeneous backends in a single collection (no runtime polymorphism)
- Type-erased handle layer needed for C API / runtime dispatch

**Mitigations:**
- `handle.hpp` provides type-erased wrappers for the C API layer
- Runtime dispatch is isolated to the `runtime/` module
- C++20 requirement is acceptable for the HPC target audience
