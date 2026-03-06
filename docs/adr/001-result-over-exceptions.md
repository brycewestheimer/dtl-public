# ADR-001: result<T> over Exceptions for Error Handling

**Status:** Accepted
**Date:** 2026-01-15

## Context

DTL must interoperate with MPI, which does not support C++ exceptions. MPI
error handlers that throw will cause `MPI_Abort` in most implementations.
Additionally, GPU backends (CUDA, HIP) cannot propagate exceptions from
device code. The library needs an error handling strategy that works across
all backends.

## Decision

DTL uses `result<T>` (similar to `std::expected<T, status>`) as the primary
error reporting mechanism throughout the library. No DTL internal function
throws an exception.

Users who prefer exceptions can opt in via `throwing_policy`, which wraps
`result<T>` at the API boundary.

The error policy is configurable per-container:
- `expected_policy` (default) — returns `result<T>`
- `throwing_policy` — throws on error
- `terminating_policy` — calls `std::terminate`
- `callback_policy<Func>` — invokes user callback

## Consequences

**Positive:**
- Safe interoperability with MPI and GPU backends
- Explicit error handling prevents silent failures
- Monadic API (`and_then`, `or_else`, `transform`) enables composable error handling
- Users who want exceptions can opt in without affecting others

**Negative:**
- More verbose than try/catch for simple cases
- Requires `[[nodiscard]]` discipline to prevent ignored errors
- Two code paths in examples (result-based and exception-based)

**Mitigations:**
- `[[nodiscard]]` applied to all result-returning functions
- Helper macros available for concise error propagation
- Examples show both idioms
