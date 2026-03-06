# ADR-004: remote_ref<T> Prohibits Implicit Conversion

**Status:** Accepted
**Date:** 2026-01-15

## Context

In distributed computing, accessing remote data has fundamentally different
cost from accessing local data. STL-like abstractions that make remote
access look like local access (`operator*`, `operator T&`) create hidden
performance hazards. A single `for (auto& x : global_view)` loop could
silently generate one network round-trip per element.

## Decision

`remote_ref<T>` deliberately deletes all implicit conversions:
- No `operator T&()` or `operator T()` (no implicit dereference)
- No `operator*()` (no pointer-like syntax)
- No implicit construction from `T`

Access requires explicit method calls:
- `.get()` — returns `result<T>`, performs remote read if needed
- `.put(value)` — writes value, performs remote write if needed
- `.is_local()` — checks if element is on this rank (no communication)
- `.async_get()` — returns `distributed_future<T>`

## Consequences

**Positive:**
- Communication cost is syntactically visible at every access point
- Impossible to accidentally write O(n) remote accesses in a loop
- Forces developers to think about data locality
- `is_local()` check enables zero-communication fast paths

**Negative:**
- More verbose than `vec[i] = 42` for simple cases
- Cannot use STL algorithms directly on `global_view` (by design)
- Learning curve for developers accustomed to transparent distribution

**Mitigations:**
- `local_view()` provides full STL compatibility for local data
- `segmented_view()` enables efficient iteration over all local partitions
- Examples demonstrate idiomatic patterns for both local and remote access
