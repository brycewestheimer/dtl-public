# ADR-003: Per-Instance MPI_Comm_dup for Library Isolation

**Status:** Accepted
**Date:** 2026-01-15

## Context

MPI communication uses tags and communicators to distinguish messages.
Libraries that share `MPI_COMM_WORLD` can interfere with each other if
they use the same tags. Multiple DTL instances in the same process (e.g.,
different solver components) must not have their messages cross-talk.

## Decision

Each DTL `environment` instance calls `MPI_Comm_dup` to create its own
communicator. This ensures complete message isolation between:
- Multiple DTL instances in the same process
- DTL and other MPI-using libraries (PETSc, Trilinos, etc.)
- DTL and user MPI code

The duplicated communicator is freed in the environment destructor.

## Consequences

**Positive:**
- Complete message isolation between library instances
- Safe concurrent use of DTL alongside other MPI libraries
- No tag management or reservation needed
- RAII cleanup prevents communicator leaks

**Negative:**
- `MPI_Comm_dup` is a collective operation (all ranks must participate)
- Small additional memory per communicator
- Cannot use `MPI_COMM_WORLD` directly for interop (must access via `native_comm()`)

**Mitigations:**
- `from_comm()` factory allows adopting an existing communicator without duplication
- `native_comm()` exposes the underlying `MPI_Comm` for advanced interop
- Environment reference counting prevents premature finalization
