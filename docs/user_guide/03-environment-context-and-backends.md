# 3. Environment, Context, and Backends

## Lifecycle model

DTL runtime usage generally follows:

1. initialize/create environment
2. create context(s)
3. run container/view/algorithm workloads
4. finalize/cleanup

## Environment responsibilities

The environment tracks backend capabilities and runtime lifecycle. It is the point where backend availability (MPI/GPU/etc.) becomes concrete.

Typical capabilities queried from environment:

- `has_mpi()`
- `has_cuda()`
- `has_hip()`
- `has_nccl()`
- `has_shmem()`

## Context responsibilities

A context encapsulates active domains and communication identity for operations. Use context as the explicit dependency for distributed containers and collectives.

Typical context queries:

- rank and size
- root status
- device affinity / device id
- validity checks

## Backend-aware programming guidance

- Branch behavior on capability checks before selecting backend-specific paths.
- Keep host fallback paths for environments where GPU backends are unavailable.
- Treat MPI participation as collective contract, not optional within a call path.

## Single-rank and non-MPI mode

DTL supports non-MPI execution for local development and many workflows:

- rank is effectively 0
- size is effectively 1
- collective APIs generally degenerate to local behavior

This mode is useful for unit tests and local correctness development.

## MPI/GPU domain composition

Advanced flows may compose contexts with additional domains (e.g., CUDA/NCCL). Ensure availability checks and explicit error handling around domain-adding operations.

## Operational best practices

1. create context once per logical execution scope
2. avoid repeatedly creating/destroying context in hot loops
3. keep context validity checks at API boundaries in mixed-language stacks

## Next step

Continue with [Chapter 4](04-distributed-containers.md) for core data model usage.

## Deep-dive reference

- [Legacy Deep-Dive: Environment](environment.md)
- [Runtime and Handle Model](13-runtime-and-handle-model.md)
