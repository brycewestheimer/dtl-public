# User Guide

This handbook is the primary end-user guide for DTL. It is organized as a chaptered path from first use to production-scale operation.

## Who should read this

- C++ developers adopting DTL for distributed and/or heterogeneous workloads
- teams integrating DTL into MPI/GPU pipelines
- users consuming C, Python, or Fortran bindings

## Recommended reading order

1. Read chapters 1-3 to establish runtime and data model foundations.
2. Read chapters 4-7 to learn core usage patterns.
3. Read chapters 8-12 as needed for bindings, reliability, performance, and operations.

```{toctree}
:maxdepth: 2

01-mental-model-and-core-concepts
02-installation-and-build-workflows
03-environment-context-and-backends
04-distributed-containers
05-views-iteration-and-data-access
06-policies-and-execution-control
07-algorithms-collectives-and-remote-operations
08-language-bindings-overview
09-error-handling-and-reliability
10-performance-tuning-and-scaling
11-troubleshooting-and-diagnostics
12-migration-and-upgrade-guidance
13-runtime-and-handle-model
```

## Quick jump links

- New to DTL: [Chapter 1](01-mental-model-and-core-concepts.md)
- Building and installing: [Chapter 2](02-installation-and-build-workflows.md)
- First distributed context: [Chapter 3](03-environment-context-and-backends.md)
- Core container and view usage (including `distributed_span`): [Chapters 4-5](04-distributed-containers.md)
- Policy composition and algorithm behavior: [Chapters 6-7](06-policies-and-execution-control.md)
- Bindings and interop: [Chapter 8](08-language-bindings-overview.md)
- Failure handling and recovery: [Chapter 9](09-error-handling-and-reliability.md)
- Performance optimization: [Chapter 10](10-performance-tuning-and-scaling.md)
- Runtime and handle semantics: [Chapter 13](13-runtime-and-handle-model.md)

## Legacy Deep-Dive References

These pages are retained as detailed topical references. The chaptered handbook above is the canonical path.

```{toctree}
:maxdepth: 1

environment
containers
views
policies
algorithms
bindings
error_handling
performance_tuning
troubleshooting
migration_v1_to_v15
```
