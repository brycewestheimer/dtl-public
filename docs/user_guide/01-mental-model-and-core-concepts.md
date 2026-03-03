# 1. Mental Model and Core Concepts

## What DTL is

DTL is a C++20 library for distributed and heterogeneous programming with STL-inspired interfaces. It is intentionally explicit about distributed behavior.

DTL is not a transparent distributed memory abstraction. It favors correctness and predictability over hidden communication.

## Core ideas

### Explicit ownership and placement

Data is distributed across ranks and may be placed in host, unified, or device memory. Placement decisions are policy-controlled.

### Explicit local vs remote behavior

- `local_view()` operations are local and should not communicate.
- remote operations (`remote_ref`, collectives, remote invocation) represent potential communication and synchronization.

### Policy-first architecture

Container behavior is shaped by orthogonal policy axes:

- partition policy
- placement policy
- execution policy
- consistency policy
- error policy

### Context-driven execution

A `context` captures backend capabilities (MPI/CPU/GPU/etc.) and runtime domains used by operations.

## Fundamental terms

- Rank: process identity in communicator space
- Local partition: data owned by current rank
- Global index space: whole container index domain
- Collective operation: all participating ranks must call the same operation
- Determinism mode: runtime policy for reproducibility vs throughput tradeoffs

## First principles for users

1. Know whether your operation is local-only or collective.
2. Choose policies explicitly for performance-sensitive workloads.
3. Assume communication has cost and should be visible in code.
4. Treat backend availability as runtime/compile-time capability, not a guarantee.

## Relationship to STL

DTL mirrors STL patterns where semantics remain honest:

- familiar container/view/algorithm style
- direct non-owning analogs where appropriate (for example `std::span` -> `dtl::distributed_span`)
- explicit extensions for distributed ownership and communication
- no fake `T&` semantics for remote references

## Recommended next step

Continue with [Chapter 2](02-installation-and-build-workflows.md) and then [Chapter 3](03-environment-context-and-backends.md).
