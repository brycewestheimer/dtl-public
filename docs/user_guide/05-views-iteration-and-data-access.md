# 5. Views, Iteration, and Data Access

## Why views matter

Views define whether an operation is local, global, segmented, or remote. Correct view selection is central to both correctness and performance.

## Local view

`local_view()` provides rank-local access and should not trigger communication.

Use for:

- local transforms
- local reductions
- staging data for collective operations

## `distributed_span` for non-owning local windows

When an API should consume distributed data without owning it, use `distributed_span` as a span-like adapter over rank-local contiguous storage plus global metadata.

- prefer container-owned `local_view()` for normal workflows
- use `distributed_span` for borrowed, explicit non-owning interfaces
- keep the owning container alive for the full span lifetime

## Global and segmented views

Use global/segmented abstractions when algorithm semantics explicitly involve global index space or segment-aware distributed traversal.

Segmented iteration is often preferred for scalable distributed algorithm structure.

## Remote reference (`remote_ref`)

Remote references are intentionally explicit and loud:

- no implicit `T&` conversion
- access semantics indicate potential communication

This avoids accidental hidden network behavior in generic-looking code.

## Iteration guidance

- Prefer local iterators inside compute kernels.
- Use segmented traversal primitives for distributed algorithms.
- Avoid writing hot loops that repeatedly perform remote operations per element without batching or algorithmic restructuring.

## Access pattern best practices

1. bulk local processing first
2. reduce communication surface area
3. isolate remote interactions and synchronization boundaries

## Common anti-patterns

- per-element remote get/put in deeply nested loops
- mixing local and collective semantics in opaque helper functions
- assuming remote access has local latency/ordering properties

## Next step

Proceed to [Chapter 6](06-policies-and-execution-control.md) for policy composition details.

## Deep-dive reference

- [Legacy Deep-Dive: Views](views.md)
- [Runtime and Handle Model](13-runtime-and-handle-model.md)
