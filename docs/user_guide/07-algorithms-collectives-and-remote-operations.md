# 7. Algorithms, Collectives, and Remote Operations

## Algorithm model

DTL algorithms combine STL-style interfaces with distributed semantics.

Typical categories:

- non-modifying algorithms (`for_each`, predicates/find/count variants)
- modifying algorithms (`transform`, copy/fill/replace-like paths)
- reductions/scans
- sorting and partitioning-related algorithms

## Local vs collective semantics

Understand contract before use:

- local algorithms operate on rank-local partitions
- collective/distributed algorithms require all ranks in communicator to participate

## Collective operations

Core collective patterns include:

- barrier
- broadcast
- gather/allgather
- reduce/allreduce
- scan/exscan

Collective calls must satisfy participation contracts to avoid deadlock or undefined behavior.

## Remote and RPC-style operations

Remote operation modules enable explicit cross-rank invocation and messaging patterns.

Guidance:

- keep payloads serialization-safe and version-aware
- handle async request lifecycle explicitly
- avoid unbounded in-flight request growth

## RMA and asynchronous requests

When using one-sided or async APIs:

- treat request handles as owned resources
- always complete/test/wait/destroy per contract
- ensure window/context lifetime outlives active requests

## Correctness guidelines

1. isolate collective boundaries in code structure
2. avoid mixing blocking collectives with unmatched control flow
3. keep remote operations explicit and auditable

## Next step

Proceed to [Chapter 8](08-language-bindings-overview.md) for multi-language usage.

## Deep-dive reference

- [Legacy Deep-Dive: Algorithms](algorithms.md)
