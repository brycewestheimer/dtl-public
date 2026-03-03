# 4. Distributed Containers

## Container families

Common distributed containers include:

- `distributed_vector`
- `distributed_array`
- `distributed_span`
- `distributed_tensor`
- `distributed_map`

Each container combines global semantics with rank-local partition ownership.

## Core container properties

- global size/shape
- local size and local offset
- partition and placement policy behavior
- validity and synchronization state

## Constructing containers

Typical construction parameters include:

- context
- global extent/size
- optional initial value
- policy set (partition/placement/execution/consistency/error)

### Example

```cpp
// Illustrative pattern (exact ctor overloads vary)
auto vec = dtl::distributed_vector<double>(ctx, N);
auto local = vec.local_view();
```

## Local data access

Prefer local operations when possible:

- use `local_view()` for algorithmic work on owned partition
- use local iterator/range operations that do not communicate

## Structural operations

Some operations are collective or involve redistribution:

- repartition/redistribution
- halo-related synchronization
- operations requiring consistent global structure

Always check operation contract and rank participation requirements.

## Container-specific notes

### Vector/array

Best for linear data-parallel patterns and reductions/scans.

### Span

Use `distributed_span` for non-owning distributed views when you need span-like local access without container ownership:

- construct from a distributed container (`dtl::distributed_span<T> span(vec)`)
- construct from explicit local pointer/size/global-size metadata
- treat it as borrowed storage: the underlying owner must outlive the span

`distributed_span` is especially useful for adapters/helpers that should not own distributed memory but still need rank-aware size metadata.

### Tensor

Use when dimensional indexing/layout is central to workload design.

### Map

Key ownership and remote key behavior can differ from local `std::map` assumptions; check distributed map semantics before assuming local insert/erase behavior.

## Safety and correctness checklist

- verify context validity before container construction
- verify policy compatibility with target backend
- use explicit barriers/sync where algorithm requires it
- for `distributed_span`, verify owner/container lifetime and refresh spans after structural changes to owner containers

## Next step

Move to [Chapter 5](05-views-iteration-and-data-access.md) to program with views and iterators correctly.

## Deep-dive reference

- [Legacy Deep-Dive: Containers](containers.md)
