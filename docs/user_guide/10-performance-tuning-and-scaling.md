# 10. Performance Tuning and Scaling

## Performance mindset

Optimize with measurement, not assumptions. Distributed performance is typically constrained by communication, synchronization frequency, data movement, and imbalance.

## High-impact levers

- partition strategy
- placement policy and transfer behavior
- communication granularity and batching
- collective frequency and scope
- local kernel/vectorization efficiency

## Partition and locality

Select partitioning to minimize cross-rank dependencies for your dominant access pattern.

- block-like partitions often help contiguous workloads
- cyclic-like partitions can smooth imbalance for irregular workloads

## Placement and memory movement

- keep data where computation occurs when possible
- minimize host-device round trips
- use unified/device policies intentionally, not by default habit

## Algorithm-level optimization

- perform local aggregation before collective reduction
- batch remote operations
- avoid per-element remote calls in hot loops

## Synchronization strategy

- reduce unnecessary global barriers
- prefer narrower synchronization domains when semantics allow
- separate control-plane sync from data-plane operations

## Benchmarking approach

1. establish baseline with fixed input and backend config
2. vary one policy/parameter at a time
3. track both throughput and tail latency
4. validate correctness after each optimization change

## Common scaling bottlenecks

- rank skew from uneven partitioning
- high-frequency small-message collectives
- implicit sync points hidden in helper abstractions

## Further reading

- `docs/user_guide/performance_tuning.md`

## Deep-dive reference

- [Legacy Deep-Dive: Performance Tuning](performance_tuning.md)
