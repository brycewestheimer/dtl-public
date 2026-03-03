# Python API Reference

**DTL Version:** 0.1.0-alpha.1
**Last Updated:** 2026-03-03

The Python package exposes the alpha binding surface from
`bindings/python/src/dtl/__init__.py` and `bindings/python/src/dtl/__init__.pyi`.
Package versioning follows PEP 440: `dtl.__version__ == "0.1.0a1"` and
`dtl.version_info == (0, 1, 0)`.

## Runtime and Ownership

- `dtl.Context` and `dtl.Environment` own native resources and should be used as context managers where practical.
- Collective calls such as `barrier`, `broadcast`, `reduce`, `allreduce`, `gather`, `scatter`, `allgather`, `allgatherv`, `alltoallv`, `gatherv`, and `scatterv` require consistent participation from all ranks in the active communicator.
- Host-only execution is supported. Backend-specific entry points must be guarded with feature checks such as `dtl.has_mpi()` and `dtl.placement_available(...)`.
- Several advanced surfaces are intentionally alpha-quality. When noted below as local-only or no-op, that behavior is the actual contract for `v0.1.0-alpha.1`.

## Version and Backend Detection

```python
import dtl

dtl.__version__      # "0.1.0a1"
dtl.version_info     # (0, 1, 0)

dtl.has_mpi()
dtl.has_cuda()
dtl.has_hip()
dtl.has_nccl()
dtl.has_shmem()

dtl.backends.available()
dtl.backends.count()
dtl.backends.name()
```

## Environment and Context

### `dtl.Environment`

Use `Environment` when you need explicit lifecycle and capability queries.

```python
with dtl.Environment() as env:
    env.has_mpi
    env.has_cuda
    env.make_world_context()
    env.make_cpu_context()
```

### `dtl.Context`

`Context` encapsulates communicator/domain selection and is the required entry
point for distributed containers and communication APIs.

```python
with dtl.Context() as ctx:
    ctx.rank
    ctx.size
    ctx.is_root
    ctx.device_id
    ctx.has_mpi
    ctx.has_cuda
    ctx.barrier()
    ctx.fence()
```

Context factories:

```python
ctx.dup()                 # collective duplicate
ctx.split(color=0, key=0) # collective split
ctx.with_cuda(0)
ctx.with_nccl(0)
```

## Containers

Factory functions exported by the package:

```python
dtl.DistributedVector(...)
dtl.DistributedArray(...)
dtl.DistributedTensor(...)
dtl.DistributedSpan(container)
dtl.DistributedMap(...)
```

Common container semantics:

- `local_view()` exposes the local partition.
- `to_numpy()` copies local data into a new NumPy array.
- Partition, placement, and execution policies are passed as keyword arguments.
- Device placements must be feature-checked before use.

### `DistributedMap` alpha limitations

- `global_size` is currently local-size semantics, not a cross-rank reduction.
- `sync()` is currently a local no-op.
- `flush()` is currently a local no-op.
- `clear()` affects only the local partition.

## Policy Constants

```python
dtl.PARTITION_BLOCK
dtl.PARTITION_CYCLIC
dtl.PARTITION_BLOCK_CYCLIC
dtl.PARTITION_HASH
dtl.PARTITION_REPLICATED

dtl.PLACEMENT_HOST
dtl.PLACEMENT_DEVICE
dtl.PLACEMENT_UNIFIED
dtl.PLACEMENT_DEVICE_PREFERRED

dtl.EXEC_SEQ
dtl.EXEC_PAR
dtl.EXEC_ASYNC
```

## Collective and Point-to-Point Operations

Collectives:

```python
dtl.allreduce(ctx, value, op=dtl.SUM)
dtl.reduce(ctx, value, op=dtl.SUM, root=0)
dtl.broadcast(ctx, value, root=0)
dtl.gather(ctx, value, root=0)
dtl.scatter(ctx, values, root=0)
dtl.allgather(ctx, value)
dtl.allgatherv(ctx, value)
dtl.alltoallv(ctx, send_data, send_counts, recv_counts)
dtl.gatherv(ctx, data, recvcounts=None, root=0)
dtl.scatterv(ctx, data, sendcounts, root=0)
```

Point-to-point:

```python
dtl.send(ctx, data, dest, tag=0)
dtl.recv(ctx, source, tag=0, dtype=None, count=0)
dtl.sendrecv(ctx, send_data, dest, source, send_tag=0, recv_tag=0)
dtl.probe(ctx, source=-1, tag=-1)
dtl.iprobe(ctx, source=-1, tag=-1)
```

## RMA and Window APIs

The package exports `Window` plus one-sided helpers:

```python
dtl.Window(...)
dtl.rma_put(...)
dtl.rma_get(...)
dtl.rma_accumulate(...)
dtl.rma_fetch_and_add(...)
dtl.rma_compare_and_swap(...)
```

These operations require communicator and memory-window participation rules
that match the active backend. Treat them as advanced alpha APIs and verify the
runtime/backend combination with `dtl.has_mpi()` or device capability checks
before use.

## Futures, Topology, and MPMD

Futures and async helpers:

```python
dtl.Future
dtl.when_all(...)
dtl.when_any(...)
dtl.async_for_each(...)
dtl.async_transform(...)
dtl.async_reduce(...)
dtl.async_sort(...)
```

Topology and MPMD exports:

```python
dtl.Topology
dtl.RoleManager
dtl.intergroup_send(...)
dtl.intergroup_recv(...)
```

These surfaces are part of the exported alpha package but remain less mature
than the core host/container path. Users should treat them as experimental and
prefer explicit feature detection and focused validation in their own runtime
environment.
