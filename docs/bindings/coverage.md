# DTL Bindings Coverage Matrix

**Last Updated:** 2026-03-06

This document shows what C++ features are exposed in each language binding and what remains deferred.

> **Note:** For authoritative policy support information, see the [Policy Capability Matrix](../capabilities/policy_matrix.md). This document focuses on API coverage; the capability matrix details which policy combinations actually work.

## Legend

| Symbol | Meaning |
|--------|---------|
| **Y** | Fully bound |
| **E** | Experimental (known stability issues) |
| **P** | Partial (subset of C++ API) |
| **-** | Not bound (by design or deferred) |

---

## Core / Context

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| Context create/destroy | Y | Y | |
| Rank / size queries | Y | Y | |
| Barrier | Y | Y | |
| Backend feature detection | Y | Y | `has_mpi()`, `has_cuda()`, etc. |
| Environment lifecycle | Y | Y | `dtl_environment_create/destroy`, `dtl.Environment()` |
| Environment state queries | Y | Y | `is_initialized()`, `ref_count()` |
| Environment backend queries | Y | Y | `has_mpi()`, `has_cuda()`, etc. |
| Environment context factories | Y | Y | `make_world_context()`, `make_cpu_context()` |
| Context domain composition (`with_cuda`) | Y | Y | |
| NCCL context composition (`with_nccl` / `split_nccl`) | Y | Y | Mode-aware in Python (`mode=`), C exposes `_ex` variants |
| NCCL mode query (`nccl_mode`) | Y | Y | C: `dtl_context_nccl_mode`, Python: `Context.nccl_mode` |
| NCCL capability query (`supports_native/hybrid`) | Y | Y | Operation-family capability checks |
| Environment options | - | - | C++ only (`environment_options`) |

## NCCL Mode-Aware Backend Surface

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| Mode enum (`native_only`, `hybrid_parity`) | Y | Y | Python constants `DTL_NCCL_MODE_*` |
| Native/hybrid capability introspection | Y | Y | `dtl_context_nccl_supports_*`, `Context.nccl_supports_*` |
| Explicit NCCL device collectives (`*_device`, `*_device_ex`) | Y | - | Python currently uses generic collective API surface |
| Hybrid parity families (`*_device_ex` for scan/var-size) | Y | - | Explicit in C ABI |

## Containers

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| `distributed_vector` | Y | Y | All 8 dtypes |
| `distributed_array` | Y | Y | All 8 dtypes |
| `distributed_span` | Y | Y | Non-owning; construct from vector/array/tensor |
| `distributed_tensor` | Y | Y | V2 vtable dispatch, 4 dtypes (f32, f64, i32, i64) |
| `distributed_map` | Y | - | V2 vtable dispatch, all dtypes, Python deferred: needs dict-like API design |
| `local_view()` (zero-copy) | Y | Y | NumPy array in Python |
| `global_size` / `local_size` | Y | Y | |
| `fill()` | Y | Y | |
| `is_dirty()` | - | Y | Python-side state tracking |
| `sync()` | - | Y | Via barrier |
| `redistribute()` | - | P | Warning: requires reallocation |

## Algorithms

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| `for_each` | Y | Y | Vector + Array |
| `transform` | Y | Y | Vector + Array |
| `copy` | Y | Y | Vector + Array |
| `fill` | Y | Y | Vector + Array |
| `find` / `find_if` | Y | Y | Vector + Array |
| `count` / `count_if` | Y | Y | Vector + Array |
| `reduce` (local) | Y | Y | Vector + Array |
| `sort` | Y | Y | Vector + Array |
| `inclusive_scan` | Y | Y | Vector + Array |
| `exclusive_scan` | Y | Y | Vector + Array |
| `minmax` | Y | Y | Vector + Array |
| Async algorithms | E | - | Python deferred: futures stability |

## Collective Operations

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| `allreduce` | Y | Y | SUM, PROD, MIN, MAX |
| `reduce` | Y | Y | To root rank |
| `broadcast` | Y | Y | From root rank |
| `gather` | Y | Y | To root rank |
| `scatter` | Y | Y | From root rank |
| `allgather` | Y | Y | |
| `allgatherv` | Y | Y | Variable-size |
| `alltoallv` | Y | Y | Variable-size |
| `send` / `recv` | Y | Y | Point-to-point |
| `sendrecv` | Y | Y | Simultaneous |
| `isend` / `irecv` | Y | - | Python async deferred |
| `ssend` / `rsend` | Y | - | Specialized send modes |

## Policies

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| Partition policies | Y | P | Vtable dispatch |
| Placement policies | Y | P | Vtable dispatch; device_id supported |
| Execution policies | Y | P | Vtable dispatch |
| Policy options at creation | Y | - | Implemented via vtable dispatch |

## RMA (Remote Memory Access)

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| Window create/destroy | Y | Y | |
| Window allocate | Y | Y | |
| Fence synchronization | Y | Y | |
| Lock/unlock | Y | Y | Exclusive + shared |
| Lock all/unlock all | Y | Y | |
| Flush operations | Y | Y | flush, flush_all, flush_local |
| `put` / `get` | Y | Y | |
| Async `put` / `get` | Y | P | Python: limited async support |
| `accumulate` | Y | Y | |
| `fetch_and_op` | Y | Y | |
| `compare_and_swap` | Y | Y | |

## MPMD (Multi-Program Multiple Data)

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| Role manager create/destroy | Y | Y | |
| Add role / initialize | Y | Y | |
| Role queries (has_role, size, rank) | Y | Y | |
| Inter-group send/recv | Y | Y | Requires MPI |

## Topology

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| CPU count | Y | Y | `std::thread::hardware_concurrency()` |
| GPU count | Y | Y | CUDA/HIP runtime detection |
| CPU affinity | Y | Y | Round-robin mapping |
| GPU device ID | Y | Y | Round-robin mapping |
| Node locality | Y | Y | MPI shared memory split |
| Node ID | Y | Y | MPI shared memory split |

## Futures

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| Future create/destroy | E | E | Experimental |
| Wait / test | E | E | Experimental |
| Get / set value | E | E | Experimental |
| `when_all` | E | E | Experimental |
| `when_any` | E | E | Experimental |

## Device Placement Support

> See [Device Placement Semantics](device_placement_semantics.md) for the authoritative binding contract.

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| `local_view()` for host placement | Y | Y | Zero-copy NumPy array |
| `local_view()` for unified placement | Y | Y | Zero-copy (sync required after GPU ops) |
| `local_view()` for device-only | - | - | Returns NULL/raises error |
| `copy_to_host()` | Y | Y | Explicit copy for device containers |
| `copy_from_host()` | Y | Y | Explicit copy to device containers |
| `is_host_accessible()` query | Y | Y | Placement capability check |
| `is_device_accessible()` query | Y | Y | Placement capability check |
| CuPy integration | - | P | Optional, requires CuPy installed |

## Remote / RPC

| Feature | C ABI | Python | Notes |
|---------|:-----:|:------:|-------|
| Action register/destroy | Y | Y | |
| Synchronous invoke | Y | Y | Same-rank + MPI remote |
| Async invoke | Y | P | Python: limited |

---

## Deferred Items (with Rationale)

| Feature | Reason | Target Phase |
|---------|--------|-------------|
| `distributed_map` Python binding | Requires dict-like API design (insert, __getitem__, iteration) | Phase 13 |
| Python mpi4py interop | MPI_Comm lifetime management across Python/C boundary | Phase 13 |
| Python policy options at creation | Requires template specialization in C ABI | Phase 13 |
| Python async algorithms | Depends on futures progress engine stability | After futures fix |
| Python `isend`/`irecv` | Depends on Python async/await integration design | Phase 13 |
| Fortran high-level convenience wrappers | Core `dtl_span_*`/container interfaces are present; additional typed convenience wrappers are deferred | Future phase |
| `environment` RAII in bindings | C++ only feature; C/Python use context pattern | By design |

---

## Coverage Summary

| Domain | C ABI | Python |
|--------|:-----:|:------:|
| Core / Context | 100% | 100% |
| Containers | 95% | 85% |
| Algorithms | 100% | 95% |
| Collectives | 100% | 90% |
| Policies | 100% | 100% |
| RMA | 100% | 95% |
| MPMD | 100% | 100% |
| Topology | 100% | 100% |
| Futures | 100% (E) | 100% (E) |
| Remote/RPC | 100% | 90% |
| **Overall** | **~99%** | **~90%** |

All containers use V2 vtable dispatch architecture. New status codes `DTL_NOT_FOUND` (1) and `DTL_END` (2) added for map sentinel returns.
