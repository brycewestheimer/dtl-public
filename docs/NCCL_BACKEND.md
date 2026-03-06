# NCCL Backend

DTL's NCCL backend provides GPU-native collective communication using NVIDIA's NCCL library.

## Device-Buffer Contract

All NCCL operations require **device buffers only**. Host pointers are rejected at the communicator level via `require_device_buffer()`. This is a deliberate design choice: NCCL operates on GPU memory, and silently staging through host memory would hide performance problems.

## Supported Collectives

| Operation | Status | Notes |
|---|---|---|
| `allreduce` | Supported | All NCCL reduction ops (sum, prod, min, max, avg) |
| `broadcast` | Supported | Root-rank broadcast to all |
| `reduce` | Supported | Reduce to root rank |
| `allgather` | Supported | Fixed-size only |
| `reduce_scatter` | Supported | Fixed-size only |
| `send` / `recv` | Supported | Point-to-point with stream ordering |
| `isend` / `irecv` | Supported | Async with CUDA event tracking |
| `barrier` | Supported | Implemented via allreduce on scratch buffer |

## Operation Modes

- `native_only`: only NCCL-native operations are accepted.
- `hybrid_parity`: non-native families can use explicit MPI-assisted staging
  through dedicated C APIs (`*_device_ex`).

## Unsupported Operations

| Operation | Reason |
|---|---|
| `scan` / `exscan` | Not native in NCCL (hybrid path available in explicit C API) |
| `gatherv` / `scatterv` | Not native in NCCL (hybrid path available in explicit C API) |
| `alltoallv` | Not native in NCCL (hybrid path available in explicit C API) |
| Host buffer operations | Device-buffer-only by design |

## Bootstrap Pattern

NCCL communicators are bootstrapped via MPI:

1. Rank 0 generates `ncclUniqueId` via `ncclGetUniqueId()`
2. `MPI_Bcast` distributes the unique ID to all ranks
3. Each rank calls `cudaSetDevice(device_id)` for its assigned GPU
4. `ncclCommInitRank()` creates the NCCL communicator
5. Each rank creates a CUDA stream for NCCL operations

## Synchronization

- **Blocking ops**: Call `cudaStreamSynchronize(stream_)` after issuing the NCCL operation
- **Async ops**: Record a CUDA event; `wait()` calls `cudaEventSynchronize()`; `test()` calls `cudaEventQuery()`
- **Barrier**: Issues `ncclAllReduce` on a scratch buffer, then synchronizes the stream

## C API Status

- `dtl_context_with_nccl()` / `dtl_context_with_nccl_ex(...)`: supported
- `dtl_context_split_nccl()` / `dtl_context_split_nccl_ex(...)`: supported
- mode/capability introspection:
  - `dtl_context_nccl_mode()`
  - `dtl_context_nccl_supports_native()`
  - `dtl_context_nccl_supports_hybrid()`
- explicit device collectives:
  - `dtl_nccl_allreduce_device(_ex)`
  - `dtl_nccl_broadcast_device(_ex)`
  - `dtl_nccl_barrier_device`
  - hybrid parity families via `_ex`:
    `gatherv/scatterv/allgatherv/alltoallv/scan/exscan`

## Examples

- C mode-aware context demo: `examples/c/nccl_modes.c`
- Python mode-aware context demo: `examples/python/scripts/nccl_modes.py`

## Rank-to-Device Mapping

Each MPI rank maps to exactly one GPU. The device ID is set explicitly via context construction. Multi-GPU-per-rank is not currently supported.
