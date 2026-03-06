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

## Unsupported Operations

| Operation | Reason |
|---|---|
| `scan` / `exscan` | Not available in NCCL API |
| `gatherv` / `scatterv` | Variable-size collectives not supported |
| `alltoall` / `alltoallv` | Not available in NCCL (use MPI for this) |
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

- `dtl_context_with_nccl()`: Returns `DTL_ERROR_NOT_SUPPORTED` (deferred)
- `dtl_context_split_nccl()`: Returns `DTL_ERROR_NOT_SUPPORTED` (deferred)

NCCL in the C API is deferred until device C containers are validated at scale. The C++ NCCL communicator is mature and used directly in integration tests.

## Rank-to-Device Mapping

Each MPI rank maps to exactly one GPU. The device ID is set explicitly via context construction. Multi-GPU-per-rank is not currently supported.
