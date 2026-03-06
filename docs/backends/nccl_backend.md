# NCCL Backend

The NCCL backend provides explicit GPU-to-GPU collective communication for CUDA
device buffers. It is not a generic MPI-parity communication layer, and it is
not selected implicitly for the generic distributed algorithm layer.

## Overview

Use NCCL when all of the following are true:

- Data is already resident in CUDA device memory.
- You need GPU-native collectives between ranks.
- You are calling the NCCL domain/communicator/adapter explicitly.

Keep using MPI for generic multi-rank algorithms, host-buffer collectives, and
contexts passed into the existing distributed algorithm layer.

## Requirements

- CUDA Toolkit 11+
- NCCL 2.7+ for send/recv support
- MPI for communicator bootstrap and domain creation
- Compatible NVIDIA GPUs

## CMake Configuration

```cmake
cmake -DDTL_ENABLE_NCCL=ON -DDTL_ENABLE_CUDA=ON -DDTL_ENABLE_MPI=ON ..
```

## Support Model

### What Is Supported

- `nccl_domain::from_mpi(...)`
- `context::with_nccl(device_id)`
- `context::split_nccl(...)` as a C++-only API
- Explicit `nccl_domain::adapter()` access
- Device-buffer point-to-point operations (`send`/`recv`, `isend`/`irecv`)
- Device-buffer collectives:
  - `broadcast`
  - `scatter`
  - `gather`
  - `allgather`
  - `alltoall`
  - `barrier`
  - typed/allreduce-sum and reduce-sum paths that are actually backed by NCCL

### What Is Not Supported

- Generic algorithm auto-dispatch from `context` to NCCL
- Host-buffer collectives
- Scalar convenience helpers such as `allreduce_sum_value`
- Scan/exscan on the NCCL adapter
- Variable-size collectives (`gatherv`, `scatterv`, `allgatherv`, `alltoallv`)
- Logical reductions (`land`, `lor`)
- MPI-style feature parity claims beyond the explicitly supported device-buffer API

## Domain Creation

Create an NCCL domain from MPI:

```cpp
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

dtl::mpi_domain mpi;
int device_id = mpi.rank() % num_gpus;

auto nccl_result = dtl::nccl_domain::from_mpi(mpi, device_id);
if (!nccl_result) {
    throw std::runtime_error(nccl_result.error().message());
}

dtl::nccl_domain nccl = std::move(*nccl_result);
```

## Context Integration

Add NCCL explicitly to an MPI context:

```cpp
#include <dtl/core/context.hpp>

dtl::mpi_context ctx;
int device_id = ctx.rank() % num_gpus;

auto nccl_result = ctx.with_nccl(device_id);
if (!nccl_result) {
    throw std::runtime_error(nccl_result.error().message());
}

auto& nccl_ctx = *nccl_result;
auto& mpi = nccl_ctx.get<dtl::mpi_domain>();
auto& nccl = nccl_ctx.get<dtl::nccl_domain>();
```

Important:

- Context rank/size queries remain MPI-oriented for generic distributed code.
- Having an NCCL domain in a context does not imply that generic algorithms will
  switch to NCCL.
- `split_nccl(...)` is currently public only in the C++ API. C, Python, and
  Fortran bindings expose `with_nccl(...)`, not `split_nccl(...)`.

## Communicator Layers

### `nccl_communicator`

This is the low-level result-returning API. It is the right layer when you want
direct NCCL control and explicit status handling.

```cpp
#include <backends/nccl/nccl_communicator.hpp>

double* d_send = /* device buffer */;
double* d_recv = /* device buffer */;

auto result = comm.allreduce(d_send, d_recv, count, dtl::nccl::nccl_op::sum);
if (!result) {
    throw std::runtime_error(result.error().message());
}
```

### `nccl_comm_adapter`

This is the explicit device-buffer adapter exposed from `nccl_domain::adapter()`.
It throws `nccl::communication_error` on failure and is intentionally narrower
than MPI.

```cpp
#include <backends/nccl/nccl_comm_adapter.hpp>

auto& adapter = nccl_domain.adapter();
adapter.allreduce_sum(d_send, d_recv, count);
adapter.broadcast(d_buf, count, 0);
```

The adapter should be read as:

- explicit
- device-buffer-only
- suitable for NCCL-native communication paths
- not a drop-in replacement for host-oriented MPI convenience APIs

## Buffer and Synchronization Semantics

- NCCL collectives require CUDA device memory.
- Host pointers, stack scalars, and host scratch buffers are invalid.
- Blocking NCCL operations synchronize the CUDA stream before returning.
- Non-blocking NCCL operations complete through CUDA event tracking, and
  `wait()` / `test()` surface CUDA/NCCL failures rather than silently hiding them.

## C API

Only NCCL context creation is currently exposed in the C bindings:

```c
#include <dtl/bindings/c/dtl_context.h>

dtl_context_t ctx;
dtl_context_create_default(&ctx);

dtl_context_t nccl_ctx;
dtl_status status = dtl_context_with_nccl(ctx, device_id, &nccl_ctx);
```

There is currently no C binding for `split_nccl(...)`.

## Unsupported Operations

| Feature | Status |
|---|---|
| Message tags | Accepted for API compatibility; ignored by NCCL |
| `probe` / `iprobe` | Unsupported |
| `ssend` / `rsend` / `issend` / `irsend` | Unsupported |
| Scalar convenience helpers | Unsupported on the NCCL adapter |
| Logical reductions | Unsupported |
| Bitwise reductions | Unsupported |
| Scan/exscan | Unsupported on the NCCL adapter |
| Variable-size collectives | Unsupported |
| RMA / one-sided communication | Unsupported |
| Generic context auto-dispatch to NCCL | Unsupported by design |

## Testing

Relevant coverage should focus on:

- device-buffer collectives on real NCCL contexts
- rejection of host buffers
- blocking completion semantics
- explicit context/domain usage

Run NCCL-tagged integration tests when MPI, CUDA, and NCCL are available:

```bash
ctest -L nccl --output-on-failure
```

## Design Notes

- NCCL remains separate from CUDA execution and generic algorithm dispatch.
- MPI remains the primary communicator for generic multi-rank algorithms.
- Unsupported MPI features stay unsupported instead of being approximated with
  unsafe host-side emulation.

## See Also

- [CUDA Backend](cuda_guide.md)
- [Backend Comparison](comparison.md)
- [NCCL/CUDA Audit](nccl_cuda_audit.md)
