# NCCL Backend

The NCCL (NVIDIA Collective Communication Library) backend provides high-performance GPU-to-GPU collective communication for DTL.

## Overview

The NCCL backend enables multi-GPU collective operations that bypass CPU memory and communicate directly between GPUs over NVLink, PCIe, or InfiniBand (with GPU Direct RDMA). This is essential for distributed deep learning and HPC workloads.

## Requirements

- **CUDA Toolkit** (11.0+)
- **NCCL Library** (2.7+ recommended for send/recv support)
- **MPI** (for bootstrapping NCCL communicator initialization)
- Compatible NVIDIA GPUs

## CMake Configuration

Enable the NCCL backend with:

```cmake
cmake -DDTL_ENABLE_NCCL=ON -DDTL_ENABLE_CUDA=ON -DDTL_ENABLE_MPI=ON ..
```

## API Overview

### Domain Creation

The `nccl_domain` represents a NCCL communication context. Create it from an MPI domain:

```cpp
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

// Create MPI domain first
dtl::mpi_domain mpi;

// Each rank specifies its GPU device
int device_id = mpi.rank() % num_gpus;

// Create NCCL domain (collective operation - all ranks must call)
auto result = dtl::nccl_domain::from_mpi(mpi, device_id);
if (result) {
    dtl::nccl_domain& nccl = *result;
    // nccl.rank(), nccl.size(), nccl.valid()
}
```

### Context Integration

Add NCCL capabilities to an existing MPI context:

```cpp
#include <dtl/core/context.hpp>
#include <dtl/core/domain_impl.hpp>

// Create base MPI context
dtl::mpi_context ctx;

// Add NCCL domain
int device_id = ctx.rank() % num_gpus;
auto nccl_result = ctx.with_nccl(device_id);

if (nccl_result) {
    auto& nccl_ctx = *nccl_result;

    // Access domains
    auto& mpi = nccl_ctx.get<dtl::mpi_domain>();
    auto& nccl = nccl_ctx.get<dtl::nccl_domain>();

    // Rank/size queries use NCCL if available
    std::cout << "Rank " << nccl_ctx.rank()
              << " of " << nccl_ctx.size() << "\n";
}
```

### NCCL Communicator

For direct NCCL operations, use `nccl_communicator`:

```cpp
#include <backends/nccl/nccl_communicator.hpp>

// The communicator is typically obtained from nccl_domain
// For direct NCCL collective operations:

dtl::nccl::nccl_communicator comm;  // (initialized from domain)

// Typed allreduce
float* d_send = /* device buffer */;
float* d_recv = /* device buffer */;
auto result = comm.allreduce(d_send, d_recv, count, dtl::nccl::nccl_op::sum);

// In-place allreduce
result = comm.allreduce_inplace(d_buf, count);

// Reduce to root
result = comm.reduce(d_send, d_recv, count, root_rank);

// Reduce-scatter
result = comm.reduce_scatter(d_send, d_recv, recv_count);
```

### Group Operations

Batch multiple NCCL operations for efficiency:

```cpp
#include <backends/nccl/nccl_group_ops.hpp>

{
    dtl::nccl::scoped_group_ops group;
    if (group.valid()) {
        // All NCCL calls here are batched
        comm.allreduce(...);
        comm.broadcast(...);
    }
    // ncclGroupEnd called automatically
}
```

## C API

NCCL context creation is available in the C bindings:

```c
#include <dtl/bindings/c/dtl_context.h>

dtl_context_t ctx;
dtl_context_create_default(&ctx);

dtl_context_t nccl_ctx;
dtl_status status = dtl_context_with_nccl(ctx, device_id, &nccl_ctx);

if (status == DTL_SUCCESS) {
    // Use nccl_ctx for GPU collective operations
}

dtl_context_destroy(nccl_ctx);
dtl_context_destroy(ctx);
```

## Device Selection

Each MPI rank must select a CUDA device. Common patterns:

### Round-Robin

```cpp
int rank, device_count;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
cudaGetDeviceCount(&device_count);
int device_id = rank % device_count;
```

### Topology-Aware

For multi-node setups, use local rank for device selection:

```cpp
// Get local rank (ranks on same node)
MPI_Comm local_comm;
MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &local_comm);

int local_rank;
MPI_Comm_rank(local_comm, &local_rank);
int device_id = local_rank % device_count;
```

## Error Handling

All NCCL operations return `result<T>` with detailed error information:

```cpp
auto result = nccl_domain::from_mpi(mpi, device_id);
if (!result) {
    std::cerr << "NCCL init failed: " << result.error().message() << "\n";
    // Handle: invalid_state, backend_error, communication_failed
}
```

Common errors:
- `invalid_state`: MPI domain not valid
- `invalid_argument`: Invalid device ID
- `backend_error`: CUDA or NCCL initialization failed
- `communication_failed`: NCCL collective operation failed

## Testing

Run NCCL tests with:

```bash
# Build with NCCL support
cmake -DDTL_ENABLE_NCCL=ON -DDTL_ENABLE_CUDA=ON \
      -DDTL_BUILD_INTEGRATION_TESTS=ON ..
make

# Run NCCL tests (requires 2+ GPUs or MPI oversubscription)
ctest -L nccl --output-on-failure

# Or manually with MPI
mpirun -n 2 ./tests/mpi/nccl/test_nccl_domain_from_mpi
```

## Limitations

- **MPI Required**: NCCL domain creation requires MPI for unique ID broadcast
- **Single Communicator**: Currently creates one communicator matching MPI_COMM_WORLD
- **No Direct P2P**: Use NCCL collectives; point-to-point returns `not_supported`

## See Also

- [Backend Selection Guide](../backend_guide/backend_selection.md)
- [CUDA Backend](../gpu/cuda_backend.md)
