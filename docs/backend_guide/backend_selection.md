# Backend Selection Guide

This guide helps you choose the right DTL backend configuration for your use case.

## Available Backends

DTL supports multiple backends that can be combined based on your hardware and requirements.

| Backend | CMake Option | Purpose | Status |
|---------|--------------|---------|--------|
| MPI | (always available) | Multi-node distributed computing | Complete |
| CPU | (always available) | CPU thread pool execution | Complete |
| CUDA | `DTL_ENABLE_CUDA` | NVIDIA GPU acceleration | Complete |
| HIP | `DTL_ENABLE_HIP` | AMD GPU acceleration | Headers (V1.2) |
| NCCL | `DTL_ENABLE_NCCL` | GPU-native collective communication | Headers (V1.2) |
| OpenSHMEM | `DTL_ENABLE_SHMEM` | PGAS communication model | Planned |

### Environment Lifecycle (V1.2)

DTL provides `environment` for unified backend lifecycle management:

```cpp
#include <dtl/core/environment.hpp>
#include <dtl/core/environment_options.hpp>

int main() {
    // DTL manages MPI init/finalize
    dtl::environment env{dtl::environment_options::defaults()};

    // Or adopt externally-initialized MPI
    MPI_Init(nullptr, nullptr);
    dtl::environment env{dtl::environment_options::adopt_mpi()};

    // Query backend availability (instance methods)
    if (env.has_mpi()) { /* ... */ }
    if (env.has_cuda()) { /* ... */ }
}
// Backends finalized on last guard destruction
```

**Backend Ownership Modes:**
- `dtl_owns` - DTL initializes and finalizes the backend
- `adopt_external` - Backend initialized externally; DTL does not finalize
- `optional` - DTL tries to initialize; silently ignores failure
- `disabled` - Backend not used

## Decision Tree

### 1. Single Node vs Multi-Node

**Single Node (one machine):**
- Use CPU backend for CPU-only workloads
- Use CUDA/HIP backend for GPU acceleration
- No MPI initialization required (use `local_context`)

**Multi-Node (cluster/supercomputer):**
- Use MPI backend for inter-node communication
- Add CUDA/HIP for GPU nodes
- Add NCCL for optimized GPU collective operations

### 2. CPU vs GPU

**CPU-Only Workloads:**
```cpp
#include <dtl/dtl.hpp>
#include <backends/cpu/cpu_executor.hpp>

// Use CPU thread pool for parallel execution
dtl::cpu::cpu_executor exec(std::thread::hardware_concurrency());
exec.parallel_for(0, n, [&](size_t i) {
    data[i] = compute(i);
});
```

**GPU-Accelerated Workloads:**
```cpp
#include <dtl/dtl.hpp>
#include <backends/cuda/cuda_memory_space.hpp>
#include <backends/cuda/cuda_executor.hpp>

// Allocate on GPU
dtl::cuda::cuda_memory_space gpu_mem;
void* d_data = gpu_mem.allocate(n * sizeof(double));

// Execute on GPU
dtl::cuda::cuda_executor exec;
exec.launch_kernel(...);
```

### 3. Communication Requirements

**No Communication (embarrassingly parallel):**
```cpp
// Local containers only, no MPI needed
dtl::distributed_vector<double> vec(local_size);
auto local = vec.local_view();
// Work purely on local data
```

**Collective Communication:**
```cpp
// MPI with collective operations
dtl::mpi::mpi_comm_adapter comm;
double local_sum = compute_local_sum();
double global_sum = comm.allreduce_sum_value(local_sum);
```

**Point-to-Point Communication:**
```cpp
// Direct rank-to-rank messaging
comm.send(&data, count, dest_rank, tag);
comm.recv(&buffer, count, src_rank, tag);
```

## Backend Combinations

### CPU Cluster (No GPUs)

**CMake Configuration:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Code Pattern:**
```cpp
#include <dtl/dtl.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#include <backends/cpu/cpu_executor.hpp>

int main(int argc, char** argv) {
    dtl::mpi::scoped_init mpi(argc, argv);
    dtl::mpi::mpi_comm_adapter comm;
    dtl::cpu::cpu_executor exec;

    // Distributed vector across MPI ranks
    dtl::distributed_vector<double> vec(global_size);

    // Local parallel computation
    auto local = vec.local_view();
    exec.parallel_for(local.size(), [&](size_t i) {
        local[i] = compute(i + vec.local_offset());
    });

    // Global reduction
    double local_sum = std::reduce(local.begin(), local.end());
    double global_sum = comm.allreduce_sum_value(local_sum);

    return 0;
}
```

### GPU Cluster (NVIDIA)

**CMake Configuration:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DDTL_ENABLE_CUDA=ON \
         -DDTL_ENABLE_NCCL=ON
```

**Code Pattern:**
```cpp
#include <dtl/dtl.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#include <backends/cuda/cuda_memory_space.hpp>
#include <backends/cuda/cuda_executor.hpp>
#include <backends/nccl/nccl_communicator.hpp>

int main(int argc, char** argv) {
    dtl::mpi::scoped_init mpi(argc, argv);

    // Set GPU based on local rank
    int local_rank = get_local_rank();
    cudaSetDevice(local_rank);

    // Use GPU memory and NCCL for GPU-direct communication
    dtl::cuda::cuda_memory_space gpu_mem;
    dtl::nccl::nccl_communicator nccl_comm;

    // Allocate on GPU
    double* d_data = static_cast<double*>(
        gpu_mem.allocate(local_size * sizeof(double)));

    // Compute on GPU (kernel launch)
    // ...

    // GPU-native collective (no CPU staging)
    nccl_comm.allreduce_sum(d_data, d_result, local_size);

    return 0;
}
```

### GPU Cluster (AMD)

**CMake Configuration:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DDTL_ENABLE_HIP=ON
```

**Code Pattern:**
```cpp
#include <dtl/dtl.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#include <backends/hip/hip_memory_space.hpp>
#include <backends/hip/hip_executor.hpp>

// Similar to CUDA pattern, using HIP equivalents
```

### Shared Memory (Single Node, No MPI)

**CMake Configuration:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Code Pattern:**
```cpp
#include <dtl/dtl.hpp>
#include <backends/shared_memory/shared_memory_communicator.hpp>
#include <backends/cpu/cpu_executor.hpp>

int main() {
    // No MPI initialization needed
    dtl::shared_memory::shared_memory_communicator comm;
    dtl::cpu::cpu_executor exec;

    // Use shared memory for inter-thread communication
    // ...

    return 0;
}
```

## Performance Considerations

### When to Use Each Backend

| Scenario | Recommended Backend | Rationale |
|----------|---------------------|-----------|
| Small data, single node | CPU only | MPI overhead not justified |
| Large data, single node | CPU + MPI | Process isolation, NUMA awareness |
| Multi-node cluster | MPI + CPU | Standard HPC configuration |
| GPU workloads | CUDA/HIP + NCCL | GPU-native operations |
| Mixed CPU/GPU | MPI + CUDA + NCCL | MPI for CPU, NCCL for GPU |

### Communication Backend Selection

| Communication Pattern | Best Backend | Notes |
|-----------------------|--------------|-------|
| CPU-to-CPU | MPI | Mature, well-optimized |
| GPU-to-GPU (same node) | NCCL | Uses NVLink/PCIe directly |
| GPU-to-GPU (cross-node) | NCCL | Uses GPU-direct RDMA |
| CPU-to-GPU | MPI + cudaMemcpy | Two-stage transfer |

### Memory Space Selection

| Workload | Memory Space | Notes |
|----------|--------------|-------|
| CPU computation | `host_memory_space` | Default, system allocator |
| GPU computation | `cuda_memory_space` | Device memory |
| Frequent CPU/GPU transfer | `cuda_managed_memory_space` | Unified memory with prefetch support |
| DMA/RDMA transfers | `pinned_memory_space` | Page-locked host (V1.2: CUDA/HIP/fallback) |
| AMD GPU computation | `hip_memory_space` | HIP device memory (V1.2 headers) |
| AMD GPU unified | `hip_managed_memory_space` | HIP managed memory with prefetch/advise (V1.2) |

### Prefetch Policies (V1.2)

For unified/managed memory, DTL provides prefetch hints:

```cpp
#include <dtl/memory/prefetch_policy.hpp>

// Prefetch policies: none, to_device, to_host, bidirectional
auto hint = dtl::make_device_prefetch(device_id, offset, size);
auto hint = dtl::make_host_prefetch(offset, size);
```

## Executor Selection

| Workload Type | Recommended Executor | Notes |
|---------------|---------------------|-------|
| Sequential | `inline_executor` | Zero overhead |
| CPU parallel | `cpu_executor` | Thread pool |
| GPU parallel | `cuda_executor` | Kernel launch |
| Mixed | Both | CPU for orchestration, GPU for compute |

**Thread Count Guidelines:**

```cpp
// Match hardware threads (typical)
dtl::cpu::cpu_executor exec;  // Uses hardware_concurrency()

// Custom thread count (e.g., for hyperthreading)
dtl::cpu::cpu_executor exec(num_physical_cores);

// Single-threaded (for debugging)
dtl::cpu::cpu_executor exec(1);
```

## Hybrid Configurations

### MPI + OpenMP + CUDA

For maximum flexibility on GPU clusters:

```cpp
// MPI across nodes
dtl::mpi::mpi_comm_adapter mpi_comm;

// OpenMP within node (via cpu_executor)
dtl::cpu::cpu_executor cpu_exec(omp_get_max_threads());

// CUDA for GPU work
dtl::cuda::cuda_executor gpu_exec;

// NCCL for GPU collectives
dtl::nccl::nccl_communicator nccl_comm;
```

### Process Placement

| Layout | Description | Use Case |
|--------|-------------|----------|
| 1 rank per node | All GPUs shared by one process | Simple, good for NCCL |
| 1 rank per GPU | Each GPU has dedicated process | Easier resource management |
| Multiple ranks per GPU | GPU sharing via MPS | Memory-limited, multi-tenant |

## Troubleshooting

### Backend Detection Issues

**MPI not found:**
```bash
# Verify MPI installation
which mpicc
mpicc --version

# Force MPI compiler
cmake .. -DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)
```

**CUDA not found:**
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Set CUDA path
export CUDA_HOME=/usr/local/cuda
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Runtime Issues

**MPI initialization fails:**
- Ensure `scoped_init` is created before any DTL distributed operations
- Check that all ranks reach the same code paths

**GPU out of memory:**
- Reduce local partition sizes
- Use unified memory for oversubscription
- Check for memory leaks with `cuda-memcheck`

**NCCL deadlock:**
- Ensure all ranks call collective operations
- Check that communicators are created in the same order

## MPI Send Mode Variants (V1.2)

DTL supports four MPI send modes via `send_mode` enum:

```cpp
#include <dtl/communication/send_mode.hpp>

// send_mode::standard     - MPI_Send (default)
// send_mode::synchronous  - MPI_Ssend (handshake, guaranteed no buffering)
// send_mode::ready        - MPI_Rsend (receiver must have pre-posted recv)
// send_mode::buffered     - MPI_Bsend (user-managed buffer)
```

The `mpi_comm_adapter` provides blocking and non-blocking variants:
- `ssend_impl()` / `issend_impl()` - Synchronous send
- `rsend_impl()` / `irsend_impl()` - Ready send

## MPMD Support (V1.2)

DTL supports Multiple Program, Multiple Data (MPMD) patterns:

```cpp
#include <dtl/mpmd/role_manager.hpp>
#include <dtl/mpmd/inter_group_comm.hpp>

// Define roles
dtl::role_manager mgr;
mgr.register_role("worker", dtl::role_assignment::first_n_ranks(3));
mgr.register_role("coordinator", dtl::role_assignment::last_rank_only());

// Initialize (assigns ranks to roles based on predicates)
mgr.initialize(world_comm);

// Query roles
if (mgr.has_role("worker")) {
    auto& group = mgr.get_group("worker");
    // group.local_rank(), group.size(), group.members()
}

// Inter-group communication via rank translation
auto world_rank = dtl::translate_to_world_rank(dest_group, local_rank);
```

## Runtime Library

DTL requires `libdtl_runtime.so` at runtime for backend lifecycle management. This shared library contains the process-global singleton (`runtime_registry`) that manages MPI, CUDA, HIP, NCCL, and SHMEM initialization and finalization.

When using CMake with `DTL::dtl`, the runtime library is linked automatically (transitive dependency). For manual builds, add `-ldtl_runtime` to your linker flags.

## See Also

- [Backend Concepts](concepts.md) - Concept definitions
- [Implementing a Backend](implementing_backend.md) - Adding new backends
