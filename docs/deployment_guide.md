# DTL Deployment Guide

This guide covers deploying DTL in production environments, from single-node development to large-scale HPC clusters.

## Deployment Scenarios

| Scenario | Configuration | Key Considerations |
|----------|---------------|-------------------|
| Development | Single node, no MPI | Fast iteration, debugging |
| Testing | Multi-process, local | CI/CD integration |
| Small cluster | 2-16 nodes | Standard MPI setup |
| Large cluster | 100+ nodes | Job scheduler, tuning |
| GPU cluster | MPI + CUDA/HIP | Device selection, NCCL |

## Prerequisites

### Compiler Requirements

| Compiler | Minimum Version | Notes |
|----------|-----------------|-------|
| GCC | 10.0+ | Full C++20 concepts support |
| Clang | 12.0+ | Full C++20 concepts support |
| NVCC | 11.4+ | With compatible host compiler |
| MSVC | 19.29+ | Visual Studio 2019 16.10+ |

### Runtime Dependencies

| Dependency | Required | Notes |
|------------|----------|-------|
| MPI | Yes | OpenMPI, MPICH, Intel MPI |
| CUDA | Optional | For NVIDIA GPUs |
| HIP | Optional | For AMD GPUs |
| NCCL | Optional | For GPU collectives |

## Installation Methods

### Method 1: From Source (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/dtl.git
cd dtl

# Configure
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/dtl

# Build and install
make -j$(nproc)
sudo make install
```

### Method 2: CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    dtl
    GIT_REPOSITORY https://github.com/your-org/dtl.git
    GIT_TAG v0.1.0-alpha.1
)
FetchContent_MakeAvailable(dtl)

target_link_libraries(myapp PRIVATE dtl::dtl)
```

### Method 3: Header-Only Integration

```bash
# Copy headers to your project
cp -r dtl/include/dtl your-project/third_party/
cp -r dtl/backends your-project/third_party/dtl/

# Add to include path
# -I your-project/third_party
```

## Configuration for Production

### CMake Build Options

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native" \
    -DDTL_ENABLE_CUDA=ON \
    -DDTL_ENABLE_NCCL=ON \
    -DDTL_BUILD_TESTS=OFF \
    -DDTL_BUILD_BENCHMARKS=OFF
```

### Compiler Optimization Flags

| Flag | Purpose |
|------|---------|
| `-O3` | Maximum optimization |
| `-march=native` | CPU-specific instructions |
| `-DNDEBUG` | Disable assertions |
| `-flto` | Link-time optimization |

### MPI Configuration

Ensure MPI is properly configured:

```bash
# Verify MPI installation
mpirun --version
mpicc --version

# Check for high-performance transports
ompi_info | grep -i transport
```

## Cluster Deployment

### SLURM Integration

Example SLURM batch script:

```bash
#!/bin/bash
#SBATCH --job-name=dtl_job
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=compute

module load mpi/openmpi
module load cuda/12.0  # If using GPUs

# Set thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run with srun
srun --mpi=pmix ./my_dtl_app
```

### PBS/Torque Integration

```bash
#!/bin/bash
#PBS -N dtl_job
#PBS -l nodes=4:ppn=32
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR
module load mpi

mpirun -np 4 -hostfile $PBS_NODEFILE ./my_dtl_app
```

### LSF Integration

```bash
#!/bin/bash
#BSUB -J dtl_job
#BSUB -n 128
#BSUB -R "span[ptile=32]"
#BSUB -W 01:00

mpirun ./my_dtl_app
```

## GPU Cluster Deployment

### Multi-GPU per Node

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4

# One MPI rank per GPU
srun --gpus-per-task=1 ./my_dtl_gpu_app
```

### Application GPU Binding

```cpp
#include <dtl/dtl.hpp>
#include <backends/cuda/cuda_memory_space.hpp>

int main(int argc, char** argv) {
    dtl::mpi::scoped_init mpi(argc, argv);
    dtl::mpi::mpi_comm_adapter comm;

    // Bind to GPU based on local rank
    int local_rank = get_local_rank();  // Implementation-specific
    cudaSetDevice(local_rank);

    // Verify GPU binding
    int device;
    cudaGetDevice(&device);
    std::cout << "Rank " << comm.rank() << " using GPU " << device << std::endl;

    return 0;
}
```

### NCCL Initialization

```cpp
#include <backends/nccl/nccl_communicator.hpp>

// Initialize NCCL communicator
ncclUniqueId id;
if (comm.rank() == 0) {
    ncclGetUniqueId(&id);
}
comm.broadcast(&id, sizeof(id), 0);

ncclComm_t nccl_comm;
ncclCommInitRank(&nccl_comm, comm.size(), id, comm.rank());
```

## Performance Tuning

### MPI Tuning

```bash
# OpenMPI tuning
export OMPI_MCA_btl=openib,self,vader
export OMPI_MCA_mpi_yield_when_idle=0

# MPICH tuning
export MPICH_ASYNC_PROGRESS=1
export MPICH_MAX_THREAD_SAFETY=multiple
```

### NUMA Binding

```bash
# Bind processes to NUMA domains
numactl --cpunodebind=0 --membind=0 ./my_dtl_app

# Or via MPI
mpirun --bind-to socket ./my_dtl_app
```

### Thread Pool Sizing

```cpp
// Match threads to available cores
size_t num_threads = std::thread::hardware_concurrency();

// For MPI: divide by ranks per node
size_t threads_per_rank = num_threads / ranks_per_node;

dtl::cpu::cpu_executor exec(threads_per_rank);
```

## Monitoring and Debugging

### Runtime Checks

```cpp
// Enable DTL debug checks (compile with -DDTL_DEBUG)
#ifdef DTL_DEBUG
    // Additional validation enabled
#endif
```

### MPI Debugging

```bash
# Run with MPI debugging
mpirun -np 4 xterm -e gdb ./my_dtl_app

# Memory checking
mpirun -np 4 valgrind --tool=memcheck ./my_dtl_app
```

### CUDA Debugging

```bash
# CUDA memory check
compute-sanitizer --tool memcheck mpirun -np 2 ./my_dtl_gpu_app

# CUDA race detection
compute-sanitizer --tool racecheck mpirun -np 2 ./my_dtl_gpu_app
```

## Troubleshooting

### Common Issues

**MPI initialization fails:**
```
Solution: Ensure MPI is initialized before any DTL operations
```

**GPU not found:**
```bash
# Check CUDA visibility
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

**Memory allocation fails:**
```cpp
// Check available memory
size_t available = gpu_mem.available_memory();
if (allocation_size > available) {
    // Handle insufficient memory
}
```

**NCCL timeout:**
```bash
# Increase timeout
export NCCL_TIMEOUT=600000  # 10 minutes
```

### Debugging Hangs

```bash
# Attach debugger to hung processes
gdb -p <pid>

# MPI debugging with MPIBound
export MPIBIND_DEBUG=1
```

## Security Considerations

### Input Validation

```cpp
// Validate external input
if (input_size > MAX_ALLOWED_SIZE) {
    return dtl::error_code::invalid_argument;
}
```

### Memory Safety

- Use `result<T>` for error handling
- Check allocation results
- Avoid raw pointers where possible

### Network Security

- Use secure MPI transports where available
- Consider encrypted communication for sensitive data
- Validate rank identities in multi-tenant environments

## Health Checks

### Pre-Deployment Verification

```bash
# Run unit tests
cd build && ctest --output-on-failure

# Run MPI tests
mpirun -np 4 ./tests/dtl_mpi_tests

# Run example
mpirun -np 4 ./examples/hello_distributed
```

### Runtime Health

```cpp
// Check communicator health
if (!comm.is_valid()) {
    // Handle invalid communicator
}

// Verify collective participation
comm.barrier();  // Will hang if not all ranks participate
```

## See Also

- [Getting Started](getting_started.md) - Initial setup
- [Backend Selection](backend_guide/backend_selection.md) - Choosing backends
