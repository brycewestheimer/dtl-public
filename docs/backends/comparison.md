# Backend Comparison

**Last Updated:** 2026-02-07

## Overview

DTL supports multiple backends that can be combined for heterogeneous distributed computing. This document compares their capabilities, use cases, and trade-offs.

## Backend Summary

| Backend | Purpose | Header Prefix | CMake Flag | Status |
|---------|---------|--------------|------------|--------|
| [CPU](cpu_guide.md) | Multi-threaded local execution | `backends/cpu/` | Always available | Production |
| [MPI](mpi_guide.md) | Distributed communication | `backends/mpi/` | `DTL_ENABLE_MPI` | Production |
| [CUDA](cuda_guide.md) | NVIDIA GPU execution | `backends/cuda/` | `DTL_ENABLE_CUDA` | Production |
| [HIP](hip_guide.md) | AMD GPU execution | `backends/hip/` | `DTL_ENABLE_HIP` | Production |
| [NCCL](nccl_backend.md) | GPU-to-GPU collectives | `backends/nccl/` | `DTL_ENABLE_NCCL` | Production |
| [OpenSHMEM](shmem_backend.md) | PGAS one-sided communication | `backends/shmem/` | `DTL_ENABLE_SHMEM` | Production |

## Feature Comparison

### Execution Capabilities

| Feature | CPU | CUDA | HIP | MPI | NCCL | SHMEM |
|---------|-----|------|-----|-----|------|-------|
| Local parallel execution | ✅ | ✅ | ✅ | — | — | — |
| Thread pool | ✅ | — | — | — | — | — |
| Stream-based async | — | ✅ | ✅ | — | — | — |
| Kernel dispatch | — | ✅ | ✅ | — | — | — |
| Execution policies | `seq`/`par`/`async` | `on_stream` | `on_stream` | — | — | — |

### Communication Capabilities

| Feature | CPU | CUDA | HIP | MPI | NCCL | SHMEM |
|---------|-----|------|-----|-----|------|-------|
| Point-to-point | — | — | — | ✅ | ✅ (2.7+) | ✅ |
| Broadcast | — | — | — | ✅ | ✅ | ✅ |
| Reduce / Allreduce | — | — | — | ✅ | ✅ | ✅ |
| Gather / Scatter | — | — | — | ✅ | — | — |
| All-to-all | — | — | — | ✅ | — | — |
| Barrier | — | — | — | ✅ | — | ✅ |
| One-sided (RMA) | — | — | — | ✅ | — | ✅ |
| Atomic operations | — | ✅ | ✅ | ✅ | — | ✅ |

### Memory Capabilities

| Feature | CPU | CUDA | HIP | MPI | NCCL | SHMEM |
|---------|-----|------|-----|-----|------|-------|
| Host memory | ✅ | — | — | ✅ | — | ✅ |
| Device memory | — | ✅ | ✅ | — | ✅ | — |
| Unified memory | — | ✅ | ✅ | — | — | — |
| Pinned memory | — | ✅ | ✅ | — | — | — |
| Symmetric memory | — | — | — | — | — | ✅ |
| RMA windows | — | — | — | ✅ | — | ✅ |

## Placement Policy Support

| Placement | CPU | CUDA | HIP |
|-----------|-----|------|-----|
| `host_only` | ✅ Default | ✅ | ✅ |
| `device_only<N>` | — | ✅ | ✅ |
| `device_only_runtime` | — | ✅ | ✅ |
| `unified_memory` | — | ✅ | ✅ |
| `device_preferred` | — | ✅ | ✅ |
| `explicit_placement` | ✅ | ✅ | ✅ |

## Status Code Ranges

Each backend has its own error code range in `dtl::status_code`:

| Backend | Code Range | Key Code |
|---------|-----------|----------|
| Communication (MPI) | 100–199 | `mpi_error` = 530 |
| Memory | 200–299 | `memory_error` = 200 |
| Backend (generic) | 500–599 | `backend_error` = 500 |
| CUDA | — | `cuda_error` = 510 |
| HIP | — | `hip_error` = 520 |
| MPI | — | `mpi_error` = 530 |
| NCCL | — | `nccl_error` = 540 |
| SHMEM | — | `shmem_error` = 550 |

## Common Backend Combinations

### CPU-Only

No GPU, no MPI. Single-process, multi-threaded.

```bash
cmake -DDTL_ENABLE_MPI=OFF ..
```

```cpp
dtl::environment env;
dtl::distributed_vector<double> vec(10000, 1, 0);
dtl::for_each(dtl::par{}, vec, [](double& x) { x *= 2.0; });
```

### MPI + CPU

Multi-process distributed computing on CPUs.

```bash
cmake -DDTL_ENABLE_MPI=ON ..
```

```cpp
dtl::environment env(argc, argv);
auto comm = dtl::world_comm();
dtl::distributed_vector<int> vec(100000, comm.size(), comm.rank());
auto sum = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{}, comm);
```

### MPI + CUDA

Multi-node, multi-GPU with MPI for inter-node communication.

```bash
cmake -DDTL_ENABLE_MPI=ON -DDTL_ENABLE_CUDA=ON ..
```

```cpp
dtl::environment env(argc, argv);
auto comm = dtl::world_comm();
dtl::distributed_vector<float, dtl::device_only<0>> vec(1000000, comm.size(), comm.rank());
```

### MPI + CUDA + NCCL

Multi-GPU with optimized GPU-to-GPU collectives.

```bash
cmake -DDTL_ENABLE_MPI=ON -DDTL_ENABLE_CUDA=ON -DDTL_ENABLE_NCCL=ON ..
```

### MPI + HIP

Multi-node with AMD GPUs.

```bash
cmake -DDTL_ENABLE_MPI=ON -DDTL_ENABLE_HIP=ON ..
```

### MPI + SHMEM

Hybrid MPI + PGAS programming.

```bash
cmake -DDTL_ENABLE_MPI=ON -DDTL_ENABLE_SHMEM=ON ..
```

## Performance Characteristics

| Aspect | CPU | CUDA/HIP | MPI | NCCL | SHMEM |
|--------|-----|----------|-----|------|-------|
| Latency | Low | Kernel launch overhead | Network-dependent | Low (GPU-direct) | Low (RDMA) |
| Throughput | Memory bandwidth limited | High (parallel cores) | Network bandwidth | NVLink/PCIe | Fabric-dependent |
| Scalability | Single node | Single node (multi-GPU) | Multi-node | Multi-GPU | Multi-node |
| Best data size | Any | Large (amortize launch) | Any | Large | Any |

## Decision Guide

```
Need distributed computing?
├── No → CPU backend (seq/par/async policies)
├── Yes
│   ├── Using NVIDIA GPUs?
│   │   ├── Single GPU → CUDA backend
│   │   ├── Multi-GPU, same node → CUDA + NCCL
│   │   └── Multi-node, multi-GPU → MPI + CUDA + NCCL
│   ├── Using AMD GPUs?
│   │   ├── Single GPU → HIP backend
│   │   └── Multi-node → MPI + HIP
│   ├── CPU-only cluster?
│   │   └── MPI backend
│   └── Need one-sided communication?
│       └── SHMEM or MPI RMA
```

## See Also

- [CPU Backend Guide](cpu_guide.md)
- [CUDA Backend Guide](cuda_guide.md)
- [HIP Backend Guide](hip_guide.md)
- [MPI Backend Guide](mpi_guide.md)
- [NCCL Backend](nccl_backend.md)
- [OpenSHMEM Backend](shmem_backend.md)
