# Backend Comparison

**Last Updated:** 2026-03-06

## Overview

DTL supports multiple backends that can be combined for heterogeneous
distributed computing. This table reflects the public, supported surface, not
internal experiments or partially implemented parity work.

## Backend Summary

| Backend | Purpose | Header Prefix | CMake Flag | Status |
|---------|---------|--------------|------------|--------|
| [CPU](cpu_guide.md) | Multi-threaded local execution | `backends/cpu/` | Always available | Production |
| [MPI](mpi_guide.md) | Distributed communication | `backends/mpi/` | `DTL_ENABLE_MPI` | Production |
| [CUDA](cuda_guide.md) | NVIDIA GPU execution | `backends/cuda/` | `DTL_ENABLE_CUDA` | Production |
| [HIP](hip_guide.md) | AMD GPU execution | `backends/hip/` | `DTL_ENABLE_HIP` | Production |
| [NCCL](nccl_backend.md) | Explicit GPU-native collectives | `backends/nccl/` | `DTL_ENABLE_NCCL` | Experimental |
| [OpenSHMEM](shmem_backend.md) | PGAS one-sided communication | `backends/shmem/` | `DTL_ENABLE_SHMEM` | Production |

## Execution Capabilities

| Feature | CPU | CUDA | HIP | MPI | NCCL | SHMEM |
|---------|-----|------|-----|-----|------|-------|
| Local parallel execution | ✅ | ✅ | ✅ | — | — | — |
| Thread pool | ✅ | — | — | — | — | — |
| Stream-based async | — | ✅ | ✅ | — | — | — |
| Kernel dispatch | — | ✅ | ✅ | — | — | — |
| Execution policies | `seq`/`par`/`async` | `on_stream` | `on_stream` | — | — | — |

## Communication Capabilities

| Feature | CPU | CUDA | HIP | MPI | NCCL | SHMEM |
|---------|-----|------|-----|-----|------|-------|
| Point-to-point | — | — | — | ✅ | ✅ device buffers only | ✅ |
| Broadcast | — | — | — | ✅ | ✅ device buffers only | ✅ |
| Reduce / Allreduce | — | — | — | ✅ | ✅ explicit device-buffer paths | ✅ |
| Gather / Scatter | — | — | — | ✅ | ✅ fixed-size device-buffer paths | — |
| All-to-all | — | — | — | ✅ | ✅ fixed-size device-buffer paths | — |
| Barrier | — | — | — | ✅ | ✅ explicit NCCL path | ✅ |
| Variable-size collectives | — | — | — | ✅ | — | — |
| Scan / Exscan | — | CUDA local algorithms only | HIP local algorithms only | ✅ | — | — |
| One-sided (RMA) | — | — | — | ✅ | — | ✅ |

## Memory Capabilities

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

## Common Backend Combinations

### MPI + CUDA

Use this for distributed GPU work when your generic algorithm path still needs
MPI semantics or host-resident coordination.

### MPI + CUDA + NCCL

Use this when you have an explicit NCCL communication path over CUDA
device-resident buffers. Do not assume a context with NCCL will implicitly
reroute generic distributed algorithms away from MPI.

## Decision Guide

```
Need distributed computing?
├── No → CPU backend
├── Yes
│   ├── Need generic distributed algorithms or host-buffer collectives?
│   │   └── MPI backend
│   ├── Need NVIDIA GPU execution?
│   │   ├── Local GPU algorithms → CUDA backend
│   │   └── Explicit GPU-native collectives on device buffers → CUDA + NCCL
│   ├── Need AMD GPU execution?
│   │   └── HIP backend
│   └── Need one-sided communication?
│       └── SHMEM or MPI RMA
```

## Notes on NCCL

- NCCL is explicit and device-buffer-only.
- NCCL is not the generic default communicator for contexts.
- Unsupported MPI-style helpers remain unsupported rather than being emulated
  with host-side scratch or scalar wrappers.

## See Also

- [CUDA Backend Guide](cuda_guide.md)
- [MPI Backend Guide](mpi_guide.md)
- [NCCL Backend](nccl_backend.md)
- [NCCL/CUDA Audit](nccl_cuda_audit.md)
