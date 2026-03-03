# HIP Backend Guide

**Status:** Production-Ready
**Since:** DTL 0.1.0-alpha.1
**Last Updated:** 2026-02-07

## Overview

The HIP (Heterogeneous-Compute Interface for Portability) backend enables DTL to run on AMD GPUs via the ROCm software stack. HIP provides a CUDA-like programming model, and DTL's HIP backend mirrors the CUDA backend's architecture with AMD-specific adaptations.

Key capabilities:

- **AMD GPU memory management** via `hip_memory_space`
- **Stream-based execution** via `hip_executor` for asynchronous kernel dispatch
- **API parity** with the CUDA backend for most DTL operations
- **Portability** — HIP code can also target NVIDIA GPUs as a compilation target

## Requirements

- **ROCm** 5.0 or later (6.x recommended)
- **AMD GPU** with GFX9 or later architecture (MI100, MI200, MI300, RX 7000 series)
- **C++20 compiler** (GCC 10+, Clang 14+ with ROCm support)
- **CMake** 3.21+ (for HIP language support)

## CMake Configuration

Enable the HIP backend at configure time:

```bash
cmake -DDTL_ENABLE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      ..
```

### Common CMake Flags

| Flag | Default | Description |
|------|---------|-------------|
| `DTL_ENABLE_HIP` | `OFF` | Enable HIP backend |
| `CMAKE_HIP_ARCHITECTURES` | Auto | Target GPU architectures (e.g., `gfx908;gfx90a;gfx942`) |

### ROCm Installation

Install ROCm following AMD's official documentation for your distro:

```bash
# Ubuntu 22.04+
sudo apt install rocm-dev hipcc
```

Verify the installation:

```bash
hipcc --version
rocminfo  # Lists available AMD GPUs
```

## CUDA to HIP Porting

### API Mapping

HIP mirrors the CUDA API with a `hip` prefix. DTL abstracts most of these differences behind its backend interfaces:

| CUDA | HIP | DTL Abstraction |
|------|-----|-----------------|
| `cudaMalloc` | `hipMalloc` | `hip_memory_space::allocate()` |
| `cudaFree` | `hipFree` | `hip_memory_space::deallocate()` |
| `cudaMemcpy` | `hipMemcpy` | `hip_memory_transfer::copy()` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | `hip_memory_transfer::copy_async()` |
| `cudaStreamCreate` | `hipStreamCreate` | `hip_stream` RAII wrapper |
| `cudaEventCreate` | `hipEventCreate` | `hip_event` RAII wrapper |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | `hip_executor::synchronize()` |
| `cudaGetDeviceCount` | `hipGetDeviceCount` | `dtl::hip::device_count()` |

### The hipify Tool

AMD provides `hipify-perl` and `hipify-clang` to automatically convert CUDA source to HIP:

```bash
# Convert a CUDA source file to HIP
hipify-perl my_kernel.cu > my_kernel.hip.cpp

# Or use the clang-based converter for more accuracy
hipify-clang my_kernel.cu -o my_kernel.hip.cpp
```

DTL's backend code uses `#if DTL_ENABLE_HIP` guards rather than hipified CUDA code, ensuring each backend is a clean, first-class implementation.

## Memory Management

### `hip_memory_space`

The `hip_memory_space` class manages device memory allocations on AMD GPUs. It satisfies DTL's `MemorySpace` concept.

```cpp
#include <backends/hip/hip_memory_space.hpp>

dtl::hip::hip_memory_space mem_space;

// Allocate device memory
auto alloc_result = mem_space.allocate(1024);
if (alloc_result.has_value()) {
    void* ptr = alloc_result.value();
    // Use device memory...
    mem_space.deallocate(ptr, 1024);
}
```

Properties:

- `host_accessible = false` — device memory is not directly host-accessible
- `device_accessible = true` — accessible from GPU kernels
- Supports per-device allocation

### Memory Transfers

```cpp
#include <backends/hip/hip_memory_transfer.hpp>

// Host to device
dtl::hip::hip_memory_transfer::copy(d_ptr, h_ptr, size,
    dtl::hip::transfer_kind::host_to_device);

// Device to host
dtl::hip::hip_memory_transfer::copy(h_ptr, d_ptr, size,
    dtl::hip::transfer_kind::device_to_host);

// Async copy on a stream
dtl::hip::hip_memory_transfer::copy_async(d_ptr, h_ptr, size,
    dtl::hip::transfer_kind::host_to_device, stream);
```

## Executor Patterns

### `hip_executor`

The `hip_executor` provides stream-based asynchronous execution on AMD GPUs:

```cpp
#include <backends/hip/hip_executor.hpp>

// Create executor with a non-blocking stream
dtl::hip::hip_stream stream(dtl::hip::stream_flags::non_blocking);
dtl::hip::hip_executor exec(std::move(stream));

// Execute work on the GPU
exec.execute([](hipStream_t s) {
    my_kernel<<<grid, block, 0, s>>>(d_data, n);
});

// Wait for completion
exec.synchronize();
```

### Stream Management

DTL provides RAII stream wrappers for HIP:

```cpp
#include <backends/hip/hip_executor.hpp>

// Default stream
dtl::hip::hip_stream default_stream;

// Non-blocking stream (owned)
dtl::hip::hip_stream stream(dtl::hip::stream_flags::non_blocking);

// Wrap an existing hipStream_t (non-owning)
hipStream_t external_stream;
hipStreamCreate(&external_stream);
dtl::hip::hip_stream wrapped(external_stream, false);  // does not destroy on destruct
```

### Event Tracking

```cpp
#include <backends/hip/hip_event.hpp>

dtl::hip::hip_event event;
event.record(stream);

// Check completion
if (event.query()) {
    // Work is done
}

// Or block until done
event.synchronize();
```

## DTL HIP-Specific Considerations

### Placement Policies with HIP

When HIP is enabled but CUDA is not, the placement policies map to HIP equivalents:

- `device_only<N>` — Allocates via `hipMalloc` on device N
- `unified_memory` — Uses `hipMallocManaged` for managed memory
- `device_preferred` — Uses unified memory with device-preferred hints

```cpp
// HIP device allocation
dtl::distributed_vector<float, dtl::device_only<0>> vec(1000, ctx);

// HIP unified memory
dtl::distributed_vector<float, dtl::unified_memory> vec_unified(1000, ctx);
```

### Build System

When building with HIP:

- Use `hipcc` as the compiler or set up CMake's HIP language support
- ROCm's `amdclang++` can compile HIP sources with `--offload-arch=<gfx...>`
- DTL detects HIP availability via `find_package(hip)` in CMake

### Error Handling

HIP errors are mapped to DTL status codes:

```cpp
// HIP errors translated to DTL status
dtl::status_code::hip_error  // = 520
```

## Known Differences from CUDA Backend

| Feature | CUDA Backend | HIP Backend |
|---------|-------------|-------------|
| Device memory | `cudaMalloc` | `hipMalloc` |
| Unified memory | `cudaMallocManaged` | `hipMallocManaged` |
| Streams | `cudaStream_t` | `hipStream_t` |
| Events | `cudaEvent_t` | `hipEvent_t` |
| Error type | `cudaError_t` | `hipError_t` |
| Status code | `status_code::cuda_error` (510) | `status_code::hip_error` (520) |
| Warp size | 32 | 64 (AMD GCN/CDNA) |
| Shared memory | 48KB (default) | 64KB (typical) |
| Architecture flag | `CMAKE_CUDA_ARCHITECTURES=80` | `CMAKE_HIP_ARCHITECTURES=gfx90a` |
| Progress engine | CUDA event-based | HIP event-based |
| NCCL integration | Yes | RCCL (ROCm equivalent) |

### Warp Size Differences

AMD GPUs use a wavefront size of 64 (vs. NVIDIA's warp size of 32). This affects:

- Warp-level primitives (`__ballot`, `__shfl`)
- Shared memory bank conflicts
- Occupancy calculations
- Reduction patterns within a warp/wavefront

### Performance Considerations

- AMD GPUs may have different optimal block sizes than NVIDIA GPUs
- Memory coalescing rules are similar but not identical
- Use `rocprof` instead of `nsight-compute` for profiling
- Prefer `hipcc` for compilation to ensure proper HIP runtime linkage

## See Also

- [CUDA Backend Guide](cuda_guide.md) — NVIDIA GPU support
- [NCCL Backend](nccl_backend.md) — GPU-to-GPU collectives (NVIDIA)
- [Backend Comparison](comparison.md) — Feature comparison across backends
- [AMD ROCm Documentation](https://rocm.docs.amd.com/) — Official ROCm docs
