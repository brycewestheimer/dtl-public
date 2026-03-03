# CUDA Backend Guide

**Status:** Production-Ready
**Since:** DTL 0.1.0-alpha.1
**Last Updated:** 2026-02-07

## Overview

The CUDA backend enables DTL containers and algorithms to run on NVIDIA GPUs. It provides device memory allocation, stream-based asynchronous kernel dispatch, and integration with DTL's placement policy system. When combined with the MPI backend, DTL supports multi-node, multi-GPU distributed computing.

Key capabilities:

- **Device memory management** via `cuda_memory_space` and `cuda_device_memory_space`
- **Unified (managed) memory** with automatic page migration
- **Stream-based execution** via `cuda_executor` for asynchronous kernel dispatch
- **Event-based progress tracking** integrated with DTL's `distributed_future<T>`
- **Placement policies** (`device_only`, `unified_memory`, `device_preferred`) for controlling data residency

## Requirements

- **CUDA Toolkit** 11.0 or later (12.x recommended)
- **NVIDIA GPU** with Compute Capability 7.0+ (Volta or later recommended)
- **C++20 compiler** with CUDA support (GCC 10+, Clang 12+, or NVCC)
- **CMake** 3.18+ (for CUDA language support)

## CMake Configuration

Enable the CUDA backend at configure time:

```bash
cmake -DDTL_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      ..
```

### Common CMake Flags

| Flag | Default | Description |
|------|---------|-------------|
| `DTL_ENABLE_CUDA` | `OFF` | Enable CUDA backend |
| `CMAKE_CUDA_ARCHITECTURES` | Auto | Target GPU architectures (e.g., `70;80;90`) |
| `DTL_CUDA_MEMORY_POOL` | `OFF` | Enable CUDA memory pool allocator |

### Verifying CUDA Support

After building, check that CUDA is available at runtime:

```cpp
#include <dtl/core/config.hpp>
#include <iostream>

int main() {
#if DTL_ENABLE_CUDA
    std::cout << "CUDA backend enabled\n";
#else
    std::cout << "CUDA backend not available\n";
#endif
}
```

## Placement Policies

DTL uses placement policies to control where container data resides. Three CUDA-aware policies are available:

### `device_only<DeviceId>`

Allocates memory exclusively on the specified GPU device. Data is **not** accessible from the host CPU without explicit copy.

```cpp
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/placement/device_only.hpp>

// Allocate on GPU 0
dtl::distributed_vector<float, dtl::device_only<0>> vec(1000, ctx);

// Allocate on GPU 1 (different type!)
dtl::distributed_vector<float, dtl::device_only<1>> vec1(1000, ctx);

// local_view() is NOT available — data is on device only
// Use GPU algorithms or copy to host first
```

**When to use:** Maximum GPU performance; data lives entirely on the GPU and is processed by GPU kernels.

### `unified_memory`

Allocates CUDA unified (managed) memory accessible from both host and device. The CUDA runtime automatically migrates pages between CPU and GPU as needed.

```cpp
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/placement/unified_memory.hpp>

dtl::distributed_vector<float, dtl::unified_memory> vec(1000, ctx);

// Accessible from host
auto local = vec.local_view();
for (auto& elem : local) {
    elem = 1.0f;
}

// Also accessible from GPU kernels (with automatic page migration)
```

**When to use:** Prototyping, mixed host/device access patterns, or when data access patterns are irregular and hard to predict.

### `device_preferred`

Allocates unified memory with a device-preferred hint. Data resides primarily on the GPU but can be accessed from the host with automatic migration.

```cpp
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/placement/device_preferred.hpp>

dtl::distributed_vector<float, dtl::device_preferred> vec(1000, ctx);

// Hint that data should reside on GPU, but host access is possible
```

**When to use:** GPU-heavy workloads with occasional host access (e.g., for I/O or checkpointing).

### Placement Comparison

| Policy | Host Access | Device Access | Migration | Best For |
|--------|------------|---------------|-----------|----------|
| `host_only` | ✅ Direct | ❌ Copy needed | None | CPU-only workloads |
| `device_only<N>` | ❌ Copy needed | ✅ Direct | None | Pure GPU compute |
| `unified_memory` | ✅ Automatic | ✅ Automatic | Page-level | Mixed access patterns |
| `device_preferred` | ✅ Automatic | ✅ Preferred | Page-level | GPU-heavy with occasional host |

## Memory Management

### `cuda_memory_space`

The `cuda_memory_space` class manages device memory allocations. It satisfies DTL's `MemorySpace` concept.

```cpp
#include <backends/cuda/cuda_memory_space.hpp>

dtl::cuda::cuda_memory_space mem_space;

// Allocate 1024 bytes on current device
auto alloc_result = mem_space.allocate(1024);
if (alloc_result.has_value()) {
    void* ptr = alloc_result.value();
    // Use device memory...
    mem_space.deallocate(ptr, 1024);
}
```

Key properties:

- `host_accessible = false` — device memory is not directly host-accessible
- `device_accessible = true` — accessible from GPU kernels
- Default alignment: 256 bytes (CUDA standard)
- Supports per-device allocation via constructor parameter

### `cuda_unified_memory_space`

For managed memory accessible from both host and device:

```cpp
#include <dtl/memory/cuda_memory_space.hpp>

// Static interface for allocator integration
void* ptr = dtl::cuda::cuda_unified_memory_space::allocate(1024);
// Accessible from both host and device
dtl::cuda::cuda_unified_memory_space::deallocate(ptr, 1024);
```

### Memory Transfers

Use `cuda_memory_transfer` for explicit host-device copies:

```cpp
#include <backends/cuda/cuda_memory_transfer.hpp>

// Host to device
dtl::cuda::cuda_memory_transfer::copy(d_ptr, h_ptr, size,
    dtl::cuda::transfer_kind::host_to_device);

// Device to host
dtl::cuda::cuda_memory_transfer::copy(h_ptr, d_ptr, size,
    dtl::cuda::transfer_kind::device_to_host);

// Async copy on a stream
dtl::cuda::cuda_memory_transfer::copy_async(d_ptr, h_ptr, size,
    dtl::cuda::transfer_kind::host_to_device, stream);
```

## Executor Patterns

### `cuda_executor`

The `cuda_executor` provides stream-based asynchronous execution for GPU work. It integrates with DTL's futures system for event-based completion tracking.

```cpp
#include <backends/cuda/cuda_executor.hpp>

// Dispatch GPU work asynchronously
auto future = dtl::cuda::dispatch_gpu_async(stream, [](cudaStream_t s) {
    // Launch your CUDA kernel on stream s
    my_kernel<<<grid, block, 0, s>>>(d_data, n);
});

// Future resolves when GPU work completes
future.get();
```

### Retrieving Results from GPU

For operations that produce a result value on the device:

```cpp
// Dispatch and retrieve a scalar result
auto future = dtl::cuda::dispatch_gpu_async_result<float>(
    stream,
    [](cudaStream_t s) {
        // Launch reduction kernel that writes result to d_result
        reduce_kernel<<<grid, block, 0, s>>>(d_data, d_result, n);
    },
    d_result  // Device pointer to the result
);

float result = future.get().value();
```

### Stream Management

DTL provides RAII stream wrappers:

```cpp
#include <backends/cuda/cuda_executor.hpp>

// Create a non-blocking stream
dtl::cuda::cuda_stream stream(dtl::cuda::stream_flags::non_blocking);

// Use with executor
cuda_executor exec(std::move(stream));
exec.execute([](cudaStream_t s) {
    my_kernel<<<1, 256, 0, s>>>(data, n);
});
exec.synchronize();
```

## Performance Tips

### Memory Coalescing

Ensure threads in a warp access contiguous memory addresses:

```cpp
// Good: coalesced access (thread i accesses element i)
__global__ void good_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

// Bad: strided access (thread i accesses element i*stride)
__global__ void bad_kernel(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) data[idx] *= 2.0f;
}
```

### Occupancy

Choose block sizes that maximize GPU occupancy:

```cpp
int min_grid_size, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, my_kernel, 0, 0);

int grid_size = (n + block_size - 1) / block_size;
my_kernel<<<grid_size, block_size, 0, stream>>>(data, n);
```

### Async Transfers

Overlap computation with data transfers using streams:

```cpp
// Use pinned memory for async transfers
dtl::distributed_vector<float, dtl::unified_memory> vec(n, ctx);

// Prefetch to device before kernel launch
cudaMemPrefetchAsync(vec.local_data(), vec.local_size() * sizeof(float),
                     device_id, stream);

// Launch kernel (may overlap with prefetch)
my_kernel<<<grid, block, 0, stream>>>(vec.local_data(), vec.local_size());
```

### Minimize Host-Device Synchronization

Avoid unnecessary `cudaDeviceSynchronize()`. Use DTL's future-based async model instead:

```cpp
// Prefer: non-blocking dispatch
auto future = dtl::cuda::dispatch_gpu_async(stream, kernel_launcher);
// ... do other work ...
future.get();  // Block only when result is needed

// Avoid: blocking synchronization after every kernel
kernel<<<grid, block>>>(data, n);
cudaDeviceSynchronize();  // Stalls CPU
```

## Known Limitations

### WSL2 Considerations

- Unified memory is supported but may have reduced performance due to the virtualization layer
- `cudaMemPrefetchAsync` may not migrate pages as effectively
- Multi-GPU configurations may not be fully supported under WSL2
- Recommendation: Use `device_only` placement for best performance on WSL2

### Multi-GPU

- `device_only<N>` uses compile-time device selection; each device ID produces a **different type**
- Cross-device copies require explicit memory transfers
- For runtime device selection, use `device_only_runtime` (requires DTL 1.1+)
- Peer-to-peer (P2P) access between GPUs must be explicitly enabled

### General

- CUDA backend requires NVIDIA GPUs; for AMD GPUs, use the [HIP backend](hip_guide.md)
- GPU algorithms require the algorithm dispatch infrastructure — not all STL algorithms have GPU equivalents
- Error handling from CUDA API calls is wrapped in `dtl::result<T>` with `status_code::cuda_error`

## See Also

- [HIP Backend Guide](hip_guide.md) — AMD GPU support
- [NCCL Backend](nccl_backend.md) — GPU-to-GPU collective communication
- [Backend Comparison](comparison.md) — Feature comparison across backends
