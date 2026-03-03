# GPU Device Scoping

**Version:** 1.0.2
**Last Updated:** 2026-02-04

This document describes the device scoping rules for multi-device GPU usage in DTL, covering CUDA and HIP backends.

---

## Overview

When using multiple GPU devices in a single process, it is essential to correctly scope device operations to avoid "current device" surprises. DTL provides RAII device guards and container device affinity to ensure safe and predictable multi-device behavior.

### Key Principles

1. **Device guards must scope device-specific operations** - Memory allocation, deallocation, kernel launches, and stream operations must be wrapped in device guards.

2. **Container operations preserve caller's device context** - After any DTL container operation, the caller's current device is restored.

3. **Device affinity is stored at runtime** - Containers store their device ID and use guards internally for all device operations.

---

## Device Guards

### CUDA Device Guard

```cpp
#include <dtl/cuda/device_guard.hpp>

// Scope a block of operations to device 1
{
    dtl::cuda::device_guard guard(1);
    // All CUDA operations here target device 1
    cudaMalloc(&ptr, size);
    kernel<<<grid, block>>>(ptr);
}
// Previous device is automatically restored
```

### HIP Device Guard

```cpp
#include <dtl/hip/device_guard.hpp>

// Scope a block of operations to device 0
{
    dtl::hip::device_guard guard(0);
    // All HIP operations here target device 0
    hipMalloc(&ptr, size);
}
// Previous device is automatically restored
```

### Guard Behavior

| Scenario | Behavior |
|----------|----------|
| Guard to same device | No-op (no device switch needed) |
| Guard with invalid device ID (-1) | No-op |
| Nested guards | Each guard restores to its predecessor |
| Exception thrown | Destructor still restores device (RAII) |

---

## Operations Requiring Device Guards

### Memory Allocation

All device memory allocations must be guarded:

```cpp
void* allocate_on_device(int device_id, size_t size) {
    dtl::cuda::device_guard guard(device_id);
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}
```

DTL's memory spaces handle this internally:

```cpp
// cuda_device_memory_space_for<DeviceId> guards internally
void* ptr = cuda_device_memory_space_for<1>::allocate(size);
```

### Memory Deallocation

Freeing device memory should be done on the same device:

```cpp
void deallocate_on_device(int device_id, void* ptr) {
    dtl::cuda::device_guard guard(device_id);
    cudaFree(ptr);
}
```

### Kernel Launches

Launch kernels in the context of the correct device:

```cpp
void launch_on_device(int device_id, float* data, size_t n) {
    dtl::cuda::device_guard guard(device_id);
    kernel<<<grid, block>>>(data, n);
}
```

### Stream Operations

Create and synchronize streams on the correct device:

```cpp
cudaStream_t create_stream_on_device(int device_id) {
    dtl::cuda::device_guard guard(device_id);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return stream;
}
```

---

## Container Device Affinity

### Compile-Time Device Selection

Use `device_only<N>` for compile-time device specification:

```cpp
// Container on device 0
dtl::distributed_vector<float, dtl::device_only<0>> vec0(1000, ctx);

// Container on device 1 (different type!)
dtl::distributed_vector<float, dtl::device_only<1>> vec1(1000, ctx);

// Query device affinity
assert(vec0.device_id() == 0);
assert(vec1.device_id() == 1);
```

### Runtime Device Selection

Use `device_only_runtime` for runtime device specification:

```cpp
int gpu_id = select_gpu();  // Runtime value
auto gpu_ctx = base_ctx.with_cuda(gpu_id);

// Container on runtime-selected device
dtl::distributed_vector<float, dtl::device_only_runtime> vec(1000, gpu_ctx);

assert(vec.device_id() == gpu_id);
```

### Container Invariants

1. **Device ID is immutable after construction** - A container's device affinity cannot change.

2. **Internal operations are guarded** - Container methods that touch device memory use guards internally.

3. **Caller's context is preserved** - After calling any container method, the caller's current device is unchanged.

```cpp
int original = dtl::cuda::current_device_id();

{
    dtl::distributed_vector<float, dtl::device_only<1>> vec(100, ctx);
    // vec's internal allocations happened on device 1
    // but original device is already restored
}

assert(dtl::cuda::current_device_id() == original);
```

---

## Multi-Threaded Device Scoping

### Thread-Local Device Context

CUDA/HIP device context is thread-local. Each thread maintains its own "current device":

```cpp
void worker(int device_id) {
    dtl::cuda::device_guard guard(device_id);
    // This thread's operations target device_id
    // Other threads are unaffected
}

std::thread t0(worker, 0);
std::thread t1(worker, 1);
t0.join();
t1.join();
```

### Thread Safety Rules

1. **Guards are thread-safe** - Each thread's guard operates independently.

2. **Memory spaces are thread-safe** - Device memory spaces use guards internally.

3. **Container construction is thread-safe** - Multiple threads can construct containers targeting different devices.

4. **No shared mutable state** - Guards do not share state between threads.

### Best Practices for Multi-Threaded Code

```cpp
// GOOD: Each thread guards its own operations
void process_on_device(int device_id, float* data, size_t n) {
    dtl::cuda::device_guard guard(device_id);
    process_kernel<<<grid, block>>>(data, n);
    cudaStreamSynchronize(0);
}

// GOOD: Create containers with explicit device context
void create_per_thread_container(int device_id) {
    auto ctx = make_cpu_context().with_cuda(device_id);
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(1000, ctx);
    // vec is on device_id
}

// BAD: Relying on global cudaSetDevice without guards
void unsafe_operation(int device_id) {
    cudaSetDevice(device_id);  // Another thread might change this!
    cudaMalloc(&ptr, size);    // Which device? Unknown!
}
```

---

## Multi-Device Patterns

### Pattern 1: Multiple Contexts

Create separate contexts for each device:

```cpp
auto ctx0 = base_ctx.with_cuda(0);
auto ctx1 = base_ctx.with_cuda(1);

dtl::distributed_vector<float, dtl::device_only_runtime> vec0(1000, ctx0);
dtl::distributed_vector<float, dtl::device_only_runtime> vec1(1000, ctx1);
```

### Pattern 2: Device Pool Domain

Use a device pool for advanced multi-GPU:

```cpp
auto pool = dtl::cuda_device_pool_domain::create({0, 1});
if (pool) {
    pool->synchronize(0);
    pool->synchronize(1);
    pool->synchronize_all();
}
```

### Pattern 3: Per-Rank Device Assignment

Assign devices to MPI ranks:

```cpp
int num_gpus = dtl::cuda::device_count();
int my_device = ctx.rank() % num_gpus;
auto gpu_ctx = ctx.with_cuda(my_device);
```

---

## Debugging Device Issues

### Check Current Device

```cpp
int current = dtl::cuda::current_device_id();
std::cout << "Current device: " << current << "\n";
```

### Query Pointer Location

```cpp
int device = dtl::cuda::get_pointer_device(ptr);
if (device >= 0) {
    std::cout << "Pointer is on device " << device << "\n";
} else {
    std::cout << "Pointer is not device memory\n";
}
```

### Verify Container Affinity

```cpp
dtl::distributed_vector<float, dtl::device_only<1>> vec(100, ctx);
assert(vec.device_id() == 1);
assert(vec.has_device_affinity());
```

---

## Common Pitfalls

### Pitfall 1: Forgetting to Guard

```cpp
// BAD: No guard, uses whatever device is "current"
cudaMalloc(&ptr, size);

// GOOD: Explicit guard
{
    dtl::cuda::device_guard guard(target_device);
    cudaMalloc(&ptr, size);
}
```

### Pitfall 2: Freeing on Wrong Device

```cpp
// Allocated on device 0
{
    dtl::cuda::device_guard guard(0);
    cudaMalloc(&ptr, size);
}

// BAD: Freeing on device 1
cudaSetDevice(1);
cudaFree(ptr);  // Undefined behavior!

// GOOD: Free on same device
{
    dtl::cuda::device_guard guard(0);
    cudaFree(ptr);
}
```

### Pitfall 3: Assuming Device Context Persists

```cpp
// BAD: Assuming device 0 is still current after function call
cudaSetDevice(0);
process_data(data, n);  // Might change device internally
cudaMalloc(&ptr, size);  // Which device?

// GOOD: Use guard for each scope
{
    dtl::cuda::device_guard guard(0);
    process_data(data, n);
}
{
    dtl::cuda::device_guard guard(0);  // Explicit scope
    cudaMalloc(&ptr, size);
}
```

---

## See Also

- [Device Guard Headers](../include/dtl/cuda/device_guard.hpp)
- [Device Affinity Utilities](../include/dtl/containers/detail/device_affinity.hpp)
- [CUDA Device Pool Domain](../include/dtl/core/cuda_device_pool_domain.hpp)
