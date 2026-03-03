# Portable Device Code with `dtl::device::`

## Overview

DTL provides a vendor-agnostic `dtl::device::` namespace that lets you write GPU code once and run it on either NVIDIA (CUDA) or AMD (HIP) hardware. The backend is selected at compile time via CMake flags.

## Quick Start

```cpp
#include <dtl/device/device.hpp>

// Works with CUDA or HIP — no source changes needed
int num_gpus = dtl::device::device_count();

if (num_gpus > 0) {
    dtl::device::device_guard guard(0);  // RAII device selection
    dtl::device::device_buffer<float> buf(1024);
    dtl::device::fill_device(buf.data(), 1024, 0.0f);
    dtl::device::synchronize_device();
}
```

## Migration Guide

### Before (vendor-specific)

```cpp
#include <dtl/cuda/device_guard.hpp>
#include <dtl/cuda/device_buffer.hpp>
#include <dtl/cuda/cuda_algorithms.hpp>

dtl::cuda::device_guard guard(0);
dtl::cuda::device_buffer<float> buf(1024);
dtl::cuda::fill_device(buf.data(), 1024, 0.0f);
dtl::cuda::synchronize_device();
```

### After (portable)

```cpp
#include <dtl/device/device.hpp>

dtl::device::device_guard guard(0);
dtl::device::device_buffer<float> buf(1024);
dtl::device::fill_device(buf.data(), 1024, 0.0f);
dtl::device::synchronize_device();
```

The migration is a mechanical find-and-replace: `dtl::cuda::` to `dtl::device::`.

## Available Types

| Portable Type | CUDA Backend | HIP Backend |
|---------------|-------------|-------------|
| `device_guard` | `cuda::device_guard` | `hip::device_guard` |
| `stream_handle` | `cuda::stream_handle` | `hip::stream_handle` |
| `device_buffer<T>` | `cuda::device_buffer<T>` | `hip::device_buffer<T>` |
| `device_event` | `cuda::cuda_event` | `hip::hip_event` |
| `device_memory_space` | `cuda_device_memory_space` | `hip_memory_space` |
| `unified_memory_space` | `cuda_unified_memory_space` | `hip_managed_memory_space` |

## Available Functions

### Device Management

```cpp
int dtl::device::device_count();         // Number of GPUs
int dtl::device::current_device_id();    // Current GPU ID
constexpr int dtl::device::invalid_device_id;  // Sentinel value (-1)
```

### Device Algorithms (CUDA only)

```cpp
dtl::device::fill_device(ptr, count, value);
dtl::device::copy_device(src, dst, count);
dtl::device::sort_device(ptr, count);
dtl::device::reduce_sum_device(ptr, count);
dtl::device::for_each_device(ptr, count, func);
dtl::device::transform_device(in, out, count, op);
dtl::device::count_device(ptr, count, value);
dtl::device::find_device(ptr, count, value);
dtl::device::synchronize_stream(stream);
dtl::device::synchronize_device();
```

### Events

```cpp
auto event = dtl::device::make_sync_event();
auto timer = dtl::device::make_timing_event();
```

## Compile-Time Selection

The backend is selected by CMake:

```bash
# NVIDIA GPU
cmake .. -DDTL_ENABLE_CUDA=ON

# AMD GPU
cmake .. -DDTL_ENABLE_HIP=ON
```

When both are enabled, CUDA takes priority. When neither is enabled, stub implementations compile but report zero devices.

## Limitations

1. **Algorithm support**: Device algorithms (`fill_device`, `sort_device`, etc.) are currently CUDA-only. HIP algorithm parity is planned for a future release.

2. **Single-vendor model**: The current implementation selects one vendor at compile time. Mixed-vendor support (CUDA + HIP in the same process) is planned for V2.

3. **Vendor-specific features**: Features unique to one vendor (e.g., NCCL for NVIDIA, RCCL for AMD) must still use vendor-specific namespaces.

## Headers

| Header | Contents |
|--------|----------|
| `<dtl/device/device.hpp>` | Master header (includes all below) |
| `<dtl/device/device_query.hpp>` | `device_count()`, `current_device_id()` |
| `<dtl/device/device_guard.hpp>` | `device_guard` RAII class |
| `<dtl/device/stream.hpp>` | `stream_handle` class |
| `<dtl/device/buffer.hpp>` | `device_buffer<T>` template |
| `<dtl/device/event.hpp>` | `device_event`, `scoped_timer` |
| `<dtl/device/algorithms.hpp>` | GPU algorithm functions |
| `<dtl/device/memory_space.hpp>` | Memory space types |

