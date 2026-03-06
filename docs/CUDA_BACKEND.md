# CUDA Backend

DTL supports CUDA device memory and unified memory placements for distributed containers.

## Placement Semantics

### Host (`DTL_PLACEMENT_HOST` / `dtl::host_only`)
- Data stored in CPU memory via `std::vector<T>`
- Full host access via `local_view()`, `local_data()`
- No device access

### Device (`DTL_PLACEMENT_DEVICE` / `dtl::device_only_runtime`)
- Data stored in CUDA device memory via `cudaMalloc`
- **No host access**: `local_data()` returns `nullptr`
- Device access via `device_view()`, `device_data()`
- Data transfer via `copy_to_host()` / `copy_from_host()` (uses `cudaMemcpy`)
- Device ID extracted from context at construction time

### Unified (`DTL_PLACEMENT_UNIFIED` / `dtl::unified_memory`)
- Data stored in CUDA managed memory via `cudaMallocManaged`
- **Host-accessible**: `local_data()` returns valid managed pointer
- **Device-accessible**: `device_data()` returns same managed pointer
- No explicit copy needed; use `cudaDeviceSynchronize()` for coherence
- `copy_to_host()` / `copy_from_host()` use `memcpy` (already host-accessible)

### Device Preferred (`DTL_PLACEMENT_DEVICE_PREFERRED`)
- **Not supported**: Returns `DTL_ERROR_NOT_SUPPORTED`

## Host Access Rules

| Operation | Host | Device | Unified |
|---|---|---|---|
| `local_data()` | pointer | `nullptr` | pointer |
| `device_data()` | `nullptr` | pointer | pointer |
| `copy_to_host()` | memcpy | cudaMemcpy D->H | memcpy |
| `copy_from_host()` | memcpy | cudaMemcpy H->D | memcpy |
| `fill()` | std::fill | host staging + cudaMemcpy | std::fill |
| `reduce_*()` | direct | host staging + reduce | direct |
| `sort_*()` | std::sort | host staging + sort | std::sort |
| ND tensor access | supported | NOT_SUPPORTED | supported |
| `reshape()` (tensor) | supported | NOT_SUPPORTED | supported |

## Copy/View APIs

### C API
```c
// Copy helpers (work for all placements)
dtl_vector_copy_to_host(vec, host_buffer, count);
dtl_vector_copy_from_host(vec, host_buffer, count);

// Direct access (returns nullptr for incompatible placement)
const void* dtl_vector_local_data(vec);      // nullptr for device
void*       dtl_vector_device_data(vec);     // nullptr for host
```

### C++ API
```cpp
// Host access (requires host-accessible placement)
auto lv = vec.local_view();       // Compile-time gated
auto* ptr = vec.local_data();

// Device access (requires device-accessible placement)
auto dv = vec.device_view();      // Compile-time gated
```

## Context Requirements

Device and unified placements require a context with the `HAS_CUDA` domain flag.
Without it, container creation returns `DTL_ERROR_BACKEND_UNAVAILABLE`.

### C API
```c
dtl_context_t ctx;
// Context must have HAS_CUDA in domain_flags
opts.placement = DTL_PLACEMENT_DEVICE;
opts.device_id = 0;
dtl_vector_create_with_options(ctx, dtype, size, &opts, &vec);
```

### C++ API
```cpp
// Context must satisfy cuda_domain requirements
auto ctx = make_context_with_cuda(device_id);
dtl::distributed_vector<float, dtl::device_only_runtime> vec(1000, ctx);
```

## Element Type Requirements

All elements used with device/unified placements must be trivially copyable (`DeviceStorable` concept). All 10 C ABI dtypes (int8 through float64) satisfy this requirement.
