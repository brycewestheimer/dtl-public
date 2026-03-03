# Device Placement Semantics for Language Bindings

**Status:** Canonical Reference  
**Version:** 1.0.2  
**Last Updated:** 2026-02-04  
**Applies To:** C ABI, Python, Fortran bindings

---

## Overview

This document specifies how DTL language bindings must handle containers with different placement policies. It is the authoritative reference for binding implementers and documents the contract between C++ core and language wrappers.

## Placement Policy Categories

DTL supports three categories of memory placement:

| Placement | Host Accessible | Device Accessible | Binding Strategy |
|-----------|-----------------|-------------------|------------------|
| `host_only` | ✅ Yes | ❌ No | Direct views (zero-copy) |
| `unified_memory` | ✅ Yes | ✅ Yes | Direct views (with caveats) |
| `device_only<N>` / `device_only_runtime` | ❌ No | ✅ Yes | Explicit copies required |

---

## API Behavior by Placement

### 1. Host Placement (`host_only`)

**C++ Behavior:**
- `local_view()` returns `dtl::local_view<T>` with raw pointer iterators
- Full STL algorithm compatibility
- No special handling required

**C ABI Behavior:**
```c
// Returns pointer to host memory - safe to use directly
void* dtl_vector_local_data_mut(dtl_vector_handle* vec);

// Pointer can be passed to memcpy, used in loops, etc.
int* data = (int*)dtl_vector_local_data_mut(vec);
for (size_t i = 0; i < size; ++i) {
    data[i] = i;  // Safe: host memory
}
```

**Python Behavior:**
```python
# local_view() returns a NumPy array (zero-copy)
arr = vec.local_view()
arr[0] = 42  # Direct modification, no copy

# .to_numpy() is equivalent
np_arr = vec.to_numpy()
```

**Fortran Behavior:**
```fortran
! Pointer association works directly
type(c_ptr) :: data_ptr
data_ptr = dtl_vector_local_data_f(vec)
call c_f_pointer(data_ptr, data_array, [size])
data_array(1) = 42  ! Safe: host memory
```

---

### 2. Unified Memory (`unified_memory`)

**C++ Behavior:**
- `local_view()` returns `dtl::local_view<T>` with host-accessible pointers
- Memory is accessible from both host and device
- GPU access may trigger page faults/migrations (performance consideration)

**C ABI Behavior:**
```c
// Returns pointer to managed memory - host accessible
void* dtl_vector_local_data_mut(dtl_vector_handle* vec);

// IMPORTANT: Synchronize before host access after GPU operations
dtl_barrier(ctx);
int* data = (int*)dtl_vector_local_data_mut(vec);
// Now safe to read/write from host
```

**Python Behavior:**
```python
# local_view() returns NumPy array (zero-copy, managed memory)
arr = vec.local_view()

# WARNING: After GPU operations, synchronize before host access
ctx.barrier()  # Or dtl.synchronize()
print(arr[0])  # Now safe to read
```

**Performance Notes:**
- Unified memory may have lower bandwidth than explicit device memory
- Page faults on first access can cause latency spikes
- Consider prefetching hints for large data access patterns

---

### 3. Device-Only Placement (`device_only<N>`, `device_only_runtime`)

**C++ Behavior:**
- `local_view()` is **NOT available** (compile-time constraint or runtime error)
- `device_view()` returns `dtl::device_view<T>` with device pointer
- All host access requires explicit copies

**C ABI Behavior:**
```c
// dtl_vector_local_data_mut returns NULL for device-only containers
void* ptr = dtl_vector_local_data_mut(vec);
assert(ptr == NULL);  // Expected for device-only

// Use explicit copy functions instead
int* host_buffer = malloc(size * sizeof(int));
dtl_error err = dtl_vector_copy_to_host(vec, host_buffer, size);
if (err != DTL_SUCCESS) {
    // Handle error
}

// Modify on host, then copy back
host_buffer[0] = 42;
err = dtl_vector_copy_from_host(vec, host_buffer, size);
```

**C ABI Functions for Device Containers:**
```c
// Query placement capability
bool dtl_vector_is_host_accessible(const dtl_vector_handle* vec);
bool dtl_vector_is_device_accessible(const dtl_vector_handle* vec);

// Explicit copy operations
dtl_error dtl_vector_copy_to_host(
    const dtl_vector_handle* vec,
    void* host_dest,
    size_t count
);

dtl_error dtl_vector_copy_from_host(
    dtl_vector_handle* vec,
    const void* host_src,
    size_t count
);

// Async variants (with stream)
dtl_error dtl_vector_copy_to_host_async(
    const dtl_vector_handle* vec,
    void* host_dest,
    size_t count,
    dtl_stream_handle stream
);
```

**Python Behavior:**
```python
# local_view() raises an error for device-only containers
try:
    arr = vec.local_view()  # Raises RuntimeError
except RuntimeError as e:
    print("Cannot create host view of device-only container")

# Use explicit copy methods instead
host_data = vec.copy_to_host()  # Returns new NumPy array
print(host_data[0])

# Modify and copy back
host_data[0] = 42
vec.copy_from_host(host_data)

# Or use CuPy for direct GPU access (optional integration)
try:
    import cupy as cp
    cupy_arr = vec.to_cupy()  # Returns CuPy array view
    cupy_arr[0] = 42  # GPU operation
except ImportError:
    pass  # CuPy not available
```

**Python API Design:**
| Method | Host Placement | Unified | Device-Only |
|--------|----------------|---------|-------------|
| `local_view()` | NumPy array (view) | NumPy array (view) | ❌ RuntimeError |
| `to_numpy()` | NumPy array (view) | NumPy array (view) | NumPy array (**copy**) |
| `copy_to_host()` | NumPy array (copy) | NumPy array (copy) | NumPy array (copy) |
| `copy_from_host(arr)` | Copies to container | Copies to container | Copies to container |
| `to_cupy()` | ❌ Not available | CuPy array | CuPy array (view) |

**Fortran Behavior:**
```fortran
! Check if host accessible first
if (.not. dtl_vector_is_host_accessible_f(vec)) then
    ! Use explicit copy
    call dtl_vector_copy_to_host_f(vec, host_array, size, ierr)
    ! Work with host_array
    call dtl_vector_copy_from_host_f(vec, host_array, size, ierr)
else
    ! Direct pointer access
    data_ptr = dtl_vector_local_data_f(vec)
    call c_f_pointer(data_ptr, data_array, [size])
end if
```

---

## Algorithm Dispatch in Bindings

### C ABI

Algorithms automatically dispatch based on container placement:

```c
// For host containers: runs on CPU
// For device containers: runs on GPU
dtl_error dtl_vector_fill(dtl_vector_handle* vec, const void* value);

// Explicit GPU algorithms (require device-accessible containers)
dtl_error dtl_vector_fill_device(dtl_vector_handle* vec, const void* value);
```

### Python

```python
# Generic APIs auto-dispatch
vec.fill(42)  # GPU or CPU depending on placement

# Or use explicit dispatch
import dtl.algorithms as alg
alg.fill(vec, 42)  # Auto-dispatch
alg.fill_host(vec, 42)  # Force CPU (may copy)
alg.fill_device(vec, 42)  # Force GPU (error if not device-accessible)
```

---

## Error Handling

### Error Codes

| Error | Meaning | Recovery |
|-------|---------|----------|
| `DTL_ERROR_HOST_ACCESS_DENIED` | Cannot access device memory from host | Use copy functions |
| `DTL_ERROR_DEVICE_ACCESS_DENIED` | Cannot access host memory from device | Use copy or unified |
| `DTL_ERROR_TYPE_NOT_DEVICE_STORABLE` | Element type cannot be on device | Use host placement |
| `DTL_ERROR_CUDA_NOT_AVAILABLE` | CUDA required but not enabled | Rebuild with CUDA |

### Python Exceptions

```python
class HostAccessDeniedError(RuntimeError):
    """Raised when attempting host access on device-only container."""
    pass

class DeviceAccessDeniedError(RuntimeError):
    """Raised when attempting device access on host-only container."""
    pass

class TypeNotDeviceStorableError(TypeError):
    """Raised when element type cannot be stored on device."""
    pass
```

---

## Type Constraints for Device Placement

Only **DeviceStorable** types can be used with device placements:

### Definition
```cpp
template <typename T>
concept DeviceStorable = std::is_trivially_copyable_v<T>;
```

### Supported Types
- All numeric types: `int`, `float`, `double`, `int64_t`, etc.
- POD structs with only numeric members
- Fixed-size arrays of numeric types

### Unsupported Types
- `std::string`
- `std::vector`
- Classes with virtual functions
- Types with non-trivial constructors/destructors

### Binding Behavior

**C ABI:**
```c
// Creation with unsupported type returns error
dtl_error err = dtl_vector_create_with_placement(
    DTL_DTYPE_STRING,  // Not device-storable
    DTL_PLACEMENT_DEVICE_ONLY,
    &handle
);
assert(err == DTL_ERROR_TYPE_NOT_DEVICE_STORABLE);
```

**Python:**
```python
# Attempting to create device container with unsupported type
try:
    vec = dtl.Vector(dtype=object, placement='device_only')
except dtl.TypeNotDeviceStorableError:
    print("Object dtype not supported for device placement")
```

---

## Performance Recommendations

1. **Prefer host placement** for small data or frequent host access
2. **Use unified memory** for convenient GPU/CPU access with moderate performance
3. **Use device-only** for GPU-intensive workloads with minimal host access
4. **Batch copies** to amortize transfer overhead
5. **Use async copies** when overlapping compute with data movement
6. **Consider CuPy integration** for Python GPU workflows

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.2 | 2026-02-04 | Initial specification (Phase 03) |

---

## References

- [Policy Capability Matrix](../capabilities/policy_matrix.md)
