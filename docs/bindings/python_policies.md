# Python Policy Parameters

**Since:** v1.2.0 (Phase 05)

This document describes how to use policy parameters when creating distributed containers in Python.

---

## Overview

DTL Python bindings now support policy parameters in container factory functions, providing parity with the C++ and C APIs. You can specify:

- **Partition Policy**: How data is distributed across ranks
- **Placement Policy**: Where data is stored (CPU vs GPU)
- **Execution Policy**: How operations are executed
- **Device ID**: Which GPU to use for device placements
- **Block Size**: Block size for block-cyclic partitioning

---

## Quick Start

```python
import dtl
import numpy as np

with dtl.Context() as ctx:
    # Default (host memory, block partition)
    vec1 = dtl.DistributedVector(ctx, size=1000)

    # Cyclic partition on host
    vec2 = dtl.DistributedVector(
        ctx, size=1000,
        partition=dtl.PARTITION_CYCLIC
    )

    # Unified memory (requires CUDA)
    if dtl.has_cuda():
        vec3 = dtl.DistributedVector(
            ctx, size=1000,
            placement=dtl.PLACEMENT_UNIFIED
        )
```

---

## Factory Signatures

### DistributedVector

```python
def DistributedVector(
    ctx: Context,
    size: int,
    dtype=None,             # np.float64 default
    fill=None,              # Optional fill value
    *,
    partition=PARTITION_BLOCK,
    placement=PLACEMENT_HOST,
    execution=EXEC_SEQ,
    device_id: int = 0,
    block_size: int = 1
) -> DistributedVector_*
```

### DistributedArray

```python
def DistributedArray(
    ctx: Context,
    size: int,
    dtype=None,
    fill=None,
    *,
    partition=PARTITION_BLOCK,
    placement=PLACEMENT_HOST,
    execution=EXEC_SEQ,
    device_id: int = 0,
    block_size: int = 1
) -> DistributedArray_*
```

### DistributedTensor

```python
def DistributedTensor(
    ctx: Context,
    shape,
    dtype=None,
    fill=None,
    *,
    partition=PARTITION_BLOCK,     # Reserved for future use
    placement=PLACEMENT_HOST,       # Reserved for future use
    execution=EXEC_SEQ,             # Reserved for future use
    device_id: int = 0,
    block_size: int = 1
) -> DistributedTensor_*
```

> **Note:** Tensor policy parameters are accepted for API consistency but may not be fully supported in this release.

---

## Partition Policies

| Constant | Description |
|----------|-------------|
| `PARTITION_BLOCK` | Contiguous chunks per rank (default) |
| `PARTITION_CYCLIC` | Round-robin distribution |
| `PARTITION_BLOCK_CYCLIC` | Block-cyclic with specified block_size |
| `PARTITION_HASH` | Hash-based distribution |
| `PARTITION_REPLICATED` | Full copy on each rank |

```python
# Block partition (default)
vec = dtl.DistributedVector(ctx, size=1000, partition=dtl.PARTITION_BLOCK)

# Cyclic partition
vec = dtl.DistributedVector(ctx, size=1000, partition=dtl.PARTITION_CYCLIC)

# Block-cyclic with block size 16
vec = dtl.DistributedVector(
    ctx, size=1000,
    partition=dtl.PARTITION_BLOCK_CYCLIC,
    block_size=16
)
```

---

## Placement Policies

| Constant | Description | `local_view()` |
|----------|-------------|----------------|
| `PLACEMENT_HOST` | CPU memory (default) | ✅ Works |
| `PLACEMENT_DEVICE` | GPU memory only | ❌ Raises |
| `PLACEMENT_UNIFIED` | Unified/managed memory | ✅ Works |
| `PLACEMENT_DEVICE_PREFERRED` | Device-preferred with fallback | ✅ Works |

### Checking Placement Availability

```python
import dtl

# Check if CUDA is available
if dtl.has_cuda():
    print("CUDA backend available")

# Check specific placement
if dtl.placement_available(dtl.PLACEMENT_UNIFIED):
    vec = dtl.DistributedVector(
        ctx, size=1000,
        placement=dtl.PLACEMENT_UNIFIED
    )
```

### Device-Only Placement Safety

For `PLACEMENT_DEVICE`, the data resides only in GPU memory. Calling `local_view()` would create an invalid NumPy array pointing to GPU memory, so it raises an error:

```python
vec = dtl.DistributedVector(
    ctx, size=1000,
    placement=dtl.PLACEMENT_DEVICE
)

# This raises RuntimeError:
# local = vec.local_view()  # RuntimeError: Cannot create NumPy view of device-only memory

# Use copy-based APIs instead:
arr = vec.to_numpy()        # Copy GPU -> host (returns new array)
vec.from_numpy(arr)         # Copy host -> GPU
```

---

## Copy APIs (to_numpy / from_numpy)

All containers provide copy-based data access that works for any placement:

### `to_numpy()`

Copies local data to a new NumPy array. Works for all placements.

```python
vec = dtl.DistributedVector(
    ctx, size=1000, fill=42.0,
    placement=dtl.PLACEMENT_DEVICE
)

# Copy device data to host
arr = vec.to_numpy()
assert np.all(arr == 42.0)
```

### `from_numpy(arr)`

Copies data from a NumPy array to local container data. Works for all placements.

```python
vec = dtl.DistributedVector(
    ctx, size=1000,
    placement=dtl.PLACEMENT_DEVICE
)

# Copy host data to device
source = np.full(vec.local_size, 7.0)
vec.from_numpy(source)
```

> **Size Check:** `from_numpy()` validates that the source array size matches `local_size`. A mismatch raises `RuntimeError`.

---

## Execution Policies

| Constant | Description |
|----------|-------------|
| `EXEC_SEQ` | Sequential execution (default) |
| `EXEC_PAR` | Parallel execution (multi-threaded) |
| `EXEC_ASYNC` | Asynchronous execution (non-blocking) |

```python
vec = dtl.DistributedVector(
    ctx, size=1000,
    execution=dtl.EXEC_PAR
)
```

---

## Device ID

Specify which GPU device to use for device placements:

```python
# Use GPU 0 (default)
vec = dtl.DistributedVector(
    ctx, size=1000,
    placement=dtl.PLACEMENT_DEVICE,
    device_id=0
)

# Use GPU 1
vec = dtl.DistributedVector(
    ctx, size=1000,
    placement=dtl.PLACEMENT_DEVICE,
    device_id=1
)
```

---

## Container Properties

Containers expose policy-related properties:

```python
vec = dtl.DistributedVector(
    ctx, size=1000,
    placement=dtl.PLACEMENT_HOST
)

# Placement policy (integer value)
print(f"Placement: {vec.placement}")  # 0 (PLACEMENT_HOST)

# Check if local_view is safe
if vec.is_host_accessible:
    local = vec.local_view()
else:
    local = vec.to_numpy()
```

---

## Backwards Compatibility

Existing code continues to work unchanged:

```python
# These are equivalent:
vec1 = dtl.DistributedVector(ctx, size=1000)
vec2 = dtl.DistributedVector(
    ctx, size=1000,
    partition=dtl.PARTITION_BLOCK,
    placement=dtl.PLACEMENT_HOST,
    execution=dtl.EXEC_SEQ,
    device_id=0,
    block_size=1
)
```

Old-style positional arguments are still supported:

```python
vec = dtl.DistributedVector(ctx, 1000, np.float64, 42.0)  # Still works
```

---

## Error Handling

### Unavailable Placement

If a placement is not available (e.g., CUDA not compiled in), factory functions raise `ValueError`:

```python
try:
    vec = dtl.DistributedVector(
        ctx, size=1000,
        placement=dtl.PLACEMENT_DEVICE
    )
except ValueError as e:
    print(f"Placement not available: {e}")
```

### Device-Only Access

Attempting `local_view()` on device-only data raises `RuntimeError`:

```python
vec = dtl.DistributedVector(
    ctx, size=1000,
    placement=dtl.PLACEMENT_DEVICE
)

try:
    local = vec.local_view()
except RuntimeError as e:
    print(f"Use to_numpy() instead: {e}")
    arr = vec.to_numpy()
```

---

## See Also

- [Device Placement Semantics](device_placement_semantics.md)
- [Python Bindings Overview](python_bindings.md)
- [C Bindings Policy Dispatch](c_bindings.md#policy-dispatch)
