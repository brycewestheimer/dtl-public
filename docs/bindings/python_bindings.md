# Python Bindings Guide

This guide covers DTL's Python bindings, providing a Pythonic interface with seamless NumPy integration for distributed computing.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Context API](#context-api)
- [Containers](#containers)
  - [DistributedVector](#distributedvector)
  - [DistributedArray](#distributedarray)
  - [DistributedSpan](#distributedspan)
  - [DistributedTensor](#distributedtensor)
- [Collective Operations](#collective-operations)
- [Algorithm Operations](#algorithm-operations)
- [Policy Selection](#policy-selection)
- [RMA Operations](#rma-operations)
- [NumPy Integration](#numpy-integration)
- [mpi4py Interoperability](#mpi4py-interoperability)
- [Exception Handling](#exception-handling)
- [Type Annotations](#type-annotations)
- [Running with MPI](#running-with-mpi)
- [Complete Examples](#complete-examples)

---

## Overview

The Python bindings provide:

- **Pythonic API**: Follows Python conventions (PEP 8, context managers)
- **Zero-Copy NumPy Views**: `local_view()` shares memory with DTL containers
- **mpi4py Compatibility**: Pass mpi4py communicators directly
- **Full Type Annotations**: PEP 484 type hints for IDE support
- **Collective Operations**: `allreduce`, `broadcast`, `gather`, `scatter`, etc.

### Requirements

- Python 3.8+
- NumPy 1.20+
- (Optional) mpi4py 3.0+ for MPI parallelism

---

## Installation

### From Source (in DTL Build Directory)

```bash
cd dtl/build
cmake .. -DDTL_BUILD_PYTHON=ON
make _dtl
make python_install
```

### Verify Installation

```python
>>> import dtl
>>> dtl.__version__
'0.1.0a1'
>>> dtl.has_mpi()
True
```

---

## Quick Start

```python
import dtl
import numpy as np

# Create execution context
with dtl.Context() as ctx:
    print(f"Rank {ctx.rank} of {ctx.size}")

    # Create distributed vector (1000 elements, float64)
    vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)

    # Access local partition as NumPy array (zero-copy)
    local = vec.local_view()
    local[:] = np.arange(len(local)) + vec.local_offset

    # Collective reduce
    local_sum = np.sum(local)
    global_sum = dtl.allreduce(ctx, local_sum, op=dtl.SUM)

    if ctx.is_root:
        print(f"Global sum: {global_sum}")
```

Run with MPI:

```bash
mpirun -np 4 python my_script.py
```

---

## Environment API

The `Environment` class manages backend lifecycle (MPI, CUDA, HIP, NCCL, SHMEM) using reference-counted RAII semantics. This is the recommended way to initialize DTL.

### Creation

```python
# Create environment (initializes backends on first call)
env = dtl.Environment()

# As context manager (recommended)
with dtl.Environment() as env:
    ctx = env.make_world_context()
    print(f"Rank {ctx.rank} of {ctx.size}")
```

### Context Factory Methods

```python
# Create context spanning all MPI ranks (CPU)
ctx = env.make_world_context()

# Create GPU-enabled context
ctx = env.make_world_context(device_id=0)

# Create single-process CPU-only context (no MPI)
ctx = env.make_cpu_context()
```

### Backend Queries

```python
env.has_mpi     # bool: MPI available
env.has_cuda    # bool: CUDA available
env.has_hip     # bool: HIP available
env.has_nccl    # bool: NCCL available
env.has_shmem   # bool: SHMEM available
```

### Static Queries

```python
dtl.Environment.is_initialized()  # True if any environment handle exists
dtl.Environment.ref_count()       # Number of active environment handles
```

---

## Context API

The `Context` class encapsulates the execution environment. Contexts can be created directly or via `Environment` factory methods (preferred).

### Creation

```python
# Default: uses MPI_COMM_WORLD, CPU only
ctx = dtl.Context()

# With GPU device
ctx = dtl.Context(device_id=0)

# With mpi4py communicator
from mpi4py import MPI
ctx = dtl.Context(comm=MPI.COMM_WORLD)
```

### Properties

```python
ctx.rank            # int: Current rank (0 to size-1)
ctx.size            # int: Total number of ranks
ctx.is_root         # bool: True if rank 0
ctx.device_id       # int: GPU device ID (-1 for CPU only)
ctx.has_device      # bool: True if GPU enabled
ctx.has_mpi         # bool: MPI domain present
ctx.has_cuda        # bool: CUDA domain present
ctx.has_nccl        # bool: NCCL domain present
ctx.nccl_mode       # int: DTL_NCCL_MODE_* (or -1 if no NCCL)
```

### Methods

```python
ctx.barrier()  # Synchronize all ranks
ctx.fence()    # Local memory fence
ctx.dup()      # Duplicate context communicator
ctx.split(color, key=0)
ctx.with_cuda(device_id)
ctx.with_nccl(device_id, mode=dtl.DTL_NCCL_MODE_HYBRID_PARITY)
ctx.split_nccl(color, key=0, device_id=None, mode=dtl.DTL_NCCL_MODE_HYBRID_PARITY)
ctx.nccl_supports_native(dtl.DTL_NCCL_OP_ALLREDUCE)
ctx.nccl_supports_hybrid(dtl.DTL_NCCL_OP_SCAN)
```

NCCL mode constants:

```python
dtl.DTL_NCCL_MODE_NATIVE_ONLY
dtl.DTL_NCCL_MODE_HYBRID_PARITY
```

Note: explicit C ABI NCCL device-collective entry points (`dtl_nccl_*_device`
and `*_device_ex`) are currently exposed directly in C/Fortran. Python
currently uses the generic collective API surface plus explicit context/domain
selection.

### Context Manager

The recommended pattern is using `with`:

```python
with dtl.Context() as ctx:
    # Context is valid here
    vec = dtl.DistributedVector(ctx, 1000)
    # ...
# Context cleaned up automatically
```

---

## Containers

### DistributedVector

A 1D distributed array partitioned across ranks.

```python
# Create vector
vec = dtl.DistributedVector(ctx, size=10000, dtype=np.float64)

# With fill value
vec = dtl.DistributedVector(ctx, size=10000, dtype=np.float32, fill=0.0)

# Properties
vec.global_size   # Total elements across all ranks
vec.local_size    # Elements on this rank
vec.local_offset  # Global index of first local element
vec.dtype         # NumPy dtype

# Zero-copy NumPy view
local = vec.local_view()
local[:] = np.random.randn(len(local))

# Index queries
vec.is_local(global_idx)  # bool: Is index on this rank?
vec.owner(global_idx)     # int: Which rank owns index?
```

#### Supported dtypes

| NumPy dtype | Status |
|-------------|--------|
| `np.float64` | Supported |
| `np.float32` | Supported |
| `np.int64` | Supported |
| `np.int32` | Supported |
| `np.uint64` | Supported |
| `np.uint32` | Supported |
| `np.int8` | Supported |
| `np.uint8` | Supported |

### DistributedArray

A fixed-size 1D distributed array. Unlike `DistributedVector`, the size is fixed at creation time and cannot be resized.

```python
# Create fixed-size array (1000 elements, float64)
arr = dtl.DistributedArray(ctx, size=1000, dtype=np.float64)

# With fill value
arr = dtl.DistributedArray(ctx, size=1000, dtype=np.float64, fill=0.0)

# With policy selection
arr = dtl.DistributedArray(
    ctx, size=1000, dtype=np.float64,
    partition=dtl.PARTITION_CYCLIC
)

# Properties (same as DistributedVector)
arr.global_size   # Total elements (always 1000)
arr.local_size    # Elements on this rank
arr.local_offset  # Global index of first local element
arr.dtype         # NumPy dtype

# Zero-copy NumPy view
local = arr.local_view()
local[:] = np.arange(len(local))

# Index queries
arr.is_local(global_idx)  # bool: Is index on this rank?
arr.owner(global_idx)     # int: Which rank owns index?

# NO resize() method - arrays are fixed size
# arr.resize(2000)  # AttributeError!
```

**Key Difference from DistributedVector**: Arrays are fixed-size containers. Use `DistributedArray` when:
- Size is known at creation time and won't change
- You want compile-time size guarantees
- Slightly more memory-efficient than vector

### DistributedSpan

Python exposes a first-class `DistributedSpan` factory that creates typed non-owning distributed spans from distributed containers:

```python
vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)
span = dtl.DistributedSpan(vec)

local = span.local_view()     # zero-copy local NumPy view
copy = span.to_numpy()        # explicit copy
sub = span.subspan(10, 20)    # non-owning subspan
```

Key semantics:

- non-owning: span lifetime is tied to the source container lifetime
- local contiguous semantics: indexing/slicing through NumPy local views
- distributed metadata: `global_size`, `local_size`, `rank`, `num_ranks`

### DistributedTensor

An N-dimensional distributed array, partitioned along the first dimension.

```python
# Create 3D tensor (100x64x64)
tensor = dtl.DistributedTensor(ctx, shape=(100, 64, 64), dtype=np.float32)

# Properties
tensor.shape        # Tuple: Global shape
tensor.local_shape  # Tuple: Local shape (differs in dim 0)
tensor.ndim         # int: Number of dimensions
tensor.dtype        # NumPy dtype

# Zero-copy NumPy view
local = tensor.local_view()  # 3D NumPy array
local[:] = np.random.randn(*local.shape)
```

---

## Collective Operations

DTL provides distributed collective operations that work with NumPy arrays.

### Available Operations

| Function | Description |
|----------|-------------|
| `dtl.allreduce(ctx, data, op)` | Reduce across ranks, result to all |
| `dtl.reduce(ctx, data, op, root)` | Reduce across ranks, result to root |
| `dtl.broadcast(ctx, data, root)` | Send from root to all ranks |
| `dtl.gather(ctx, data, root)` | Collect data at root |
| `dtl.scatter(ctx, data, root)` | Distribute data from root |
| `dtl.allgather(ctx, data)` | Gather to all ranks |

### Reduction Operations

```python
dtl.SUM   # Sum
dtl.PROD  # Product
dtl.MIN   # Minimum
dtl.MAX   # Maximum
```

### Examples

```python
import dtl
import numpy as np

with dtl.Context() as ctx:
    # Allreduce: global sum
    local_value = np.array([ctx.rank * 10.0])
    global_sum = dtl.allreduce(ctx, local_value, op=dtl.SUM)

    # Reduce to root
    global_max = dtl.reduce(ctx, local_value, op=dtl.MAX, root=0)

    # Broadcast from root
    if ctx.is_root:
        data = np.array([1.0, 2.0, 3.0])
    else:
        data = np.zeros(3)
    data = dtl.broadcast(ctx, data, root=0)

    # Gather: collect arrays at root
    my_data = np.array([ctx.rank])
    gathered = dtl.gather(ctx, my_data, root=0)
    # On root: gathered.shape = (size, 1)

    # Scatter: distribute from root
    if ctx.is_root:
        all_data = np.arange(ctx.size * 3).reshape(ctx.size, 3)
    else:
        all_data = np.empty((ctx.size, 3))
    my_chunk = dtl.scatter(ctx, all_data, root=0)
    # my_chunk.shape = (3,)

    # Allgather: gather to all
    all_gathered = dtl.allgather(ctx, my_data)
    # all_gathered.shape = (size, 1)
```

### Working with Scalars

Collective operations handle scalars automatically:

```python
# Scalar input
local_sum = 42.0
global_sum = dtl.allreduce(ctx, local_sum, op=dtl.SUM)
# global_sum is a scalar (float)

# Array input
local_arr = np.array([1.0, 2.0, 3.0])
global_arr = dtl.allreduce(ctx, local_arr, op=dtl.SUM)
# global_arr is np.ndarray
```

---

## Algorithm Operations

DTL provides distributed algorithm operations that work on containers.

### Container-Level Algorithms

| Function | Description |
|----------|-------------|
| `dtl.for_each(container, func)` | Apply function to each element |
| `dtl.transform(src, dst, func)` | Transform elements from src to dst |
| `dtl.fill(container, value)` | Fill container with value |
| `dtl.copy(src, dst)` | Copy elements from src to dst |
| `dtl.find(container, value)` | Find first occurrence of value |
| `dtl.count(container, value)` | Count occurrences of value |
| `dtl.reduce(container, op)` | Reduce all elements |
| `dtl.sort(container)` | Sort elements |

### Examples

```python
import dtl
import numpy as np

with dtl.Context() as ctx:
    vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)
    local = vec.local_view()
    local[:] = np.random.randn(len(local))

    # Fill with value
    dtl.fill(vec, 0.0)

    # Transform with NumPy ufunc
    src = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)
    dst = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)
    src.local_view()[:] = np.arange(src.local_size)
    dtl.transform(src, dst, np.square)

    # Distributed reduce
    total = dtl.reduce(vec, op=dtl.SUM)
    maximum = dtl.reduce(vec, op=dtl.MAX)

    # Sort (distributed)
    dtl.sort(vec)
    dtl.sort(vec, reverse=True)
```

---

## Policy Selection

Policies control how data is distributed and where it resides.

### Partition Policies

Control how elements are distributed across ranks:

```python
dtl.PARTITION_BLOCK   # Contiguous chunks (default)
dtl.PARTITION_CYCLIC  # Round-robin distribution
dtl.PARTITION_HASH    # Hash-based distribution
```

```python
# Block partition: ranks get contiguous chunks
# [0,1,2,3,4,5,6,7] with 2 ranks -> rank0:[0,1,2,3], rank1:[4,5,6,7]
vec_block = dtl.DistributedVector(
    ctx, size=1000, dtype=np.float64,
    partition=dtl.PARTITION_BLOCK
)

# Cyclic partition: round-robin
# [0,1,2,3,4,5,6,7] with 2 ranks -> rank0:[0,2,4,6], rank1:[1,3,5,7]
vec_cyclic = dtl.DistributedVector(
    ctx, size=1000, dtype=np.float64,
    partition=dtl.PARTITION_CYCLIC
)
```

### Placement Policies

Control where data resides in memory:

```python
dtl.PLACEMENT_HOST     # CPU memory (default)
dtl.PLACEMENT_DEVICE   # GPU memory (requires CUDA)
dtl.PLACEMENT_UNIFIED  # CUDA unified memory
```

```python
# CPU-only vector
cpu_vec = dtl.DistributedVector(
    ctx, size=1000, dtype=np.float64,
    placement=dtl.PLACEMENT_HOST
)

# GPU vector (if CUDA available)
if dtl.has_cuda():
    gpu_vec = dtl.DistributedVector(
        ctx, size=1000, dtype=np.float64,
        placement=dtl.PLACEMENT_DEVICE
    )
```

### Querying Policies

```python
vec.partition_policy  # Returns current partition policy
vec.placement_policy  # Returns current placement policy
```

### Availability Checks

```python
dtl.has_cuda()     # bool: CUDA backend available
dtl.has_mpi()      # bool: MPI backend available
```

---

## RMA Operations

Remote Memory Access (RMA) enables one-sided communication where one rank can directly access another rank's memory.

### Window Creation

A `Window` exposes memory for remote access:

```python
import dtl
import numpy as np

with dtl.Context() as ctx:
    # Allocate window (DTL allocates memory)
    win = dtl.Window(ctx, size=1024)

    # Create window from existing array
    data = np.zeros(100, dtype=np.float64)
    win = dtl.Window(ctx, base=data)
```

### Window Properties

```python
win.size      # Size in bytes
win.base      # Base pointer (as integer)
win.is_valid  # Whether window handle is valid
```

### Synchronization

RMA operations require synchronization epochs:

```python
# Active-target synchronization (fence)
win.fence()
# ... RMA operations ...
win.fence()

# Passive-target synchronization (lock/unlock)
win.lock(target=1, mode="exclusive")
# ... RMA operations to target 1 ...
win.unlock(target=1)

# Lock all ranks
win.lock_all()
# ... RMA operations ...
win.unlock_all()

# Flush operations
win.flush(target=1)      # Complete operations to target
win.flush_all()          # Complete operations to all targets
win.flush_local(target=1)     # Local completion only
win.flush_local_all()         # Local completion for all
```

### Data Transfer

```python
import dtl
import numpy as np

with dtl.Context() as ctx:
    win = dtl.Window(ctx, size=1024)

    # Put data to remote window
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    win.fence()
    dtl.rma_put(win, target=0, offset=0, data=data)
    win.fence()

    # Get data from remote window
    win.fence()
    result = dtl.rma_get(win, target=0, offset=0, size=24, dtype=np.float64)
    win.fence()
    # result is np.array([1.0, 2.0, 3.0])
```

### Atomic Operations

```python
# Accumulate: atomically add to remote memory
dtl.rma_accumulate(win, target=0, offset=0, data=np.array([1.0]), op=dtl.SUM)

# Fetch and add: get old value, add new
old_value = dtl.rma_fetch_and_add(
    win, target=0, offset=0,
    addend=1.0, dtype=np.float64
)

# Compare and swap: atomic conditional update
old_value = dtl.rma_compare_and_swap(
    win, target=0, offset=0,
    compare=0.0, swap=1.0, dtype=np.float64
)
# If remote value was 0.0, it's now 1.0
# old_value contains the value that was there
```

### RMA Example: Distributed Counter

```python
"""Atomic distributed counter using RMA."""
import dtl
import numpy as np

with dtl.Context() as ctx:
    # Each rank has a counter window
    counter = np.zeros(1, dtype=np.int64)
    win = dtl.Window(ctx, base=counter)

    # All ranks increment rank 0's counter
    win.lock(target=0, mode="exclusive")
    old = dtl.rma_fetch_and_add(
        win, target=0, offset=0,
        addend=1, dtype=np.int64
    )
    win.unlock(target=0)

    ctx.barrier()

    if ctx.is_root:
        print(f"Final counter value: {counter[0]}")
        # Should equal ctx.size
```

---

## NumPy Integration

### Zero-Copy Views

`local_view()` returns a NumPy array that shares memory with the DTL container:

```python
vec = dtl.DistributedVector(ctx, 1000)
local = vec.local_view()

# Modifications affect the container
local[:] = np.sin(np.arange(len(local)))

# NumPy operations work directly
mean = np.mean(local)
local_sorted = np.sort(local)  # Creates copy
np.multiply(local, 2, out=local)  # In-place, zero-copy
```

### View Lifetime

Views are valid as long as the container exists:

```python
# CORRECT: Container outlives view
vec = dtl.DistributedVector(ctx, 1000)
local = vec.local_view()
local[:] = 0  # Safe

# INCORRECT: Container destroyed before view used
local = dtl.DistributedVector(ctx, 1000).local_view()  # vec destroyed!
local[:] = 0  # UNDEFINED BEHAVIOR
```

### Performance Tips

```python
# GOOD: Vectorized NumPy operations
local = vec.local_view()
local[:] = np.exp(local)  # Fast, uses SIMD

# BAD: Python loops
for i in range(len(local)):
    local[i] = np.exp(local[i])  # Slow, Python overhead
```

---

## mpi4py Interoperability

DTL integrates with mpi4py for advanced MPI usage:

```python
from mpi4py import MPI
import dtl

# Use existing communicator
world = MPI.COMM_WORLD
ctx = dtl.Context(comm=world)

# Split communicator
color = ctx.rank % 2
sub_comm = world.Split(color, ctx.rank)
sub_ctx = dtl.Context(comm=sub_comm)

# Operations within sub-communicator
vec = dtl.DistributedVector(sub_ctx, 1000)
# Collectives only involve ranks in sub_comm
```

### Without mpi4py

If mpi4py is not installed, `Context()` uses MPI_COMM_WORLD internally:

```python
# Works without mpi4py
ctx = dtl.Context()  # Uses MPI_COMM_WORLD
```

---

## Exception Handling

DTL exceptions map C status codes to Python:

```python
class DTLError(Exception):
    """Base class for DTL exceptions."""

class CommunicationError(DTLError):
    """MPI/communication failure."""

class MemoryError(DTLError, MemoryError):
    """Memory allocation failure."""

class BoundsError(DTLError, IndexError):
    """Index out of bounds."""

class InvalidArgumentError(DTLError, ValueError):
    """Invalid argument."""

class BackendError(DTLError):
    """Backend (MPI/CUDA) failure."""
```

### Handling Exceptions

```python
try:
    vec = dtl.DistributedVector(ctx, -1)  # Invalid size
except dtl.InvalidArgumentError as e:
    print(f"Invalid argument: {e}")
except dtl.DTLError as e:
    print(f"DTL error: {e}")
```

---

## Type Annotations

DTL includes PEP 484 type hints and a `.pyi` stub file:

```python
# IDE will show types
def allreduce(
    ctx: Context,
    data: Union[np.ndarray, float, int],
    op: str = "sum"
) -> Union[np.ndarray, float, int]: ...
```

### Type Checking

```bash
mypy your_script.py
```

---

## Running with MPI

### Basic Execution

```bash
# Run with 4 MPI ranks
mpirun -np 4 python my_script.py

# With OpenMPI hostfile
mpirun -np 8 --hostfile hosts python my_script.py
```

### pytest-mpi

For testing:

```bash
pip install pytest-mpi
mpirun -np 4 python -m pytest tests/ --with-mpi
```

---

## Complete Examples

### Example 1: Distributed Mean

```python
"""Compute mean of distributed data."""
import dtl
import numpy as np

with dtl.Context() as ctx:
    # Create vector with 1M elements
    vec = dtl.DistributedVector(ctx, size=1_000_000, dtype=np.float64)

    # Fill with random data
    local = vec.local_view()
    np.random.seed(42 + ctx.rank)
    local[:] = np.random.randn(len(local))

    # Local sum and count
    local_sum = np.sum(local)
    local_count = len(local)

    # Global reduction
    global_sum = dtl.allreduce(ctx, local_sum, op=dtl.SUM)
    global_count = dtl.allreduce(ctx, float(local_count), op=dtl.SUM)

    global_mean = global_sum / global_count

    if ctx.is_root:
        print(f"Global mean: {global_mean:.6f}")
```

### Example 2: Distributed Dot Product

```python
"""Compute dot product of two distributed vectors."""
import dtl
import numpy as np

with dtl.Context() as ctx:
    N = 100_000

    # Create two vectors
    a = dtl.DistributedVector(ctx, size=N, dtype=np.float64)
    b = dtl.DistributedVector(ctx, size=N, dtype=np.float64)

    # Initialize
    local_a = a.local_view()
    local_b = b.local_view()

    local_a[:] = np.arange(len(local_a)) + a.local_offset
    local_b[:] = 2.0

    # Local dot product
    local_dot = np.dot(local_a, local_b)

    # Global sum
    global_dot = dtl.allreduce(ctx, local_dot, op=dtl.SUM)

    if ctx.is_root:
        # Expected: 2 * sum(0..N-1) = 2 * N*(N-1)/2 = N*(N-1)
        expected = N * (N - 1)
        print(f"Dot product: {global_dot:.0f}")
        print(f"Expected:    {expected}")
```

### Example 3: Stencil Computation with Halo Exchange

```python
"""1D stencil with neighbor communication."""
import dtl
import numpy as np

with dtl.Context() as ctx:
    N = 1000
    vec = dtl.DistributedVector(ctx, size=N, dtype=np.float64)

    local = vec.local_view()
    local[:] = np.sin(np.linspace(0, 2*np.pi, vec.global_size))[vec.local_offset:vec.local_offset+len(local)]

    # Simple 3-point stencil: new[i] = 0.25*old[i-1] + 0.5*old[i] + 0.25*old[i+1]
    # (Boundary handling omitted for simplicity)

    result = np.zeros_like(local)

    for i in range(1, len(local) - 1):
        result[i] = 0.25 * local[i-1] + 0.5 * local[i] + 0.25 * local[i+1]

    # Copy boundaries
    result[0] = local[0]
    result[-1] = local[-1]

    local[:] = result

    ctx.barrier()

    if ctx.is_root:
        print("Stencil computation complete")
```

---

## References

- [C Bindings Guide](c_bindings.md) (underlying C ABI)
- [User Guide: Bindings Overview](../user_guide/bindings.md)
- [Example Notebooks](../../examples/python/notebooks/)
