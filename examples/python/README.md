# DTL Python Examples

This directory contains example Python scripts and Jupyter notebooks
demonstrating the DTL (Distributed Template Library) Python bindings.

## Prerequisites

1. Install DTL with Python bindings:
   ```bash
   pip install dtl
   ```

2. For MPI examples, install mpi4py:
   ```bash
   pip install mpi4py
   ```

## Directory Structure

```
examples/python/
├── README.md              # This file
├── scripts/               # Standalone Python scripts
│   ├── hello_dtl.py       # Basic DTL usage
│   ├── distributed_reduce.py  # Distributed reduction example
│   └── nccl_modes.py      # NCCL native/hybrid mode demo
└── notebooks/             # Jupyter notebooks
    ├── 01_getting_started.ipynb
    ├── 02_distributed_vectors.ipynb
    ├── 03_distributed_tensors.ipynb
    └── 04_algorithms.ipynb
```

## Running Scripts

### Single Process

```bash
python scripts/hello_dtl.py
```

### With MPI

```bash
mpirun -np 4 python scripts/hello_dtl.py
mpirun -np 4 python scripts/distributed_reduce.py
mpirun -np 2 python scripts/nccl_modes.py
```

## Running Notebooks

For single-process use, launch Jupyter normally:

```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

For MPI-parallel notebooks, use ipyparallel or run cells as scripts:

```bash
# Extract code from notebook and run with MPI
jupyter nbconvert --to script notebooks/02_distributed_vectors.ipynb
mpirun -np 4 python notebooks/02_distributed_vectors.py
```

## Quick Start

```python
import dtl
import numpy as np

# Create context
with dtl.Context() as ctx:
    print(f"Rank {ctx.rank} of {ctx.size}")

    # Create distributed vector
    vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)

    # Get local view (zero-copy NumPy array)
    local = vec.local_view()
    local[:] = ctx.rank  # Fill with rank number

    # Synchronize
    ctx.barrier()

    print(f"Rank {ctx.rank}: local_size={vec.local_size}, "
          f"offset={vec.local_offset}")
```

## Examples Overview

### hello_dtl.py
Basic introduction to DTL:
- Creating a context
- Querying rank and size
- Feature detection (MPI, CUDA)

### distributed_reduce.py
Demonstrates distributed operations:
- Creating distributed vectors
- Computing local sums
- Distributed reduction (requires MPI)

### nccl_modes.py
Demonstrates mode-aware NCCL context behavior:
- `with_nccl(..., mode=...)` for native-only and hybrid parity modes
- `split_nccl(..., mode=...)`
- `nccl_supports_native` / `nccl_supports_hybrid` capability queries

### Notebooks

1. **01_getting_started.ipynb**: Introduction to DTL concepts
2. **02_distributed_vectors.ipynb**: Working with distributed 1D arrays
3. **03_distributed_tensors.ipynb**: Working with N-dimensional distributed arrays
4. **04_algorithms.ipynb**: Distributed algorithms (transform, reduce)

## API Reference

### Context

```python
dtl.Context(comm=None, device_id=-1)
```
- `comm`: mpi4py MPI.Comm object (None for MPI_COMM_WORLD)
- `device_id`: GPU device ID (-1 for CPU-only)

Properties:
- `rank`: Current process rank (0 to size-1)
- `size`: Total number of processes
- `is_root`: True if rank 0
- `device_id`: GPU device ID
- `has_device`: True if GPU enabled

Methods:
- `barrier()`: Synchronize all ranks
- `with_cuda(device_id)`: Add CUDA domain
- `with_nccl(device_id, mode=...)`: Add NCCL domain with explicit mode
- `split_nccl(color, key=0, device_id=None, mode=...)`: Split + NCCL domain
- `nccl_supports_native(op)` / `nccl_supports_hybrid(op)`: Capability queries

### DistributedVector

```python
dtl.DistributedVector(ctx, size, dtype=np.float64, fill=None)
```

Properties:
- `global_size`: Total elements across all ranks
- `local_size`: Elements on this rank
- `local_offset`: Global index of first local element

Methods:
- `local_view()`: Get NumPy array view of local data
- `fill(value)`: Fill all local elements

### DistributedTensor

```python
dtl.DistributedTensor(ctx, shape, dtype=np.float64, fill=None)
```

Properties:
- `ndim`: Number of dimensions
- `shape`: Global shape tuple
- `local_shape`: Local shape on this rank
- `global_size`: Total elements
- `local_size`: Local elements

Methods:
- `local_view()`: Get NumPy array view of local data
- `fill(value)`: Fill all local elements

## Supported Data Types

| NumPy dtype | DTL Support |
|-------------|-------------|
| `np.float64` | Vector, Tensor |
| `np.float32` | Vector, Tensor |
| `np.int64` | Vector, Tensor |
| `np.int32` | Vector, Tensor |
| `np.uint64` | Vector |
| `np.uint32` | Vector |
| `np.uint8` | Vector |
| `np.int8` | Vector |
