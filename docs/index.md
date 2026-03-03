# Distributed Template Library (DTL)

DTL is a C++20 header-only template library providing STL-inspired abstractions for distributed and heterogeneous computing. It uses policy-based design with C++20 concepts to enable portable code across CPUs, GPUs, and multi-node systems.

**Current Version**: 0.1.0-alpha.1
**Status**: alpha pre-release

> **New to DTL?** Start with the [README](../README.md) for a quick overview and installation guide.

---

## Quick Links

| Getting Started | User Guide | Reference |
|-----------------|------------|-----------|
| [Installation & Quick Start](getting_started.md) | [User Guide Index](user_guide/index.md) | [API Reference](api_reference/) |
| [Hello Distributed Example](getting_started.md#hello-distributed) | [Containers](user_guide/04-distributed-containers.md) | [Developer Guide](developer_guide/index.md) |
| [Building from Source](getting_started.md#building-from-source) | [Views](user_guide/05-views-iteration-and-data-access.md) | [Developer Runtime/Handle](developer_guide/12-runtime-and-handle-development.md) |
| | [Policies](user_guide/06-policies-and-execution-control.md) | |
| | [Algorithms](user_guide/07-algorithms-collectives-and-remote-operations.md) | |
| | [Bindings](user_guide/08-language-bindings-overview.md) | |
| | [Runtime and Handles](user_guide/13-runtime-and-handle-model.md) | |

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide/index
api_reference/index
bindings/c_bindings
bindings/python_bindings
bindings/fortran_bindings
developer_guide/index
```

---

## Key Design Philosophy

**DTL does not pretend distribution is transparent.** Remote access is "syntactically loud" via `remote_ref<T>` (no implicit `T&` conversions), and communication costs are explicit in the API.

### Core Principles

1. **STL familiarity where honest** - Local views and iterators align with STL expectations
2. **Explicit distribution** - Ownership, partitioning, and communication are first-class concepts
3. **HPC-first architecture** - Compile-time binding and thin wrappers by default
4. **Orthogonal policy system** - Partition, placement, execution, consistency, and error policies are independent axes
5. **Segmented iteration** - The primary performance substrate for distributed algorithms

---

## Documentation Structure

### For New Users

Start with the [Getting Started Guide](getting_started.md) to build DTL and run your first distributed program.

### User Guide

The [User Guide](user_guide/) covers day-to-day usage of DTL:

- **[Environment](user_guide/environment.md)** - Backend lifecycle and context creation
- **[Containers](user_guide/containers.md)** - `distributed_vector`, `distributed_array`, `distributed_span`, `distributed_tensor`, `distributed_map`
- **[Views](user_guide/views.md)** - `local_view`, `global_view`, `segmented_view`, `remote_ref`
- **[Policies](user_guide/policies.md)** - Partition, placement, execution, consistency, error
- **[Algorithms](user_guide/algorithms.md)** - `for_each`, `transform`, `reduce`, `scan`, `sort`
- **[Error Handling](user_guide/error_handling.md)** - Result-based vs throwing patterns
- **[Language Bindings](user_guide/bindings.md)** - C, Python, and Fortran bindings

### Language Bindings

DTL supports multiple programming languages:

- **[C Bindings](bindings/c_bindings.md)** - Stable C ABI for interoperability
- **[Python Bindings](bindings/python_bindings.md)** - NumPy-integrated Python API
- **[Fortran Bindings](bindings/fortran_bindings.md)** - Via ISO_C_BINDING

### Migration Guide

- **[From STL](migration/from_stl.md)** - Mapping STL patterns to DTL equivalents

### Backend Guide

For backend developers and advanced users:

- **[Backend Concepts](backend_guide/concepts.md)** - Communicator, MemorySpace, Executor
- **[Backend Selection](backend_guide/backend_selection.md)** - When to use MPI vs CUDA
- **[Implementing a Backend](backend_guide/implementing_backend.md)** - Step-by-step guide

### Reference Documentation

- **[API Reference](api_reference/)** - Doxygen-generated documentation

---

## Quick Example

```cpp
#include <dtl/dtl.hpp>
#include <iostream>

int main() {
    // Create a distributed vector (standalone mode, no MPI required)
    dtl::distributed_vector<int> vec(100, /*num_ranks=*/1, /*my_rank=*/0);

    // Fill with values using local view (STL-compatible, no communication)
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i);
    }

    // Use DTL's for_each algorithm
    dtl::for_each(vec, [](int& x) { x = x * x; });

    // Local reduce (no MPI communication needed in single-rank mode)
    int sum = dtl::local_reduce(vec, 0, std::plus<>{});
    std::cout << "Sum of squares: " << sum << "\n";

    return 0;
}
```

---

## Alpha Limitations

The following features are planned for future releases:

- `distributed_vector::redistribute()` runtime partition changes
- `remote_ref` cross-rank get/put operations
- Network topology multi-host discovery
- Distributed map binding surface (C++ core available, bindings deferred)
- Remote/RPC binding surface (deferred)

---

## Supported Backends

| Backend | Status | Description |
|---------|--------|-------------|
| CPU (host_only) | Complete | Single-node, CPU-only execution |
| MPI | Complete | Multi-node distributed execution |
| MPI RMA | Complete | One-sided remote memory access |
| CUDA | Experimental | GPU acceleration (NVIDIA) |
| HIP | Experimental | GPU acceleration (AMD) |
| NCCL | Experimental | GPU collective communication |

### CUDA Placement Policies

DTL provides transparent GPU memory management through placement policies:

```cpp
// GPU-resident data on device 0
dtl::distributed_vector<float, dtl::device_only<0>> gpu_vec0(N, 1, 0);

// GPU-resident data on device 1 (different type, different device)
dtl::distributed_vector<float, dtl::device_only<1>> gpu_vec1(N, 1, 0);

// Each container carries device affinity
assert(gpu_vec0.device_id() == 0);
assert(gpu_vec1.device_id() == 1);

// Unified memory (host+device accessible)
dtl::distributed_vector<float, dtl::unified_memory> unified_vec(N, 1, 0);
auto local = unified_vec.local_view();  // Direct host access
local[0] = 42.0f;  // Write from host
transform_kernel<<<grid, block>>>(unified_vec.local_data(), n);  // Use on GPU

// GPU with host fallback (uses device if CUDA enabled, host otherwise)
dtl::distributed_vector<float, dtl::device_preferred> vec(N, 1, 0);
```

**Device Selection**: Allocations for `device_only<N>` are scoped to device N using RAII device guards. The caller's current CUDA context device is preserved across container construction and destruction.

See `examples/gpu/` for complete GPU examples.

---

## Requirements

- **C++ Standard**: C++20
- **Compilers**: GCC 11+, Clang 15+, MSVC 19.29+
- **Optional**: MPI (OpenMPI, MPICH), CUDA 11.4+

See [Getting Started](getting_started.md) for detailed build instructions.

---

## Project Links

- **Repository**: [GitHub](https://github.com/brycewestheimer/dtl-public)
- **Issue Tracker**: [GitHub Issues](https://github.com/brycewestheimer/dtl-public/issues)
