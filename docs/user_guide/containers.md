# Legacy Deep-Dive: Containers

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [04-distributed-containers.md](04-distributed-containers.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


DTL provides distributed containers that partition data across multiple ranks while offering familiar STL-like interfaces for local operations.

---

## Table of Contents

- [Overview](#overview)
- [distributed_vector](#distributed_vector)
  - [Construction](#construction)
  - [Size and Capacity](#size-and-capacity)
  - [Element Access](#element-access)
  - [Iterators](#iterators)
  - [Modifiers](#modifiers)
- [distributed_array](#distributed_array)
  - [Key Differences from distributed_vector](#key-differences-from-distributed_vector)
  - [Construction](#construction-1)
  - [Size and Queries](#size-and-queries)
  - [Element Access](#element-access-1)
  - [No resize() Method](#no-resize-method)
- [distributed_span](#distributed_span)
  - [Construction](#construction-2)
  - [Core Operations](#core-operations)
  - [Lifetime and Invalidation](#lifetime-and-invalidation)
- [distributed_tensor](#distributed_tensor)
  - [ND Construction](#nd-construction)
  - [Indexing](#indexing)
  - [Layout](#layout)
- [Partitioning](#partitioning)
- [Common Patterns](#common-patterns)

---

## Overview

DTL containers differ from STL containers in key ways:

| Aspect | STL Container | DTL Container |
|--------|---------------|---------------|
| Memory | Single process | Distributed across ranks |
| Element access | Direct `T&` | Local: `T&`, Remote: `remote_ref<T>` |
| Iteration | Full container | Local partition (primary), segmented (bulk) |
| Size | `size()` | `local_size()`, `global_size()` |
| Modification | Local only | May require collective operations |

### Container Types

| Container | Description | Status |
|-----------|-------------|--------|
| `distributed_vector<T>` | 1D distributed sequence (resizable) | V1.0 |
| `distributed_array<T, N>` | 1D fixed-size distributed sequence | V1.0 |
| `distributed_span<T>` | 1D non-owning distributed span view | V1.0 |
| `distributed_tensor<T, Rank>` | ND distributed array | V1.0 |
| `distributed_map<K,V>` | Distributed hash map | V1.0 |
| `distributed_ordered_map<K,V>` | Distributed ordered map | Deferred |

---

## distributed_vector

`distributed_vector<T>` is a 1D distributed container that partitions elements across ranks using a block partition by default.

### Construction

#### Standalone Mode (No MPI)

```cpp
#include <dtl/dtl.hpp>

// Create a vector with 1000 elements on a single rank
dtl::distributed_vector<double> vec(1000, /*num_ranks=*/1, /*my_rank=*/0);
```

#### Multi-Rank Construction

```cpp
#include <dtl/dtl.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each rank participates; data is partitioned automatically
    dtl::distributed_vector<double> vec(1000, size, rank);

    // With 4 ranks, each owns ~250 elements
    // Rank 0: elements 0-249
    // Rank 1: elements 250-499
    // Rank 2: elements 500-749
    // Rank 3: elements 750-999

    MPI_Finalize();
    return 0;
}
```

#### Construction with Initial Value

```cpp
// All local elements initialized to 42.0
dtl::distributed_vector<double> vec(1000, size, rank, 42.0);
```

### Size and Capacity

```cpp
dtl::distributed_vector<int> vec(1000, size, rank);

// Global properties (same on all ranks)
dtl::size_type global = vec.global_size();  // 1000

// Local properties (rank-specific)
dtl::size_type local = vec.local_size();    // ~250 per rank
dtl::size_type offset = vec.global_offset(); // Starting global index

// Rank information
dtl::rank_t my_rank = vec.rank();
dtl::rank_t total = vec.num_ranks();

// Query ownership
dtl::rank_t owner = vec.owner_of(500);  // Which rank owns global index 500?
bool is_local = vec.is_local(500);      // Does this rank own index 500?
```

### Element Access

#### Local Access (Recommended)

For performance, always use `local_view()` when working with local elements:

```cpp
auto local = vec.local_view();

// STL-compatible access
local[0] = 1.0;                    // First local element
double val = local[local.size()-1]; // Last local element

// Bounds-checked access
try {
    local.at(999999);  // Throws if out of bounds
} catch (const std::out_of_range& e) {
    // Handle error
}

// Range-based for loop
for (double& x : local) {
    x *= 2.0;
}

// STL algorithm compatibility
std::sort(local.begin(), local.end());
auto it = std::find(local.begin(), local.end(), 42.0);
```

#### Global Access (Explicit Communication)

For remote elements, use `global_view()` which returns `remote_ref<T>`:

```cpp
auto global = vec.global_view();

// Global indexing returns remote_ref<T>, not T&
auto ref = global[500];  // Type: remote_ref<double>

// Explicit get/put operations (may communicate)
double val = ref.get();   // Read (may be remote)
ref.put(99.0);            // Write (may be remote)

// Local fast path: if index is local, operations are fast
// but syntax is the same (explicit is still required)
auto local_ref = global[vec.global_offset()];  // Known to be local
double fast = local_ref.get();  // Still requires .get()
```

### Iterators

DTL provides local iterators that are STL-compatible:

```cpp
auto local = vec.local_view();

// Iterator types
auto it = local.begin();   // Random-access iterator
auto end = local.end();

// STL algorithm compatibility
std::for_each(local.begin(), local.end(), [](double& x) { x *= 2; });
std::accumulate(local.begin(), local.end(), 0.0);

// Reverse iteration
for (auto rit = local.rbegin(); rit != local.rend(); ++rit) {
    // Process in reverse
}
```

### Modifiers

#### Local Modifications

Modifications to local elements are direct:

```cpp
auto local = vec.local_view();
local[0] = 42.0;  // Direct write

// Fill local partition
std::fill(local.begin(), local.end(), 0.0);
```

#### Structural Operations (Collective)

Resizing and redistribution are collective operations:

```cpp
// Resize requires all ranks to participate
vec.resize(2000);  // Global size changes; each rank gets new partition

// After resize, previous views are INVALIDATED
// auto stale = local;  // BAD: 'local' is now invalid
auto fresh = vec.local_view();  // OK: get a new view
```

---

## distributed_array

`distributed_array<T, N>` is a fixed-size 1D distributed container where the size `N` is a compile-time constant (template parameter).

### Key Differences from distributed_vector

| Feature | distributed_vector | distributed_array |
|---------|-------------------|-------------------|
| Size | Runtime, resizable | Compile-time constant |
| `resize()` | Supported | **Not available** |
| Use case | Dynamic data | Fixed-size data |
| Memory | Slightly more overhead | Minimal overhead |

### Construction

```cpp
#include <dtl/dtl.hpp>

// Fixed-size array with 1000 elements
dtl::distributed_array<double, 1000> arr(size, rank);

// With initial value
dtl::distributed_array<double, 1000> arr(size, rank, 0.0);
```

### Size and Queries

```cpp
dtl::distributed_array<int, 1000> arr(size, rank);

// Compile-time constant
constexpr dtl::size_type N = arr.extent;  // 1000

// Runtime queries (same as vector)
dtl::size_type global = arr.global_size();   // 1000
dtl::size_type local = arr.local_size();     // ~250 per rank
dtl::size_type offset = arr.global_offset(); // Starting global index
```

### Element Access

Array access is identical to vector:

```cpp
// Local view (recommended)
auto local = arr.local_view();
local[0] = 42.0;

// Range-based iteration
for (double& x : local) {
    x *= 2.0;
}

// Global view (with remote_ref)
auto global = arr.global_view();
auto ref = global[500];  // remote_ref<double>
```

### No resize() Method

Unlike `distributed_vector`, arrays cannot be resized:

```cpp
// arr.resize(2000);  // ERROR: resize() does not exist

// If you need resizing, use distributed_vector instead:
dtl::distributed_vector<double> vec(1000, size, rank);
vec.resize(2000);  // OK
```

---

## distributed_span

`distributed_span<T>` is a non-owning distributed analog of `std::span<T>`. It provides local contiguous access and distributed metadata (`rank`, `num_ranks`, global size).

### Construction

```cpp
#include <dtl/containers/distributed_span.hpp>

// From a distributed container (borrowed memory)
dtl::distributed_vector<int> vec(1000, size, rank);
dtl::distributed_span<int> s1(vec);

// From explicit pointer and distributed metadata
dtl::distributed_span<int> s2(
    vec.local_data(),
    vec.local_size(),
    vec.size(),
    vec.rank(),
    vec.num_ranks()
);
```

### Core Operations

```cpp
auto global_n = s1.size();
auto local_n = s1.local_size();
int* ptr = s1.data();

// Local contiguous semantics
s1[0] = 42;
auto head = s1.first(16);
auto tail = s1.subspan(16);

// Rank metadata
auto me = s1.rank();
auto p = s1.num_ranks();
```

### Lifetime and Invalidation

- `distributed_span` does not own data.
- Owner/container lifetime must exceed the span lifetime.
- Structural changes in owner containers can invalidate span assumptions.
- Recreate spans after owner `resize()`/`redistribute()` operations.

---

## distributed_tensor

`distributed_tensor<T, Rank>` is an ND distributed container for multi-dimensional arrays.

### ND Construction

```cpp
#include <dtl/dtl.hpp>

// 2D tensor: 100 x 100 matrix
dtl::distributed_tensor<double, 2> matrix({100, 100}, size, rank);

// 3D tensor: 64 x 64 x 64 cube
dtl::distributed_tensor<float, 3> cube({64, 64, 64}, size, rank);

// With initial value
dtl::distributed_tensor<int, 2> grid({50, 50}, size, rank, 0);
```

### Indexing

#### Global Extents

```cpp
dtl::distributed_tensor<double, 2> mat({100, 100}, size, rank);

// Global shape
auto extents = mat.global_extents();
std::size_t rows = extents[0];  // 100
std::size_t cols = extents[1];  // 100

// Local shape (partition-dependent)
auto local_ext = mat.local_extents();
```

#### Local mdspan Access

```cpp
// Get local mdspan for efficient ND access
auto local = mat.local_mdspan();

// ND indexing into local storage
for (std::size_t i = 0; i < local.extent(0); ++i) {
    for (std::size_t j = 0; j < local.extent(1); ++j) {
        local(i, j) = static_cast<double>(i * 100 + j);
    }
}
```

#### Views

```cpp
// Local view (flattened, 1D iteration)
auto local_view = mat.local_view();
for (double& x : local_view) {
    x = 0.0;
}

// Segmented view (for bulk operations)
auto seg_view = mat.segmented_view();
for (auto& segment : seg_view.segments()) {
    // Process each segment locally
}

// Global view (with remote_ref for remote access)
auto global = mat.global_view();
auto ref = global({50, 50});  // ND global index
double val = ref.get();
```

### Layout

DTL tensors use row-major (C-style) layout by default:

```cpp
// Row-major: last index varies fastest
// Element (i, j) is at offset: i * cols + j
dtl::distributed_tensor<double, 2> mat({3, 4}, 1, 0);

auto local = mat.local_mdspan();
// Memory layout: [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] ...
```

---

## distributed_map

`distributed_map<K, V>` is a distributed hash map that partitions key-value pairs across ranks using hash partitioning.

### Construction

```cpp
#include <dtl/dtl.hpp>

// Create a distributed hash map
dtl::distributed_map<std::string, double> map(size, rank);
```

### Operations

```cpp
// Insert (local key - key hashes to this rank)
map.insert("key1", 42.0);

// Check if a key is local
if (map.is_local("key1")) {
    // Key hashes to this rank
}

// Lookup (local key)
auto it = map.find("key1");
if (it != map.end()) {
    double val = it->second;
}

// Check ownership before insert
dtl::rank_t owner = map.owner_of("key2");
if (owner == rank) {
    map.insert("key2", 99.0);
}
```

### Iteration

```cpp
// Iterate over local entries only
for (const auto& [key, value] : map.local_view()) {
    std::cout << key << ": " << value << "\n";
}
```

### V1.0.0 Limitations

- Remote key insert/erase not yet implemented (requires RPC)
- Use `owner_of()` to check ownership before operations
- Use `is_local()` to filter operations to local keys

---

## Partitioning

DTL containers support different partitioning strategies:

### Block Partition (Default)

Contiguous chunks assigned to each rank:

```cpp
// 1000 elements, 4 ranks
// Rank 0: [0, 250)
// Rank 1: [250, 500)
// Rank 2: [500, 750)
// Rank 3: [750, 1000)
dtl::distributed_vector<int> vec(1000, 4, my_rank);
```

### Cyclic Partition

Round-robin assignment (planned):

```cpp
// dtl::distributed_vector<int, dtl::cyclic_partition<>> vec(...);
// Rank 0: indices 0, 4, 8, ...
// Rank 1: indices 1, 5, 9, ...
// etc.
```

### Block-Cyclic Partition

Combines block and cyclic (planned):

```cpp
// dtl::distributed_vector<int, dtl::block_cyclic_partition<64>> vec(...);
// Blocks of 64 elements distributed cyclically
```

---

## Common Patterns

### Initialize Local Data

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);
auto local = vec.local_view();

// Each rank initializes its portion based on global indices
for (dtl::size_type i = 0; i < local.size(); ++i) {
    dtl::size_type global_idx = vec.global_offset() + i;
    local[i] = std::sin(static_cast<double>(global_idx) * 0.01);
}
```

### Local Computation with Global Reduction

```cpp
// Compute local partial sum
auto local = vec.local_view();
double local_sum = std::accumulate(local.begin(), local.end(), 0.0);

// Global reduction (requires MPI)
double global_sum;
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

### Transform All Elements

```cpp
dtl::distributed_vector<int> vec(1000, size, rank);

// Using DTL algorithm
dtl::for_each(vec, [](int& x) { x *= 2; });

// Or manually with local view
auto local = vec.local_view();
std::transform(local.begin(), local.end(), local.begin(),
               [](int x) { return x * 2; });
```

### Stencil with Halo Exchange

```cpp
dtl::distributed_tensor<double, 2> grid({100, 100}, size, rank);

// Define halo regions (neighbors needed for stencil)
auto halo = grid.halo_view(/*width=*/1);

// Exchange halo data with neighbors (collective)
halo.exchange();

// Now compute stencil using local data + halo
auto local = grid.local_mdspan();
auto halo_data = halo.local_mdspan();

for (std::size_t i = 1; i < local.extent(0) - 1; ++i) {
    for (std::size_t j = 1; j < local.extent(1) - 1; ++j) {
        // 5-point stencil
        grid_new(i, j) = 0.25 * (
            local(i-1, j) + local(i+1, j) +
            local(i, j-1) + local(i, j+1)
        );
    }
}
```

### View Validity

Views are invalidated by structural operations. Always obtain fresh views after resize/redistribute:

```cpp
auto local = vec.local_view();

// ... use local ...

vec.resize(2000);  // STRUCTURAL OPERATION

// local is now INVALID - do not use!
// Using it will fail deterministically (debug assertion or error)

auto fresh_local = vec.local_view();  // Get fresh view
// ... use fresh_local ...
```

---

## Performance Guidelines

1. **Use `local_view()` for local iteration** - No communication overhead
2. **Prefer segmented operations** - Better for bulk distributed algorithms
3. **Avoid per-element global access in loops** - Each `.get()/.put()` may communicate
4. **Use halo exchange for stencils** - Bulk communication is more efficient than point-to-point
5. **Minimize structural operations** - Resize/redistribute require collective communication

---

## See Also

- [Views Guide](views.md) - Detailed view semantics
- [Algorithms Guide](algorithms.md) - DTL distributed algorithms
- [Policies Guide](policies.md) - Partitioning and other policies
