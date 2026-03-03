# Legacy Deep-Dive: Views

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [05-views-iteration-and-data-access.md](05-views-iteration-and-data-access.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


Views are the central interface layer in DTL. They expose access and iteration semantics while constraining communication and invalidation behavior.

---

## Table of Contents

- [Overview](#overview)
- [local_view](#local_view)
  - [STL Compatibility](#stl-compatibility)
  - [No Communication Guarantee](#no-communication-guarantee)
- [global_view](#global_view)
  - [Global Indexing](#global-indexing)
  - [remote_ref Access](#remote_ref-access)
- [remote_ref](#remote_ref)
  - [Syntactic Loudness](#syntactic-loudness)
  - [Operations](#operations)
  - [When to Use](#when-to-use)
- [segmented_view](#segmented_view)
  - [The Performance Path](#the-performance-path)
  - [Segment Iteration](#segment-iteration)
- [View Validity and Invalidation](#view-validity-and-invalidation)
- [Best Practices](#best-practices)

---

## Overview

DTL provides four view types, each serving a distinct purpose:

| View | Purpose | Communication | Iterator Category |
|------|---------|---------------|-------------------|
| `local_view` | Local-only access | Never | Random-access |
| `global_view` | Global logical access | On `remote_ref` ops | N/A (returns `remote_ref`) |
| `segmented_view` | Bulk distributed iteration | Never (per-segment) | Forward (over segments) |
| `remote_ref<T>` | Explicit remote element | Explicit `get()/put()` | N/A (proxy type) |

### The DTL View Philosophy

DTL follows a clear hierarchy:

1. **Fast path**: Use `local_view` or `segmented_view` for bulk operations (no communication)
2. **Correct path**: Use `global_view` + `remote_ref` for sparse remote access (explicit communication)
3. **Forbidden path**: No implicit `T&` for potentially remote elements (prevents hidden communication)

---

## local_view

`local_view` provides STL-compatible access to locally-owned elements only.

### Basic Usage

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);
auto local = vec.local_view();

// Direct element access
local[0] = 42.0;
double val = local[10];

// Bounds-checked access
try {
    double x = local.at(999999);
} catch (const std::out_of_range& e) {
    // Index out of bounds
}

// Size information
std::size_t n = local.size();  // Number of local elements
bool empty = local.empty();
```

### STL Compatibility

`local_view` is fully compatible with STL algorithms:

```cpp
auto local = vec.local_view();

// Range-based for loop
for (double& x : local) {
    x *= 2.0;
}

// STL algorithms
std::sort(local.begin(), local.end());
std::fill(local.begin(), local.end(), 0.0);

auto sum = std::accumulate(local.begin(), local.end(), 0.0);
auto it = std::find(local.begin(), local.end(), 42.0);
auto count = std::count_if(local.begin(), local.end(),
                           [](double x) { return x > 0; });

// Reverse iteration
for (auto rit = local.rbegin(); rit != local.rend(); ++rit) {
    // Process in reverse
}

// Iterator arithmetic (random-access)
auto mid = local.begin() + local.size() / 2;
auto dist = std::distance(local.begin(), mid);
```

### No Communication Guarantee

**Critical guarantee**: `local_view` operations NEVER communicate.

This is enforced by design:
- Local views only access locally-owned elements
- All operations are pure local memory operations
- No network traffic, no MPI calls, no latency

```cpp
auto local = vec.local_view();

// These operations are ALL local-only:
local[0] = 1.0;                    // Direct memory write
double x = local[0];               // Direct memory read
std::sort(local.begin(), local.end());  // Local sort
auto sum = std::accumulate(...);   // Local accumulation
```

This guarantee makes `local_view` the primary interface for performance-critical code.

---

## global_view

`global_view` represents the logical global container with explicit remote access.

### Global Indexing

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);
auto global = vec.global_view();

// Global index space
dtl::size_type global_size = global.size();  // 1000 (total across all ranks)
```

### remote_ref Access

**Key principle**: Global indexing returns `remote_ref<T>`, not `T&`.

```cpp
auto global = vec.global_view();

// operator[] returns remote_ref<T>, NOT T&
auto ref = global[500];  // Type: remote_ref<double>

// You CANNOT do this:
// double& bad = global[500];  // COMPILE ERROR: no implicit conversion

// You MUST explicitly read/write:
double val = ref.get();   // Explicit read (may communicate)
ref.put(99.0);            // Explicit write (may communicate)
```

### ND Global Indexing

For tensors, global view uses ND indices:

```cpp
dtl::distributed_tensor<double, 2> mat({100, 100}, size, rank);
auto global = mat.global_view();

// ND global index
auto ref = global({50, 50});  // remote_ref<double> for element (50, 50)
double val = ref.get();
ref.put(42.0);
```

---

## remote_ref

`remote_ref<T>` is DTL's "syntactically loud" proxy for fine-grained remote access.

### Syntactic Loudness

DTL's core design principle requires that remote access be explicit. `remote_ref` achieves this by:

1. **No implicit conversion to `T&`** - You cannot accidentally get a reference
2. **No implicit conversion to `T*`** - You cannot accidentally get a pointer
3. **No implicit conversion to `bool`** - No implicit truth testing
4. **No implicit dereference** - Must call `.get()` explicitly

```cpp
auto global = vec.global_view();
auto ref = global[500];

// These all FAIL to compile:
// double& bad1 = ref;         // No implicit T& conversion
// double* bad2 = &ref;        // No implicit T* conversion
// if (ref) { }                // No implicit bool conversion
// double bad3 = *ref;         // No implicit dereference

// This is the ONLY way to read:
double val = ref.get();

// This is the ONLY way to write:
ref.put(42.0);
```

### Operations

#### Basic Read/Write

```cpp
auto ref = global[idx];

// Synchronous read
double val = ref.get();

// Synchronous write
ref.put(42.0);
```

#### Error Handling

Under result-based error policy:

```cpp
// get() returns result<T>
auto result = ref.get();
if (result.has_value()) {
    double val = result.value();
} else {
    auto error = result.error();
    // Handle communication error
}

// put() returns result<void>
auto put_result = ref.put(42.0);
if (!put_result) {
    // Handle write error
}
```

Under throwing error policy:

```cpp
try {
    double val = ref.get();
    ref.put(42.0);
} catch (const dtl::communication_error& e) {
    // Handle error
}
```

#### Identity Information

```cpp
auto ref = global[idx];

// Query the global index
auto global_idx = ref.global_index();

// Query the owning rank
dtl::rank_t owner = ref.owner();

// Check if local
bool is_local = ref.is_local();
```

### When to Use

**Use `remote_ref` for:**
- Debugging and correctness verification
- Sparse remote operations (few elements)
- Algorithms that need explicit remote access
- Prototyping before optimization

**Avoid `remote_ref` for:**
- Dense iteration over remote data
- Performance-critical inner loops
- Bulk operations (use halo exchange or redistribution instead)

```cpp
// BAD: Per-element remote access in a loop
auto global = vec.global_view();
double sum = 0.0;
for (dtl::size_type i = 0; i < global.size(); ++i) {
    sum += global[i].get();  // SLOW: one communication per element
}

// GOOD: Local computation + collective reduction
auto local = vec.local_view();
double local_sum = std::accumulate(local.begin(), local.end(), 0.0);
double global_sum;
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

---

## segmented_view

`segmented_view` is DTL's primary performance substrate for distributed algorithms.

### The Performance Path

The DTL performance model is:

1. **Iterate segments locally** (no communication)
2. **Compute local results**
3. **Communicate in bulk** (collectives, halo exchange)
4. **Repeat**

`segmented_view` enables step 1 efficiently.

### Basic Usage

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);
auto segv = vec.segmented_view();

// Iterate over local segments
for (auto& segment : segv.segments()) {
    // Each segment is a local-only view
    auto local_range = segment.local_range();

    for (double& x : local_range) {
        x *= 2.0;  // Process locally
    }
}
```

### Segment Iteration

Each segment provides:

```cpp
for (auto& segment : segv.segments()) {
    // Global index information
    auto global_start = segment.global_offset();
    auto global_end = segment.global_offset() + segment.size();

    // Local iterable range (STL-compatible)
    auto range = segment.local_range();

    // Use with STL algorithms
    std::transform(range.begin(), range.end(), range.begin(),
                   [](double x) { return x * x; });

    // Segment metadata
    auto seg_id = segment.id();  // Stable ID for debugging
}
```

### Segmented Distributed Algorithms

DTL algorithms are built on segmented iteration:

```cpp
// Distributed reduce pattern
template<typename Container, typename T, typename BinaryOp>
T distributed_reduce(Container& c, T init, BinaryOp op) {
    auto segv = c.segmented_view();

    // Step 1: Local partial reduction (no communication)
    T local_result = init;
    for (auto& segment : segv.segments()) {
        for (auto& x : segment.local_range()) {
            local_result = op(local_result, x);
        }
    }

    // Step 2: Global reduction (collective communication)
    T global_result;
    // MPI_Allreduce or similar...

    return global_result;
}
```

### No Communication Guarantee

Like `local_view`, `segmented_view` guarantees no communication during iteration:

```cpp
auto segv = vec.segmented_view();

// These operations are ALL local-only:
for (auto& seg : segv.segments()) {    // Local iteration
    for (auto& x : seg.local_range()) { // Local range access
        x = 0.0;                         // Local memory write
    }
}
```

Communication happens only when you explicitly call collective operations.

---

## View Validity and Invalidation

Views track structural epochs to ensure safety.

### Structural Operations Invalidate Views

Certain operations change the container's structure and invalidate all views:

| Operation | Invalidates Views? |
|-----------|-------------------|
| `resize()` | Yes |
| `redistribute()` | Yes |
| Element modification | No |
| `local_view()` access | No |

### Detection and Failure

DTL detects use of invalidated views:

```cpp
auto local = vec.local_view();

// Use view normally
local[0] = 42.0;

// Structural operation
vec.resize(2000);

// View is now INVALID
// Using it will fail deterministically:
local[0] = 1.0;  // Debug: assertion failure
                 // Release: returns structural_invalidation error
```

### Safe Pattern

Always obtain fresh views after structural operations:

```cpp
void process(dtl::distributed_vector<double>& vec) {
    auto local = vec.local_view();

    // Phase 1: Process
    for (double& x : local) {
        x *= 2.0;
    }

    // Phase 2: Resize
    vec.resize(vec.global_size() * 2);

    // Phase 3: Process again - GET FRESH VIEW
    auto fresh_local = vec.local_view();  // Must get new view
    for (double& x : fresh_local) {
        x += 1.0;
    }
}
```

### Epoch Checking

Views carry an epoch at creation:

```cpp
auto local = vec.local_view();
auto epoch_at_creation = local.epoch();

// After structural operation
vec.resize(2000);

// Views from before resize have stale epoch
// Container has advanced epoch
// Comparison detects staleness
```

---

## Best Practices

### 1. Prefer Local Views

For any operation on local data, use `local_view`:

```cpp
// GOOD: Local view for local operations
auto local = vec.local_view();
std::sort(local.begin(), local.end());

// BAD: Global view when you only need local data
auto global = vec.global_view();
for (std::size_t i = vec.global_offset(); i < vec.global_offset() + vec.local_size(); ++i) {
    auto ref = global[i];  // Unnecessary indirection
    double val = ref.get();
}
```

### 2. Use Segmented Views for Distributed Algorithms

```cpp
// GOOD: Segmented iteration
auto segv = vec.segmented_view();
double local_sum = 0.0;
for (auto& seg : segv.segments()) {
    for (double x : seg.local_range()) {
        local_sum += x;
    }
}

// Then collective reduction...
```

### 3. Bulk Communication Over Point-to-Point

```cpp
// BAD: Per-element remote access
for (dtl::size_type i = 0; i < 1000; ++i) {
    remote_data[i] = global[i].get();  // 1000 communications!
}

// GOOD: Use halo exchange or redistribution
auto halo = tensor.halo_view(1);
halo.exchange();  // One bulk communication
```

### 4. Check View Validity in Long-Running Code

```cpp
void long_computation(Container& c) {
    auto local = c.local_view();

    for (int iteration = 0; iteration < 1000; ++iteration) {
        // Process
        for (auto& x : local) {
            x = compute(x);
        }

        // If structure might change
        if (needs_resize(iteration)) {
            c.resize(new_size);
            local = c.local_view();  // Refresh view
        }
    }
}
```

### 5. Document Communication Points

Make communication explicit in your code:

```cpp
void distributed_compute(Container& c) {
    auto local = c.local_view();

    // Phase 1: Local computation (no communication)
    for (auto& x : local) {
        x = expensive_local_compute(x);
    }

    // COMMUNICATION POINT
    auto halo = c.halo_view(1);
    halo.exchange();  // <-- Communication here

    // Phase 2: Stencil with halo data
    // ...

    // COMMUNICATION POINT
    double local_result = local_reduce();
    double global_result;
    MPI_Allreduce(&local_result, &global_result, ...);  // <-- Communication here
}
```

---

## Summary

| View | Use For | Communication |
|------|---------|---------------|
| `local_view` | STL-like local operations | Never |
| `global_view` | Explicit global indexing | On `remote_ref.get()/put()` |
| `segmented_view` | Distributed algorithms | Never (bulk ops are separate) |
| `remote_ref` | Sparse remote access | Explicit on each operation |

**Key takeaway**: DTL makes communication explicit. Use local views for performance, remote_ref for correctness, and segmented views for scalable distributed algorithms.

---

## See Also

- [Containers Guide](containers.md) - Container types and construction
- [Algorithms Guide](algorithms.md) - DTL distributed algorithms
