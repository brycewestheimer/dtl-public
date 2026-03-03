# STL to DTL Migration Guide

This guide helps you migrate code from STL containers and algorithms to DTL equivalents for distributed computing.

---

## Table of Contents

- [Philosophy](#philosophy)
- [Container Mapping](#container-mapping)
- [View Mapping](#view-mapping)
- [Algorithm Mapping](#algorithm-mapping)
- [Common Patterns](#common-patterns)
- [Pitfalls to Avoid](#pitfalls-to-avoid)
- [Migration Checklist](#migration-checklist)

---

## Philosophy

DTL is **STL-inspired but not STL-identical**. Key differences:

| Aspect | STL | DTL |
|--------|-----|-----|
| Memory model | Single process | Distributed across ranks |
| Element access | Always local `T&` | Local `T&`, Remote `remote_ref<T>` |
| Iteration | Full container | Local partition (primary) |
| Communication | N/A | Explicit |
| Error handling | Exceptions | Configurable (result or exceptions) |

**Core principle**: DTL makes distribution explicit. Code that "just works" in STL may need restructuring in DTL.

---

## Container Mapping

### std::vector → distributed_vector

**STL:**
```cpp
std::vector<double> vec(1000);
vec[0] = 42.0;
double x = vec[500];
```

**DTL:**
```cpp
// Standalone mode (single rank)
dtl::distributed_vector<double> vec(1000, 1, 0);

// Multi-rank mode
dtl::distributed_vector<double> vec(1000, num_ranks, my_rank);

// Local access via local_view
auto local = vec.local_view();
local[0] = 42.0;  // First local element

// Global access via global_view (explicit)
auto global = vec.global_view();
double x = global[500].get();  // May communicate
```

### std::array/std::span → local_view

Local views provide STL-compatible access:

**STL:**
```cpp
std::span<double> span(data, size);
for (auto& x : span) {
    x *= 2;
}
```

**DTL:**
```cpp
auto local = vec.local_view();
for (auto& x : local) {
    x *= 2;
}
// Identical iteration pattern
```

### Multi-dimensional: std::mdspan → distributed_tensor

**STL (C++23):**
```cpp
std::mdspan<double, std::extents<100, 100>> mat(data);
mat(50, 50) = 42.0;
```

**DTL:**
```cpp
dtl::distributed_tensor<double, 2> mat({100, 100}, num_ranks, my_rank);

// Local mdspan access
auto local = mat.local_mdspan();
local(i, j) = 42.0;  // Local indices

// Global access
auto global = mat.global_view();
global({50, 50}).put(42.0);  // Global indices, may communicate
```

---

## View Mapping

### Direct Reference vs remote_ref

**STL (direct reference):**
```cpp
double& ref = vec[i];
ref = 42.0;  // Direct write
double x = ref;  // Direct read
```

**DTL (local view - same as STL):**
```cpp
auto local = vec.local_view();
double& ref = local[i];  // Direct reference to local element
ref = 42.0;
double x = ref;
```

**DTL (global view - explicit remote):**
```cpp
auto global = vec.global_view();
auto ref = global[i];  // remote_ref<double>, NOT double&

// Must use explicit operations
ref.put(42.0);         // Write
double x = ref.get();  // Read

// These DO NOT compile:
// double& bad = global[i];  // Error: no implicit conversion
// *global[i] = 42.0;        // Error: no implicit dereference
```

### Range-Based For Loops

**STL:**
```cpp
for (double& x : vec) {
    x *= 2;
}
```

**DTL (local):**
```cpp
// Local view: identical pattern
for (double& x : vec.local_view()) {
    x *= 2;
}
```

**DTL (global):**
```cpp
// NO direct global iteration - use segmented view
auto segv = vec.segmented_view();
for (auto& segment : segv.segments()) {
    for (double& x : segment.local_range()) {
        x *= 2;
    }
}
```

---

## Algorithm Mapping

### std::for_each

**STL:**
```cpp
std::for_each(vec.begin(), vec.end(), [](double& x) { x *= 2; });
```

**DTL:**
```cpp
// Operates on local partition
dtl::for_each(vec, [](double& x) { x *= 2; });

// Or with explicit local view
auto local = vec.local_view();
std::for_each(local.begin(), local.end(), [](double& x) { x *= 2; });
```

### std::transform

**STL:**
```cpp
std::transform(input.begin(), input.end(), output.begin(),
               [](double x) { return x * x; });
```

**DTL:**
```cpp
dtl::transform(input, output, [](double x) { return x * x; });
```

### std::accumulate / std::reduce

**STL:**
```cpp
double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
```

**DTL (local only):**
```cpp
double local_sum = dtl::local_reduce(vec, 0.0, std::plus<>{});
```

**DTL (global):**
```cpp
double global_sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
// All ranks participate; all ranks get the result
```

### std::sort

**STL:**
```cpp
std::sort(vec.begin(), vec.end());
```

**DTL (local):**
```cpp
dtl::local_sort(vec);  // Sorts local partition only
```

**DTL (global):**
```cpp
dtl::distributed_sort(vec);  // Global sort, collective operation
```

### std::find / std::find_if

**STL:**
```cpp
auto it = std::find(vec.begin(), vec.end(), value);
```

**DTL (local):**
```cpp
auto local = vec.local_view();
auto it = std::find(local.begin(), local.end(), value);
```

**DTL (global):**
```cpp
// No built-in global find; must implement with collective
auto local = vec.local_view();
auto it = std::find(local.begin(), local.end(), value);
bool found_local = (it != local.end());

// Collective to determine if found globally
bool found_global;
MPI_Allreduce(&found_local, &found_global, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
```

---

## Common Patterns

### Pattern 1: Initialize All Elements

**STL:**
```cpp
std::vector<double> vec(1000);
for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<double>(i);
}
```

**DTL:**
```cpp
dtl::distributed_vector<double> vec(1000, num_ranks, my_rank);
auto local = vec.local_view();

for (size_t i = 0; i < local.size(); ++i) {
    // Compute global index
    size_t global_idx = vec.global_offset() + i;
    local[i] = static_cast<double>(global_idx);
}
```

### Pattern 2: Compute Sum

**STL:**
```cpp
double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
```

**DTL:**
```cpp
// Local sum only
double local_sum = dtl::local_reduce(vec, 0.0, std::plus<>{});

// Global sum
double global_sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
```

### Pattern 3: Find Min/Max

**STL:**
```cpp
auto [min_it, max_it] = std::minmax_element(vec.begin(), vec.end());
double min_val = *min_it;
double max_val = *max_it;
```

**DTL:**
```cpp
// Local min/max
auto local = vec.local_view();
auto [local_min_it, local_max_it] = std::minmax_element(local.begin(), local.end());
double local_min = *local_min_it;
double local_max = *local_max_it;

// Global min/max
double global_min, global_max;
MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
```

### Pattern 4: Filter Elements

**STL:**
```cpp
std::vector<int> filtered;
std::copy_if(vec.begin(), vec.end(), std::back_inserter(filtered),
             [](int x) { return x > 0; });
```

**DTL:**
```cpp
// Filter locally
auto local = vec.local_view();
std::vector<int> local_filtered;
std::copy_if(local.begin(), local.end(), std::back_inserter(local_filtered),
             [](int x) { return x > 0; });

// Note: local_filtered is local to each rank
// For global filtered result, need explicit redistribution
```

### Pattern 5: Dot Product

**STL:**
```cpp
double dot = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
```

**DTL:**
```cpp
// Transform-reduce pattern
auto local_a = a.local_view();
auto local_b = b.local_view();

double local_dot = std::inner_product(local_a.begin(), local_a.end(),
                                      local_b.begin(), 0.0);

double global_dot;
MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
```

---

## Pitfalls to Avoid

### Pitfall 1: Ignoring Distribution

**Wrong:**
```cpp
// Assumes all elements are local
for (size_t i = 0; i < vec.global_size(); ++i) {
    vec.global_view()[i].put(i);  // Massive communication!
}
```

**Correct:**
```cpp
auto local = vec.local_view();
for (size_t i = 0; i < local.size(); ++i) {
    local[i] = vec.global_offset() + i;  // Local only
}
```

### Pitfall 2: Forgetting Collective Participation

**Wrong:**
```cpp
if (rank == 0) {
    auto sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});  // DEADLOCK
}
```

**Correct:**
```cpp
auto sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});  // All ranks
if (rank == 0) {
    std::cout << "Sum: " << sum << "\n";
}
```

### Pitfall 3: Using Stale Views

**Wrong:**
```cpp
auto local = vec.local_view();
// ... operations ...
vec.resize(2000);
local[0] = 42.0;  // ERROR: view invalidated
```

**Correct:**
```cpp
auto local = vec.local_view();
// ... operations ...
vec.resize(2000);
local = vec.local_view();  // Refresh
local[0] = 42.0;  // OK
```

### Pitfall 4: Assuming Global Order

**Wrong:**
```cpp
dtl::local_sort(vec);  // Only sorts local partition!
// Elements are NOT globally sorted
```

**Correct:**
```cpp
dtl::distributed_sort(vec);  // Global sort
// Now globally sorted
```

### Pitfall 5: Implicit Conversions

**Won't Compile:**
```cpp
auto global = vec.global_view();
double x = global[500];  // Error: remote_ref<double> not implicitly convertible
```

**Correct:**
```cpp
auto global = vec.global_view();
double x = global[500].get();  // Explicit
```

---

## Migration Checklist

Use this checklist when migrating STL code to DTL:

### Container Migration
- [ ] Replace `std::vector<T>` with `dtl::distributed_vector<T>`
- [ ] Add rank information to constructors (`num_ranks`, `my_rank`)
- [ ] Review all element access for local vs global needs

### Access Pattern Migration
- [ ] Replace direct `vec[i]` with `local_view()[i]` for local access
- [ ] Use `global_view()[i].get()/put()` for global access
- [ ] Update range-based for loops to use `local_view()`

### Algorithm Migration
- [ ] Replace `std::accumulate` with `dtl::local_reduce` or `dtl::distributed_reduce`
- [ ] Replace `std::sort` with `dtl::local_sort` or `dtl::distributed_sort`
- [ ] Update all algorithms to operate on appropriate views

### Communication Points
- [ ] Identify where global results are needed
- [ ] Add collective operations for cross-rank data
- [ ] Ensure all ranks participate in collective calls

### Error Handling
- [ ] Choose error policy (`expected` or `throwing`)
- [ ] Add error checking for operations that may fail
- [ ] Handle collective errors appropriately

### Testing
- [ ] Test with single rank (standalone mode)
- [ ] Test with multiple ranks
- [ ] Verify results match sequential reference

---

## Quick Reference

| STL | DTL Local | DTL Global |
|-----|-----------|------------|
| `vec[i]` | `local[i]` | `global[i].get()` |
| `vec.size()` | `vec.local_size()` | `vec.global_size()` |
| `for (auto& x : vec)` | `for (auto& x : local)` | Use segmented_view |
| `std::accumulate` | `dtl::local_reduce` | `dtl::distributed_reduce` |
| `std::sort` | `dtl::local_sort` | `dtl::distributed_sort` |
| `std::transform` | `dtl::transform` | `dtl::transform` |

---

## See Also

- [Containers Guide](../user_guide/containers.md)
- [Views Guide](../user_guide/views.md)
- [Algorithms Guide](../user_guide/algorithms.md)
