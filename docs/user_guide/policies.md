# Legacy Deep-Dive: Policies

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [06-policies-and-execution-control.md](06-policies-and-execution-control.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


DTL uses a policy-based design that separates concerns into orthogonal configuration axes. This allows flexible, compile-time configuration of distributed behavior.

---

## Table of Contents

- [Overview](#overview)
- [The Five Policy Axes](#the-five-policy-axes)
- [Partition Policies](#partition-policies)
- [Placement Policies](#placement-policies)
- [Execution Policies](#execution-policies)
- [Consistency Policies](#consistency-policies)
- [Error Policies](#error-policies)
- [Policy Composition](#policy-composition)
- [Policy Precedence](#policy-precedence)

---

## Overview

Distributed programming entangles multiple concerns:
- How data is partitioned across ranks
- Where data resides (host vs device memory)
- How operations execute (sync vs async)
- When writes become visible
- How errors are handled

DTL separates these into **five orthogonal policy axes**, allowing you to configure each independently.

### Why Policies?

```cpp
// Without policies: hardcoded behavior
distributed_vector<double> vec(1000);  // What partition? What memory? What error handling?

// With policies: explicit, configurable behavior
distributed_vector<double, block_partition<>, host_only> vec(1000);
// Or using policy_set for runtime composition
```

---

## The Five Policy Axes

| Axis | Question | Default |
|------|----------|---------|
| **Partition** | How is data divided across ranks? | `block_partition` |
| **Placement** | Where does data live (host/device)? | `host_only` |
| **Execution** | How do operations execute? | `seq` (synchronous) |
| **Consistency** | When are writes visible? | `bulk_synchronous` |
| **Error** | How are errors reported? | `expected` (result-based) |

---

## Partition Policies

Partition policies determine how global indices map to ranks.

### block_partition (Default)

Divides data into contiguous chunks:

```cpp
// 1000 elements across 4 ranks:
// Rank 0: indices [0, 250)
// Rank 1: indices [250, 500)
// Rank 2: indices [500, 750)
// Rank 3: indices [750, 1000)

dtl::distributed_vector<double, dtl::block_partition<>> vec(1000, size, rank);

// Block partition is the default
dtl::distributed_vector<double> vec_default(1000, size, rank);  // Same as above
```

Properties:
- Contiguous local storage
- Good cache locality
- Simple ownership queries
- Best for sequential access patterns

### cyclic_partition

Round-robin element distribution (planned):

```cpp
// 1000 elements across 4 ranks:
// Rank 0: indices 0, 4, 8, 12, ...
// Rank 1: indices 1, 5, 9, 13, ...
// Rank 2: indices 2, 6, 10, 14, ...
// Rank 3: indices 3, 7, 11, 15, ...

dtl::distributed_vector<double, dtl::cyclic_partition<>> vec(1000, size, rank);
```

Properties:
- Better load balancing for irregular access
- Non-contiguous local storage
- Higher overhead for sequential access

### block_cyclic_partition

Combines block and cyclic (planned):

```cpp
// Block size 64, cyclic distribution:
// Rank 0: indices [0,64), [256,320), ...
// Rank 1: indices [64,128), [320,384), ...
// etc.

dtl::distributed_vector<double, dtl::block_cyclic_partition<64>> vec(1000, size, rank);
```

Properties:
- Balance between locality and load balancing
- Standard in scientific computing (ScaLAPACK)

### hash_partition

Hash-based distribution (for associative containers):

```cpp
// Elements distributed by hash of key
dtl::distributed_unordered_map<std::string, int, dtl::hash_partition<>> map(size, rank);

// Custom hash function
dtl::distributed_unordered_map<Key, Value, dtl::hash_partition<MyHash>> map(size, rank);
```

### replicated

Full copy on each rank:

```cpp
// Every rank has complete copy
dtl::distributed_vector<double, dtl::replicated> lookup_table(1000, size, rank);
```

Properties:
- No communication for reads
- Writes require synchronization
- Memory scales with rank count

---

## Placement Policies

Placement policies determine where data resides physically.

### host_only (Default)

Data resides in host (CPU) memory:

```cpp
dtl::distributed_vector<double, dtl::block_partition<>, dtl::host_only> vec(1000, size, rank);

// host_only is the default
dtl::distributed_vector<double> vec_default(1000, size, rank);  // Same as above
```

Properties:
- Universal compatibility
- No GPU required
- Standard allocators

### device_only

Data resides in device (GPU) memory:

```cpp
// Requires DTL_ENABLE_CUDA or DTL_ENABLE_HIP
dtl::distributed_vector<double, dtl::block_partition<>, dtl::device_only<0>> vec(1000, size, rank);

// Access requires GPU kernels or explicit transfer
auto local = vec.local_view();  // Returns device pointer
```

Properties:
- Data stays on GPU
- Host access requires transfer
- Best for GPU-only workflows

### device_preferred

Prefers device memory with automatic fallback:

```cpp
dtl::distributed_vector<double, dtl::block_partition<>, dtl::device_preferred> vec(1000, size, rank);

// Uses GPU memory if available, host memory otherwise
```

### unified_memory

CUDA Unified Memory (managed memory):

```cpp
dtl::distributed_vector<double, dtl::block_partition<>, dtl::unified_memory> vec(1000, size, rank);

// Accessible from both host and device
// Automatic page migration
```

Properties:
- Convenience for mixed host/device access
- Performance implications from page faults
- Requires CUDA unified memory support

---

## Execution Policies

Execution policies control how operations are performed.

### seq (Default)

Synchronous, blocking execution:

```cpp
// Operation completes before returning
dtl::for_each(dtl::seq, vec, [](double& x) { x *= 2; });

// seq is the default
dtl::for_each(vec, [](double& x) { x *= 2; });  // Same as above
```

Properties:
- Simple to reason about
- Deterministic completion
- No concurrent execution

### par

Parallel execution (blocking):

```cpp
// Uses multiple threads, but still blocks until complete
dtl::for_each(dtl::par, vec, [](double& x) { x *= 2; });
```

Properties:
- Utilizes multiple CPU cores
- Still blocks caller
- Thread-safe functor required

### par_unseq

Parallel and vectorized (blocking):

```cpp
// Enables SIMD and multi-threading
dtl::for_each(dtl::par_unseq, vec, [](double& x) { x *= 2; });
```

Properties:
- Maximum CPU parallelism
- Functor must be vectorization-safe
- No synchronization in functor

### async

Non-blocking execution:

```cpp
// Returns immediately with a future
auto future = dtl::for_each(dtl::async, vec, [](double& x) { x *= 2; });

// Do other work...

// Wait for completion
future.wait();
```

Properties:
- Enables overlap of computation and communication
- Returns future/event handle
- Requires explicit synchronization

### Usage with Algorithms

```cpp
// Transform with parallel execution
dtl::transform(dtl::par, vec, output, [](double x) { return x * x; });

// Reduce with async execution
auto future = dtl::reduce(dtl::async, vec, 0.0, std::plus<>{});
// ... do other work ...
double result = future.get();
```

---

## Consistency Policies

Consistency policies define when writes become visible to other ranks.

### bulk_synchronous (Default)

BSP model with explicit barriers:

```cpp
// Writes not visible until barrier
dtl::distributed_vector<double, ..., dtl::bulk_synchronous> vec(1000, size, rank);

auto local = vec.local_view();
local[0] = 42.0;  // Local write

// Writes become visible after barrier
vec.barrier();
```

Properties:
- Clear synchronization points
- Simple reasoning about visibility
- Standard HPC model

### sequential_consistent

Strongest consistency (planned):

```cpp
dtl::distributed_vector<double, ..., dtl::sequential_consistent> vec(1000, size, rank);

// All operations appear in a single global order
// Higher synchronization overhead
```

### release_acquire

C++ memory model consistency (planned):

```cpp
// Writes in release-ordered operations visible to acquire-ordered readers
```

### relaxed

Minimal ordering (planned):

```cpp
// Only atomicity guaranteed, no ordering
// Maximum performance, complex reasoning
```

---

## Error Policies

Error policies determine how errors are reported.

### expected (Default)

Result-based error handling:

```cpp
dtl::distributed_vector<double, ..., dtl::expected> vec(1000, size, rank);

auto global = vec.global_view();
auto result = global[500].get();

if (result.has_value()) {
    double val = result.value();
} else {
    auto error = result.error();
    // Handle error
}
```

Properties:
- No exceptions
- Explicit error checking
- Compile-time enforced handling

### throwing

Exception-based error handling:

```cpp
dtl::distributed_vector<double, ..., dtl::throwing> vec(1000, size, rank);

try {
    auto global = vec.global_view();
    double val = global[500].get();  // Throws on error
} catch (const dtl::communication_error& e) {
    // Handle error
}
```

Properties:
- Familiar exception patterns
- Automatic propagation
- Cannot be ignored

---

## Policy Composition

### Using policy_set

Combine policies into a single set:

```cpp
using my_policies = dtl::policy_set<
    dtl::block_partition<>,
    dtl::host_only,
    dtl::par,
    dtl::bulk_synchronous,
    dtl::expected
>;

dtl::distributed_vector<double, my_policies> vec(1000, size, rank);
```

### Partial Specification

Unspecified axes use defaults:

```cpp
// Only specify partition, others use defaults
dtl::distributed_vector<double, dtl::cyclic_partition<>> vec(1000, size, rank);
// Equivalent to:
// dtl::distributed_vector<double, cyclic_partition<>, host_only, seq, bulk_synchronous, expected>
```

### Call-Site Override

Override policies per-operation:

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);  // Default policies

// Override execution policy for this call
dtl::for_each(dtl::par, vec, [](double& x) { x *= 2; });

// Override multiple policies
dtl::for_each(
    dtl::policy_set<dtl::par, dtl::async>{},
    vec,
    [](double& x) { x *= 2; }
);
```

---

## Policy Precedence

When multiple policy sources exist, precedence is:

1. **Call-site policy_set** (highest priority)
2. **Container-level defaults**
3. **Context default policy_set**
4. **Library defaults** (lowest priority)

```cpp
// Context with default parallel execution
auto ctx = dtl::context(dtl::policy_set<dtl::par>{});

// Container uses context default (par)
dtl::distributed_vector<double> vec(ctx, 1000, size, rank);

// Operation uses container default (par)
dtl::for_each(vec, func);  // Parallel execution

// Call-site override beats all
dtl::for_each(dtl::seq, vec, func);  // Sequential execution
```

### Conflict Detection

Conflicting policies at the same level cause errors:

```cpp
// COMPILE ERROR: two partition policies
dtl::distributed_vector<double,
    dtl::policy_set<dtl::block_partition<>, dtl::cyclic_partition<>>
> vec(1000, size, rank);
```

---

## Common Policy Combinations

### High-Performance Computing (Default)

```cpp
using hpc_policies = dtl::policy_set<
    dtl::block_partition<>,
    dtl::host_only,
    dtl::par,
    dtl::bulk_synchronous,
    dtl::expected
>;
```

### GPU Accelerated

```cpp
using gpu_policies = dtl::policy_set<
    dtl::block_partition<>,
    dtl::device_only<0>,
    dtl::par,
    dtl::bulk_synchronous,
    dtl::expected
>;
```

### Development/Debugging

```cpp
using debug_policies = dtl::policy_set<
    dtl::block_partition<>,
    dtl::host_only,
    dtl::seq,           // Sequential for easier debugging
    dtl::bulk_synchronous,
    dtl::throwing       // Exceptions for stack traces
>;
```

### Maximum Throughput

```cpp
using throughput_policies = dtl::policy_set<
    dtl::block_partition<>,
    dtl::device_preferred,
    dtl::par_unseq,
    dtl::bulk_synchronous,
    dtl::expected
>;
```

---

## See Also

- [Containers Guide](containers.md) - Using containers with policies
- [Algorithms Guide](algorithms.md) - Algorithm execution policies
- [Error Handling Guide](error_handling.md) - Error policy details
