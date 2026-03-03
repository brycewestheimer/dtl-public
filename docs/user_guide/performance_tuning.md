# Legacy Deep-Dive: Performance Tuning

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [10-performance-tuning-and-scaling.md](10-performance-tuning-and-scaling.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


This guide covers the key decisions that affect DTL application performance, from view selection and execution policies to memory placement and communication patterns.

---

## Table of Contents

- [View Selection](#view-selection)
  - [local_view: The Fastest Path](#local_view-the-fastest-path)
  - [segmented_view: The Distributed Algorithm Path](#segmented_view-the-distributed-algorithm-path)
  - [global_view: The Correctness Path](#global_view-the-correctness-path)
  - [View Selection Summary](#view-selection-summary)
- [Execution Policy Selection](#execution-policy-selection)
  - [seq: Sequential Execution](#seq-sequential-execution)
  - [par: Parallel Execution](#par-parallel-execution)
  - [async: Asynchronous Execution](#async-asynchronous-execution)
  - [Execution Policy Decision Tree](#execution-policy-decision-tree)
- [Placement Policy Impact](#placement-policy-impact)
  - [host_only (Default)](#host_only-default)
  - [device_only](#device_only)
  - [unified_memory](#unified_memory)
  - [device_preferred](#device_preferred)
  - [Placement Policy Comparison](#placement-policy-comparison)
- [Avoiding remote_ref in Loops](#avoiding-remote_ref-in-loops)
- [Batch vs Single-Element Operations on distributed_map](#batch-vs-single-element-operations-on-distributed_map)
- [MPI Thread Level Selection](#mpi-thread-level-selection)
- [NCCL vs MPI for GPU Collectives](#nccl-vs-mpi-for-gpu-collectives)
- [Memory Allocation Patterns](#memory-allocation-patterns)
  - [Pre-allocate Containers](#pre-allocate-containers)
  - [Reuse Views](#reuse-views)
  - [Minimize Structural Operations](#minimize-structural-operations)

---

## View Selection

Choosing the right view is the single most impactful performance decision in DTL. Each view type has fundamentally different communication characteristics.

### local_view: The Fastest Path

`local_view` provides zero-overhead, contiguous access to the local partition. Its iterators are raw pointers (`T*`), making it fully compatible with STL algorithms, SIMD vectorization, and compiler auto-optimization.

**When to use:**
- All operations that only need local data
- STL algorithm integration
- Performance-critical inner loops
- Any situation where you do not need remote data

```cpp
auto local = vec.local_view();

// Direct pointer access -- identical to std::vector::data()
double* ptr = local.data();
std::size_t n = local.size();

// Fully vectorizable loop
for (std::size_t i = 0; i < n; ++i) {
    ptr[i] = std::sin(ptr[i]);
}

// STL algorithms work at full speed
std::sort(local.begin(), local.end());
auto sum = std::accumulate(local.begin(), local.end(), 0.0);
```

**Performance characteristics:**
- Zero communication overhead
- Contiguous random-access iterators (raw pointers)
- Compiler can auto-vectorize loops
- Cache-friendly sequential access
- Construction cost: O(1)

### segmented_view: The Distributed Algorithm Path

`segmented_view` is the primary iteration substrate for distributed algorithms. It provides iteration over segments (one per rank), where each segment exposes a `local_view` for data access. A one-time O(p) offset cache is built at construction.

**When to use:**
- Writing distributed algorithms (local compute + collective communication)
- Iterating over the distributed structure
- When you need both local data and distribution metadata

```cpp
auto segv = vec.segmented_view();

// Efficient: iterate only local segment, skip remote
segv.for_each_local([](double& x) {
    x = std::sqrt(x);
});

// Or use the local_segment() shortcut
auto seg = segv.local_segment();
for (auto& x : seg) {
    x *= 2.0;
}
```

**Performance characteristics:**
- No communication during iteration
- O(p) construction cost (offset cache, where p = number of ranks)
- O(1) per-segment offset lookups after construction
- Local segments provide `local_view` (same contiguous access)
- Slight overhead compared to `local_view()` directly (segment descriptor copy)

**When to prefer `local_view()` over `segmented_view()`:**

If you only need the local data and do not need segment metadata (rank, global_offset), call `local_view()` directly. It avoids the O(p) offset cache construction.

```cpp
// Prefer this for simple local operations:
auto local = vec.local_view();
std::transform(local.begin(), local.end(), local.begin(),
               [](double x) { return x * x; });

// Use segmented_view when you need distribution awareness:
auto segv = vec.segmented_view();
auto seg = segv.local_segment();
index_t global_start = seg.global_offset;  // Need this metadata
```

### global_view: The Correctness Path

`global_view` provides a logical global index space. Every access returns `remote_ref<T>`, even for local elements. This is by design to make communication costs explicit.

**When to use:**
- Debugging and correctness verification
- Sparse, infrequent remote element access
- Prototyping before optimization
- Algorithms that genuinely need global indexing

**When NOT to use:**
- Performance-critical inner loops
- Dense iteration over any elements (local or remote)
- Bulk data access patterns

```cpp
auto global = vec.global_view();

// Every access returns remote_ref -- even local elements
auto ref = global[idx];     // remote_ref<double>
double val = ref.get().value();  // Explicit, may communicate

// NEVER do this in a loop:
// for (index_t i = 0; i < global.size(); ++i) {
//     sum += global[i].get().value();  // O(n) communications!
// }
```

**Performance characteristics:**
- O(1) per-access, but each access returns `remote_ref` (indirection)
- Remote element access triggers communication (RMA get/put)
- Even local element access has `remote_ref` wrapper overhead
- No built-in batching -- each `get()/put()` is independent

### View Selection Summary

| Scenario | Recommended View | Reason |
|----------|-----------------|--------|
| Local computation | `local_view()` | Zero overhead, STL compatible |
| Distributed algorithm | `segmented_view()` | Distribution-aware, no communication |
| Sparse remote access | `global_view()` | Explicit communication via `remote_ref` |
| Performance-critical loop | `local_view()` | Raw pointer iterators, vectorizable |
| Need global offset metadata | `segmented_view()` | Provides `global_offset` per segment |
| Debugging | `global_view()` | Shows logical global structure |

**Rule of thumb:** Start with `local_view()`. Use `segmented_view()` when you need distribution metadata. Use `global_view()` only for sparse remote access or prototyping.

---

## Execution Policy Selection

Execution policies control how local computation within an algorithm is dispatched. They do not affect communication patterns -- communication is determined by the algorithm itself.

### seq: Sequential Execution

Single-threaded, blocking, deterministic.

```cpp
dtl::for_each(dtl::seq{}, vec, [](double& x) { x *= 2.0; });
```

**Characteristics:**
- Deterministic element processing order
- No threading overhead
- SIMD vectorization is still allowed
- Easiest to debug

**Best for:**
- Small data sizes (< 10K elements per rank)
- Operations with complex dependencies
- Debugging and development
- When deterministic ordering is required

### par: Parallel Execution

Multi-threaded within a rank, blocking completion.

```cpp
dtl::for_each(dtl::par{}, vec, [](double& x) { x *= 2.0; });

// Explicit thread count
dtl::for_each(dtl::par_n<4>{}, vec, [](double& x) { x *= 2.0; });
```

**Characteristics:**
- Multi-threaded local computation (uses `cpu_executor` thread pool)
- Non-deterministic element processing order
- Call blocks until all threads finish
- User-provided functions must be thread-safe

**Best for:**
- Large local partitions (> 100K elements per rank)
- Compute-bound operations (not memory-bound)
- Independent element operations (map, filter, transform)
- When you have spare CPU cores (i.e., not oversubscribed with MPI)

**Thread count guidance:**

With MPI, each rank is a process. If you have `R` ranks per node and `C` cores per node, set threads to `C / R`:

```cpp
unsigned int threads_per_rank = std::thread::hardware_concurrency() / ranks_per_node;
dtl::for_each(dtl::par_n<threads_per_rank>{}, vec, f);
```

### async: Asynchronous Execution

Non-blocking, returns `distributed_future<T>`.

```cpp
auto future = dtl::async_reduce(vec, 0.0, std::plus<>{});
// ... overlap with other work ...
double result = future.get();  // Block when result is needed
```

**Characteristics:**
- Returns immediately with a future
- Enables overlap of computation and communication
- Supports continuation chaining via `.then()`
- Requires polling or background progress for completion

**Best for:**
- Overlapping communication with computation
- Pipelining multiple operations
- Non-blocking collective operations
- Latency hiding in multi-phase algorithms

**Progress requirements:**

Async operations require progress to complete. Either poll explicitly or enable background progress:

```cpp
// Option 1: Explicit polling
auto future = dtl::async_reduce(vec, 0.0, std::plus<>{});
while (!future.is_ready()) {
    dtl::futures::progress_engine::instance().poll();
    // do other work
}

// Option 2: Just call .get() -- it polls internally
double result = future.get();
```

### Execution Policy Decision Tree

```
Is data size < 10K elements per rank?
  YES --> Use seq{}
  NO  --> Is the operation compute-bound (not memory-bound)?
            YES --> Are there spare CPU cores?
                      YES --> Use par{}
                      NO  --> Use seq{} (avoid oversubscription)
            NO  --> Do you need to overlap with communication?
                      YES --> Use async{}
                      NO  --> Use seq{} (memory-bound won't benefit from par)
```

---

## Placement Policy Impact

Placement policies determine where container data resides in memory. The choice affects both access patterns and performance.

### host_only (Default)

Data resides in CPU memory. This is the default and the simplest option.

```cpp
dtl::distributed_vector<float> vec(1000);  // Implicitly host_only
dtl::distributed_vector<float, dtl::host_only> vec_explicit(1000);
```

- No GPU interaction needed
- Direct CPU access, no copies
- STL algorithms work directly
- Suitable for CPU-only workloads and most MPI-based applications

### device_only

Data resides exclusively on a specific GPU device. Host access requires explicit copies.

```cpp
dtl::distributed_vector<float, dtl::device_only<0>> vec(1000, ctx);
```

- Data allocated via `cudaMalloc` (or equivalent)
- GPU kernels access data directly -- no transfer latency
- Host access requires `cudaMemcpy` (explicit)
- Best for GPU-resident compute pipelines
- Compile-time device selection via template parameter

**Performance tip:** If your entire pipeline runs on GPU, `device_only` avoids all host-device transfer overhead. The data never touches host memory.

### unified_memory

Data is accessible from both host and device via CUDA Unified Memory (`cudaMallocManaged`).

```cpp
dtl::distributed_vector<float, dtl::unified_memory> vec(1000, ctx);
```

- Automatic page migration between host and device
- No explicit copies needed -- access from either side
- First access after migration incurs page fault latency
- Use `prefetch_to_device()` / `prefetch_to_host()` to reduce migration stalls

**Performance tip:** Prefetch data before you need it:

```cpp
auto local = vec.local_view();
// Prefetch to GPU before kernel launch
dtl::unified_memory::prefetch_to_device(local.data(), local.size_bytes(), 0);

// GPU kernel runs without page faults
launch_kernel(local.data(), local.size());
```

**When to use unified_memory:**
- Mixed host/device access patterns
- Prototyping GPU code before optimizing transfers
- When migration overhead is acceptable (large, infrequent transfers)

**When NOT to use:**
- Tight host-device ping-pong patterns (migration thrashing)
- When you know exactly which side needs data (use `device_only` or `host_only`)
- Performance-critical GPU kernels (explicit copies give more control)

### device_preferred

Data is preferentially placed on GPU with automatic host fallback if GPU memory is exhausted.

```cpp
dtl::distributed_vector<float, dtl::device_preferred> vec(1000, ctx);
```

- Attempts `cudaMalloc` first
- Falls back to host memory if allocation fails
- Useful for workloads that can gracefully degrade

### Placement Policy Comparison

| Policy | Location | Host Access | Device Access | Copies Needed | Best For |
|--------|----------|-------------|---------------|---------------|----------|
| `host_only` | CPU RAM | Direct | Explicit copy | To device | CPU workloads |
| `device_only<N>` | GPU N | Explicit copy | Direct | To host | GPU-resident pipelines |
| `unified_memory` | Migrated | Automatic | Automatic | None (implicit) | Mixed access |
| `device_preferred` | GPU (fallback CPU) | Depends | Depends | Depends | Flexible GPU |

---

## Avoiding remote_ref in Loops

The most common performance mistake in DTL is using `remote_ref` in a loop. Each `get()` or `put()` on a remote element triggers a separate communication operation.

### The Anti-Pattern

```cpp
// DISASTROUS PERFORMANCE: one communication per element
auto global = vec.global_view();
double sum = 0.0;
for (index_t i = 0; i < global.size(); ++i) {
    sum += global[i].get().value();  // N communications total!
}
```

For a vector of 1,000,000 elements across 4 ranks, this triggers ~750,000 remote `get()` calls -- each with network latency.

### The Correct Pattern

```cpp
// FAST: local computation + one collective
auto local = vec.local_view();
double local_sum = std::accumulate(local.begin(), local.end(), 0.0);
// One allreduce replaces 750,000 remote gets
double global_sum = dtl::reduce(vec, 0.0, std::plus<>{});
```

### When remote_ref Is Acceptable

`remote_ref` is designed for sparse, infrequent access where the communication is intentional:

```cpp
// OK: Occasional boundary element exchange
auto global = vec.global_view();
if (my_rank > 0) {
    // Get one boundary element from the previous rank
    auto boundary = global[my_local_start - 1].get().value();
    // Use it in a stencil computation
}
```

For bulk boundary exchange, use halo regions instead:

```cpp
// BETTER: Halo exchange for stencil patterns
auto halo = tensor.halo_view(1);
halo.exchange();  // One collective instead of per-element gets
```

---

## Batch vs Single-Element Operations on distributed_map

`distributed_map` uses hash-based key distribution. Single-element operations on remote keys require point-to-point communication. Batch operations amortize this cost.

### Single-Element (Slow for Remote Keys)

```cpp
dtl::distributed_map<std::string, int> map(ctx);

// Each insert/lookup on a remote key triggers communication
for (const auto& key : keys) {
    map.insert(key, compute_value(key));  // May communicate per insert
}
```

### Batch Operations (Preferred)

```cpp
// Batch insert: sorts keys by owner rank, sends in bulk
std::vector<std::pair<std::string, int>> batch;
for (const auto& key : keys) {
    batch.emplace_back(key, compute_value(key));
}
map.batch_insert(batch);  // One round of communication
```

### Local-Only Iteration

When iterating over a `distributed_map`, only local key-value pairs are visited. This is always communication-free:

```cpp
// Iterating local pairs -- no communication
for (const auto& [key, value] : map) {
    process(key, value);  // All pairs are local to this rank
}
```

---

## MPI Thread Level Selection

MPI thread support level affects both correctness and performance. DTL requests the thread level via `environment_options`.

| Level | Constant | Meaning | When to Use |
|-------|----------|---------|-------------|
| `MPI_THREAD_SINGLE` | 0 | Only one thread calls MPI | Single-threaded DTL, no `par{}` |
| `MPI_THREAD_FUNNELED` | 1 | Only the main thread calls MPI | `par{}` for local compute, collectives on main thread |
| `MPI_THREAD_SERIALIZED` | 2 | Any thread may call MPI, but not concurrently | Safe default for most DTL usage |
| `MPI_THREAD_MULTIPLE` | 3 | Any thread may call MPI concurrently | `async{}` operations with concurrent collectives |

### Recommendations

**For most applications:** Request `MPI_THREAD_SERIALIZED`. This allows using `par{}` for local computation while DTL serializes MPI calls internally.

```cpp
auto opts = dtl::environment_options::defaults();
// defaults() already requests MPI_THREAD_SERIALIZED
dtl::environment env(argc, argv, opts);
```

**For async-heavy workloads:** Request `MPI_THREAD_MULTIPLE` if you use `async{}` execution and the progress engine needs to make concurrent MPI calls.

```cpp
auto opts = dtl::environment_options::defaults();
opts.mpi_thread_level = 3;  // MPI_THREAD_MULTIPLE
dtl::environment env(argc, argv, opts);
```

**Performance note:** `MPI_THREAD_MULTIPLE` may reduce MPI performance on some implementations due to internal locking. Only use it if you genuinely need concurrent MPI calls from multiple threads.

---

## NCCL vs MPI for GPU Collectives

When both NCCL and MPI are available, the choice of collective backend significantly affects GPU communication performance.

### When to Use NCCL

- GPU-to-GPU collectives (allreduce, broadcast, allgather)
- Multi-GPU within a node (NVLink, NVSwitch)
- Large message sizes (> 1MB)
- Deep learning and dense linear algebra workloads

NCCL is optimized for GPU-resident data and can use NVLink/NVSwitch for intra-node transfers, bypassing PCIe entirely.

### When to Use MPI

- CPU-to-CPU communication
- Small message sizes (< 1KB), where NCCL launch overhead dominates
- Point-to-point operations (MPI has richer P2P support)
- Mixed CPU/GPU workloads where data is on host
- Heterogeneous clusters without NVLink

### Hybrid Strategy

For the best performance, use NCCL for GPU collectives and MPI for everything else:

```cpp
auto opts = dtl::environment_options::defaults();
// Enable both backends
dtl::environment env(argc, argv, opts);

// CPU operations use MPI automatically
auto cpu_ctx = env.make_world_context();

// GPU operations can leverage NCCL
auto gpu_ctx = env.make_world_context(/*device_id=*/0);
```

---

## Memory Allocation Patterns

### Pre-allocate Containers

Avoid repeated allocation and deallocation in loops:

```cpp
// BAD: Allocates and deallocates each iteration
for (int iter = 0; iter < 1000; ++iter) {
    dtl::distributed_vector<double> temp(n, ctx);
    compute(temp);
}  // temp deallocated here

// GOOD: Allocate once, reuse
dtl::distributed_vector<double> temp(n, ctx);
for (int iter = 0; iter < 1000; ++iter) {
    compute(temp);
}
```

### Reuse Views

Views are lightweight (pointer + size), but `segmented_view` has O(p) construction cost for the offset cache. Reuse views when possible:

```cpp
// OK for local_view (O(1) construction)
for (int iter = 0; iter < 1000; ++iter) {
    auto local = vec.local_view();  // Cheap
    process(local);
}

// Better for segmented_view (O(p) construction)
auto segv = vec.segmented_view();  // Build cache once
for (int iter = 0; iter < 1000; ++iter) {
    segv.for_each_local([](double& x) { x *= 2.0; });
}
// Note: view is invalid after structural operations (resize, redistribute)
```

### Minimize Structural Operations

`resize()` and `redistribute()` are collective operations that reallocate memory and invalidate all views:

```cpp
// BAD: Resize in a loop
for (int iter = 0; iter < n; ++iter) {
    vec.resize(vec.size() + 1);  // Collective + realloc each iteration
}

// GOOD: Compute final size, resize once
vec.resize(vec.size() + n);
```

---

## See Also

- [Views Guide](views.md) -- View types and semantics
- [Policies Guide](policies.md) -- Full policy reference
- [Environment Guide](environment.md) -- Backend lifecycle
