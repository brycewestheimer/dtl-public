# CPU Backend Guide

**Status:** Production-Ready
**Since:** DTL 0.1.0-alpha.1
**Last Updated:** 2026-02-07

## Overview

The CPU backend provides multi-threaded parallel execution for DTL operations using a thread pool. It is the default execution backend on systems without GPU accelerators and serves as the baseline for all DTL workloads.

Key capabilities:

- **Thread pool executor** with configurable thread count
- **Execution policies** (`seq`, `par`, `async`) for controlling parallelism
- **Task submission** via `std::future`-based async interface
- **Automatic thread count detection** via `std::thread::hardware_concurrency()`
- **No external dependencies** — uses standard C++ threading primitives

## Configuration

### Thread Pool Size

The `cpu_executor` manages a `thread_pool` that defaults to the number of hardware threads:

```cpp
#include <backends/cpu/cpu_executor.hpp>

// Default: uses hardware_concurrency() threads
dtl::cpu::thread_pool pool;

// Explicit: 8 worker threads
dtl::cpu::thread_pool pool(8);

// Query pool size
std::cout << "Workers: " << pool.size() << "\n";
```

### CMake Configuration

The CPU backend is always available — no special CMake flags are needed. However, you can control related settings:

```bash
cmake -DDTL_ENABLE_MPI=OFF \    # CPU-only, no MPI
      ..
```

## Execution Policies

DTL provides three execution policies that control how algorithms process local data. These are passed as the first argument to algorithm functions.

### `dtl::seq` — Sequential Execution

Single-threaded execution on the calling thread. No parallelism overhead.

```cpp
#include <dtl/policies/execution/seq.hpp>

dtl::distributed_vector<int> vec(10000, ctx);

// Process elements sequentially on one thread
dtl::for_each(dtl::seq{}, vec, [](int& x) { x *= 2; });

// Reduce sequentially
auto sum = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{});
```

**Best for:**
- Small data sizes where parallelism overhead exceeds benefit
- Debugging (deterministic execution order)
- Operations with side effects that require ordering

### `dtl::par` — Parallel Execution

Multi-threaded execution using the thread pool. Work is divided among worker threads.

```cpp
#include <dtl/policies/execution/par.hpp>

dtl::distributed_vector<double> vec(1000000, ctx);

// Process elements in parallel across multiple threads
dtl::for_each(dtl::par{}, vec, [](double& x) { x = std::sin(x); });

// Parallel reduce (local phase is multi-threaded)
auto sum = dtl::reduce(dtl::par{}, vec, 0.0, std::plus<>{});

// Parallel sort
dtl::sort(dtl::par{}, vec);
```

**Best for:**
- Large data sets that benefit from multi-core processing
- Compute-intensive per-element operations
- Production workloads on multi-core machines

### `dtl::async` — Asynchronous Execution

Returns a `distributed_future<T>` immediately, allowing the caller to continue while work proceeds in the background.

```cpp
#include <dtl/policies/execution/async.hpp>

dtl::distributed_vector<int> vec(100000, ctx);

// Launch sort asynchronously — returns immediately
auto future = dtl::sort(dtl::async{}, vec);

// ... do other work while sort runs ...

// Block until complete
future.get();
```

**Best for:**
- Overlapping computation with communication
- Pipeline-style processing
- Non-blocking algorithms in latency-sensitive applications

## Usage Patterns

### Basic CPU Processing

The most common pattern: create a container, get a local view, process elements:

```cpp
#include <dtl/dtl.hpp>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);

    // Default placement is host_only (CPU memory)
    dtl::distributed_vector<double> vec(10000, 4, 0);

    // Get local view — direct access to local partition
    auto local = vec.local_view();

    // Use STL algorithms directly on local view
    std::fill(local.begin(), local.end(), 1.0);
    std::sort(local.begin(), local.end());

    // Or use DTL algorithms with execution policies
    dtl::for_each(dtl::par{}, vec, [](double& x) { x *= 2.0; });
}
```

### Choosing the Right Execution Policy

| Scenario | Recommended Policy | Reason |
|----------|-------------------|--------|
| < 1,000 elements | `seq` | Parallelism overhead dominates |
| > 10,000 elements, simple ops | `par` | Good parallelism benefit |
| Compute-intensive per-element | `par` | Maximizes throughput |
| Needs deterministic order | `seq` | Single-threaded ordering |
| Overlap with communication | `async` | Non-blocking execution |
| Debugging | `seq` | Reproducible behavior |

### Thread Pool Patterns

Submit custom tasks to the thread pool:

```cpp
#include <backends/cpu/cpu_executor.hpp>

dtl::cpu::thread_pool pool;

// Submit work and get a future
auto result = pool.submit([]() {
    // Expensive computation
    return compute_answer();
});

// Get the result (blocks until ready)
auto answer = result.get();

// Wait for all pending tasks
pool.wait();

// Check pending task count
std::cout << "Pending: " << pool.pending() << "\n";
```

### Local View with STL

For purely local operations, you can bypass DTL algorithms entirely and use STL directly on the local view:

```cpp
auto local = vec.local_view();

// STL algorithms work directly on local_view iterators
auto it = std::find(local.begin(), local.end(), target);
auto count = std::count_if(local.begin(), local.end(), predicate);
std::transform(local.begin(), local.end(), local.begin(), transform_fn);
```

This is safe because `local_view` is purely local — no communication is involved, and `is_stl_safe_v<local_view<T>>` is `true`.

## The CPU Executor

### Architecture

The `cpu_executor` wraps a `thread_pool` and satisfies DTL's `Executor` and `ParallelExecutor` concepts:

```
cpu_executor
├── thread_pool (N worker threads)
│   ├── Worker 0
│   ├── Worker 1
│   ├── ...
│   └── Worker N-1
└── Task queue (thread-safe)
```

### Executor Interface

```cpp
#include <backends/cpu/cpu_executor.hpp>

dtl::cpu::cpu_executor exec;

// Execute a callable
exec.execute([]() {
    // Work to do
});

// Parallel for
exec.parallel_for(0, n, [&](size_t i) {
    data[i] = compute(i);
});

// Synchronize
exec.synchronize();
```

## Performance Tips

### Cache-Friendly Access

Access data sequentially to maximize CPU cache utilization:

```cpp
// Good: sequential access (cache-friendly)
auto local = vec.local_view();
for (size_t i = 0; i < local.size(); ++i) {
    local[i] = process(local[i]);
}

// Bad: random access (cache-unfriendly)
for (size_t i = 0; i < local.size(); ++i) {
    size_t j = random_index(i);
    local[j] = process(local[j]);
}
```

### NUMA Awareness

On multi-socket systems, memory allocation locality matters. DTL's CPU backend respects the system's default NUMA policy. For explicit NUMA control:

- Pin threads to specific cores using OS-level tools (`numactl`, `taskset`)
- Ensure first-touch initialization runs on the intended socket
- Use per-socket partition schemes for large distributed containers

```bash
# Run with NUMA binding
numactl --localalloc --cpunodebind=0 ./my_dtl_app
```

### Avoid False Sharing

When multiple threads write to adjacent memory locations, cache line contention can degrade performance:

```cpp
// Bad: threads updating adjacent counters
std::vector<int> counters(num_threads);  // False sharing!

// Better: pad to cache line size
struct alignas(64) PaddedCounter { int value = 0; };
std::vector<PaddedCounter> counters(num_threads);
```

### Right-Size the Thread Pool

More threads are not always better:

- For CPU-bound work: `num_threads ≈ hardware_concurrency()`
- For I/O-bound work: `num_threads > hardware_concurrency()` may help
- For MPI + threads: coordinate with MPI's thread support level
- Over-subscription degrades performance due to context switching

## See Also

- [MPI Backend Guide](mpi_guide.md) — Distributed communication
- [CUDA Backend Guide](cuda_guide.md) — GPU acceleration
- [Backend Comparison](comparison.md) — Feature comparison across backends
