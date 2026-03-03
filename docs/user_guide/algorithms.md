# Legacy Deep-Dive: Algorithms

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [07-algorithms-collectives-and-remote-operations.md](07-algorithms-collectives-and-remote-operations.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


DTL provides distributed algorithms that operate efficiently across partitioned data. This guide covers the core algorithms and their usage patterns.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm Categories](#algorithm-categories)
- [for_each](#for_each)
- [transform](#transform)
- [reduce](#reduce)
- [scan](#scan)
- [find and count](#find-and-count)
- [sort](#sort)
- [Execution Policies](#execution-policies)
- [Best Practices](#best-practices)

---

## Overview

DTL algorithms follow the STL model but are designed for distributed execution:
- Operate on distributed containers or views
- Use segmented iteration for performance
- Support execution policy overrides
- Make communication explicit

### Key Principles

1. **Local operations are fast** - Algorithms operate on local partitions without communication
2. **Global operations are explicit** - Collective communication is clearly documented
3. **Segmented iteration** - The performance path for all distributed algorithms
4. **No hidden remote access** - Algorithms never perform per-element remote gets in hot paths

---

## Algorithm Categories

DTL classifies algorithms into three domains:

| Domain | Communication | Example |
|--------|---------------|---------|
| **Local** | Never | `local_reduce`, `local_sort` |
| **Collective** | All ranks participate | `distributed_reduce`, `broadcast` |
| **Distributed** | Point-to-point or mixed | `distributed_sort` |

### Local Algorithms

Operate only on local partition; no communication:

```cpp
// Local reduce - no MPI calls
int sum = dtl::local_reduce(vec, 0, std::plus<>{});

// Equivalent to manual local iteration
auto local = vec.local_view();
int sum = std::accumulate(local.begin(), local.end(), 0);
```

### Collective Algorithms

Require all ranks to participate:

```cpp
// All ranks must call this
double global_sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});

// If any rank doesn't participate: undefined behavior or hang
```

### Distributed Algorithms

May combine local and collective operations:

```cpp
// Distributed sort: local sort + sample sort + redistribution
dtl::distributed_sort(vec);
```

---

## for_each

Applies a function to each element in the container.

### Basic Usage

```cpp
dtl::distributed_vector<int> vec(1000, size, rank);

// Apply function to all local elements
dtl::for_each(vec, [](int& x) { x *= 2; });
```

### With Execution Policy

```cpp
// Parallel execution
dtl::for_each(dtl::par, vec, [](int& x) { x *= 2; });

// Sequential execution
dtl::for_each(dtl::seq, vec, [](int& x) { x *= 2; });
```

### Signature

```cpp
template<typename Container, typename UnaryFunc>
void for_each(Container& c, UnaryFunc f);

template<typename ExecutionPolicy, typename Container, typename UnaryFunc>
void for_each(ExecutionPolicy&& policy, Container& c, UnaryFunc f);
```

### Semantics

- Applies `f` to each element in the local partition
- Does NOT communicate
- Order of application depends on execution policy
- Function must be safe for concurrent execution with `par` policy

### Example: Initialize Based on Global Index

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);

dtl::for_each(vec, [&vec](double& x, dtl::size_type local_idx) {
    dtl::size_type global_idx = vec.global_offset() + local_idx;
    x = std::sin(static_cast<double>(global_idx) * 0.01);
});
```

---

## transform

Applies a transformation and writes results to an output container.

### Basic Usage

```cpp
dtl::distributed_vector<int> input(1000, size, rank);
dtl::distributed_vector<int> output(1000, size, rank);

// Transform: square each element
dtl::transform(input, output, [](int x) { return x * x; });
```

### In-Place Transform

```cpp
// Same input and output
dtl::transform(vec, vec, [](int x) { return x * 2; });
```

### Binary Transform

```cpp
dtl::distributed_vector<double> a(1000, size, rank);
dtl::distributed_vector<double> b(1000, size, rank);
dtl::distributed_vector<double> result(1000, size, rank);

// Element-wise addition
dtl::transform(a, b, result, [](double x, double y) { return x + y; });
```

### Signature

```cpp
// Unary transform
template<typename InputContainer, typename OutputContainer, typename UnaryFunc>
void transform(const InputContainer& input, OutputContainer& output, UnaryFunc f);

// Binary transform
template<typename Input1, typename Input2, typename Output, typename BinaryFunc>
void transform(const Input1& in1, const Input2& in2, Output& out, BinaryFunc f);
```

### Requirements

- Input and output must have same global size
- Input and output must have compatible partitioning
- Local sizes must match (same partition policy)

---

## reduce

Reduces elements to a single value using a binary operation.

### local_reduce

Reduces only local elements; no communication:

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);

// Sum of local elements only
double local_sum = dtl::local_reduce(vec, 0.0, std::plus<>{});
```

### distributed_reduce

Reduces all elements across all ranks (collective):

```cpp
// Global sum across all ranks
double global_sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});

// All ranks receive the same result
```

### reduce_to_root

Reduces to a single rank:

```cpp
// Only rank 0 gets the result
double sum = dtl::reduce_to_root(vec, 0.0, std::plus<>{}, /*root=*/0);

if (rank == 0) {
    std::cout << "Total: " << sum << "\n";
}
```

### Signature

```cpp
// Local reduce
template<typename Container, typename T, typename BinaryOp>
T local_reduce(const Container& c, T init, BinaryOp op);

// Distributed reduce (allreduce)
template<typename Container, typename T, typename BinaryOp>
T distributed_reduce(const Container& c, T init, BinaryOp op);

// Reduce to root
template<typename Container, typename T, typename BinaryOp>
T reduce_to_root(const Container& c, T init, BinaryOp op, rank_t root);
```

### Requirements

- Binary operation should be associative (order may vary)
- For deterministic results, operation should be commutative
- Type `T` must be transportable for distributed reduce

### Implementation Pattern

Distributed reduce uses segmented iteration internally:

```cpp
// Conceptual implementation
template<typename Container, typename T, typename BinaryOp>
T distributed_reduce(const Container& c, T init, BinaryOp op) {
    auto segv = c.segmented_view();

    // Phase 1: Local reduction (no communication)
    T local_result = init;
    for (auto& segment : segv.segments()) {
        for (const auto& x : segment.local_range()) {
            local_result = op(local_result, x);
        }
    }

    // Phase 2: Global reduction (collective)
    T global_result;
    MPI_Allreduce(&local_result, &global_result, ...);

    return global_result;
}
```

---

## scan

DTL provides distributed prefix sum (scan) operations that compute cumulative results across all ranks.

### inclusive_scan

Computes prefix sums where each element includes the current value:

```cpp
dtl::distributed_vector<int> vec(1000, size, rank);
// Fill with values...

// Distributed inclusive scan (collective)
dtl::inclusive_scan(vec, std::plus<>{});
// vec[i] now contains sum of elements 0..i across all ranks
```

### exclusive_scan

Computes prefix sums where each element excludes the current value:

```cpp
// Distributed exclusive scan (collective)
dtl::exclusive_scan(vec, 0, std::plus<>{});
// vec[i] now contains sum of elements 0..(i-1) across all ranks
// vec[0] = initial value (0)
```

### Signature

```cpp
// Inclusive scan
template<typename Container, typename BinaryOp>
void inclusive_scan(Container& c, BinaryOp op);

// Exclusive scan
template<typename Container, typename T, typename BinaryOp>
void exclusive_scan(Container& c, T init, BinaryOp op);
```

### Implementation

Distributed scan uses a two-phase algorithm:
1. **Local scan**: Each rank computes local prefix sums
2. **Cross-rank prefix**: Ranks exchange partial sums to compute global offsets
3. **Adjustment**: Local values are adjusted by the global prefix

This is a collective operation - all ranks must participate.

---

## find and count

DTL provides search and counting algorithms for distributed containers.

### find / find_if

Find the first element matching a value or predicate:

```cpp
dtl::distributed_vector<int> vec(1000, size, rank);

// Find first element equal to value (searches local partition)
auto it = dtl::find(vec, 42);
if (it != vec.local_view().end()) {
    std::cout << "Found 42 at local index " << (it - vec.local_view().begin()) << "\n";
}

// Find first element matching predicate
auto it2 = dtl::find_if(vec, [](int x) { return x > 100; });
```

### count / count_if

Count elements matching a value or predicate:

```cpp
// Count elements equal to value (local count)
auto n = dtl::count(vec, 42);

// Count elements matching predicate
auto n2 = dtl::count_if(vec, [](int x) { return x % 2 == 0; });

// For global count, use distributed_reduce
std::size_t local_count = dtl::count(vec, 42);
std::size_t global_count;
MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
```

### Signature

```cpp
// Find
template<typename Container, typename T>
auto find(const Container& c, const T& value);

template<typename Container, typename Predicate>
auto find_if(const Container& c, Predicate pred);

// Count
template<typename Container, typename T>
size_type count(const Container& c, const T& value);

template<typename Container, typename Predicate>
size_type count_if(const Container& c, Predicate pred);
```

### Semantics

- **find/find_if**: Search local partition only; returns local iterator
- **count/count_if**: Count in local partition only; for global count, use allreduce
- These algorithms do NOT communicate automatically

---

## sort

Sorts distributed data while maintaining global order.

### local_sort

Sorts only the local partition:

```cpp
dtl::distributed_vector<int> vec(1000, size, rank);

// Sort local partition only
// Global order NOT maintained across ranks
dtl::local_sort(vec);
```

### distributed_sort

Sorts globally across all ranks (collective):

```cpp
// After this, global order is maintained:
// vec[0] on any rank <= vec[1] <= ... <= vec[n-1]
dtl::distributed_sort(vec);

// With custom comparator
dtl::distributed_sort(vec, std::greater<>{});  // Descending
```

### Signature

```cpp
// Local sort
template<typename Container>
void local_sort(Container& c);

template<typename Container, typename Compare>
void local_sort(Container& c, Compare comp);

// Distributed sort
template<typename Container>
void distributed_sort(Container& c);

template<typename Container, typename Compare>
void distributed_sort(Container& c, Compare comp);
```

### Implementation

Distributed sort typically uses:
1. Local sort of each partition
2. Sample sort for pivot selection
3. Data redistribution based on pivots
4. Final local sort

This is a collective operation with significant communication.

---

## Execution Policies

All DTL algorithms support execution policy overrides.

### Available Policies

| Policy | Behavior |
|--------|----------|
| `dtl::seq` | Sequential, single-threaded |
| `dtl::par` | Parallel, multi-threaded |
| `dtl::par_unseq` | Parallel and vectorized |
| `dtl::async` | Non-blocking, returns future |

### Usage

```cpp
// Sequential (default)
dtl::for_each(vec, func);
dtl::for_each(dtl::seq, vec, func);

// Parallel
dtl::for_each(dtl::par, vec, func);

// Parallel + vectorized
dtl::for_each(dtl::par_unseq, vec, func);

// Async
auto future = dtl::for_each(dtl::async, vec, func);
// ... do other work ...
future.wait();
```

### Policy Requirements

| Policy | Function Requirements |
|--------|----------------------|
| `seq` | None |
| `par` | Thread-safe (no data races) |
| `par_unseq` | Thread-safe + vectorization-safe |
| `async` | Thread-safe |

### Async Pattern

```cpp
// Start async operations
auto f1 = dtl::transform(dtl::async, vec1, out1, func1);
auto f2 = dtl::transform(dtl::async, vec2, out2, func2);

// Operations run concurrently

// Wait for all
f1.wait();
f2.wait();

// Or wait for any
// dtl::when_any(f1, f2);  // Future feature
```

---

## Best Practices

### 1. Use Local Algorithms When Possible

```cpp
// GOOD: Local reduce when only local result needed
double local_sum = dtl::local_reduce(vec, 0.0, std::plus<>{});

// Only use distributed_reduce when global result needed
double global_sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
```

### 2. Prefer Segmented Operations

DTL algorithms use segmented iteration internally. When writing custom algorithms, follow the same pattern:

```cpp
// Custom distributed algorithm
template<typename Container>
double custom_operation(Container& c) {
    auto segv = c.segmented_view();

    double local_result = 0.0;
    for (auto& segment : segv.segments()) {
        for (double& x : segment.local_range()) {
            local_result += custom_compute(x);
        }
    }

    // Single collective operation
    double global_result;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_result;
}
```

### 3. Match Execution Policies to Workload

```cpp
// Light work per element: use par_unseq for vectorization
dtl::for_each(dtl::par_unseq, vec, [](double& x) { x *= 2.0; });

// Heavy work per element: use par for threading
dtl::for_each(dtl::par, vec, [](double& x) {
    x = expensive_computation(x);
});

// Debugging: use seq
dtl::for_each(dtl::seq, vec, [](double& x) {
    std::cout << x << "\n";  // Not thread-safe
});
```

### 4. Be Aware of Collective Requirements

```cpp
// All ranks MUST call collective algorithms
if (rank == 0) {
    // BAD: Only rank 0 calls reduce - will deadlock!
    // auto sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
}

// GOOD: All ranks call
auto sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
if (rank == 0) {
    std::cout << "Sum: " << sum << "\n";
}
```

### 5. Use Transform-Reduce for Efficiency

```cpp
// Compute sum of squares
// Option 1: Transform then reduce (two passes)
dtl::transform(vec, temp, [](double x) { return x * x; });
double sum_sq = dtl::distributed_reduce(temp, 0.0, std::plus<>{});

// Option 2: Single pass with transform_reduce (more efficient)
double sum_sq = dtl::transform_reduce(
    vec,
    0.0,
    std::plus<>{},
    [](double x) { return x * x; }
);
```

---

## Algorithm Reference

| Algorithm | Category | Communication |
|-----------|----------|---------------|
| `for_each` | Local | None |
| `transform` | Local | None |
| `copy` | Local | None |
| `fill` | Local | None |
| `find` / `find_if` | Local | None |
| `count` / `count_if` | Local | None |
| `minmax` | Local | None |
| `local_reduce` | Local | None |
| `distributed_reduce` | Collective | Allreduce |
| `reduce_to_root` | Collective | Reduce |
| `transform_reduce` | Collective | Allreduce |
| `inclusive_scan` | Collective | Prefix exchange |
| `exclusive_scan` | Collective | Prefix exchange |
| `local_sort` | Local | None |
| `distributed_sort` | Distributed | All-to-all |
| `broadcast` | Collective | Broadcast |
| `gather` | Collective | Gather |
| `scatter` | Collective | Scatter |

---

## See Also

- [Containers Guide](containers.md) - Container types and operations
- [Views Guide](views.md) - Understanding segmented_view
- [Policies Guide](policies.md) - Execution policies in detail
