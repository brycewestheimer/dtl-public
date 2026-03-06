# C++ API Quick Reference

**DTL Version:** 0.1.0-alpha.1
**Last Updated:** 2026-02-25

This document provides a quick reference for the most commonly used DTL C++ APIs. All symbols are in the `dtl::` namespace unless otherwise noted. DTL requires C++20.

---

## Containers

### `distributed_vector<T, Policies...>`

Distributed 1D sequence container (analog of `std::vector`).

**Header:** `#include <dtl/containers/distributed_vector.hpp>`

```cpp
// Construction
dtl::distributed_vector<int> vec(global_size, num_ranks, my_rank);
dtl::distributed_vector<float, dtl::device_only<0>> gpu_vec(1000, ctx);

// Size queries
vec.size();                 // Global size
vec.local_size();           // Elements on this rank
vec.rank();                 // This rank's ID
vec.num_ranks();            // Total number of ranks
vec.empty();                // True if global size is 0

// Views
auto local = vec.local_view();         // local_view<T> (no communication)
auto global = vec.global_view();       // global_view (may communicate)
auto seg = vec.segmented_view();       // segmented_view (per-rank segments)

// Data access
vec.local_data();           // Raw pointer to local data

// Collective operations
vec.resize(new_global_size);           // Collective resize
vec.barrier();                         // Synchronize all ranks
```

**Example:**
```cpp
dtl::distributed_vector<double> vec(10000, 4, 0);
auto local = vec.local_view();
for (auto& elem : local) {
    elem = 1.0;
}
vec.barrier();
```

---

### `distributed_array<T, N, Policies...>`

Fixed-size distributed array (analog of `std::array`).

**Header:** `#include <dtl/containers/distributed_array.hpp>`

```cpp
// Construction (N is compile-time global size)
dtl::distributed_array<int, 1000> arr(num_ranks, my_rank);

// Compile-time extent
static_assert(decltype(arr)::extent == 1000);

// Same view interface as distributed_vector
auto local = arr.local_view();
auto global = arr.global_view();
```

**Key difference from `distributed_vector`:** No `resize()`, `clear()`, or `push_back()` — size is fixed at compile time.

---

### `distributed_tensor<T, Rank, Layout, Policies...>`

N-dimensional distributed tensor.

**Header:** `#include <dtl/containers/distributed_tensor.hpp>`

```cpp
// Construction with extents
dtl::nd_extent<3> shape = {100, 64, 64};
dtl::distributed_tensor<float, 3> tensor(shape, num_ranks, my_rank);

// Layout policies
dtl::distributed_tensor<float, 2, dtl::row_major> row_tensor(shape, ...);
dtl::distributed_tensor<float, 2, dtl::column_major> col_tensor(shape, ...);

// Size queries
tensor.extents();           // nd_extent<Rank> of global shape
tensor.total_size();        // Total global elements
tensor.local_size();        // Local elements on this rank

// Multi-dimensional access
auto local = tensor.local_view();
```

---

### `distributed_map<K, V, Hash, KeyEqual, Policies...>`

Distributed hash map (analog of `std::unordered_map`).

**Header:** `#include <dtl/containers/distributed_map.hpp>`

```cpp
// Construction
dtl::distributed_map<std::string, int> map(ctx);

// Insert and access
map.insert("key1", 42);
auto ref = map["key1"];     // Returns remote_ref<int>
if (ref.is_local()) {
    int val = ref.get().value();
}

// Local iteration
for (auto& [key, value] : map.local_view()) {
    process(key, value);
}

// Queries
map.local_size();           // Local entry count
map.is_local("key1");       // True if key is on this rank
map.owner("key1");          // Rank that owns this key
```

---

### `distributed_span<T, Extent>`

Non-owning view over distributed memory (analog of `std::span`).

**Header:** `#include <dtl/containers/distributed_span.hpp>`

```cpp
// From a container
dtl::distributed_span<int> span(vec);

// From raw data
dtl::distributed_span<int> span(ptr, local_size, global_size);

// Size queries
span.size();                // Global size
span.local_size();          // Local size
span.data();                // Pointer to local data
span.rank();                // Current rank metadata
span.num_ranks();           // Total rank metadata
```

---

## Algorithms

All algorithms are in `#include <dtl/algorithms/algorithms.hpp>` or individual headers.

### `for_each`

Apply a function to each local element.

**Header:** `#include <dtl/algorithms/non_modifying/for_each.hpp>`

```cpp
dtl::for_each(dtl::par{}, vec, [](int& x) { x *= 2; });
dtl::for_each(vec, [](int& x) { x *= 2; });  // Default: seq
```

---

### `transform`

Apply a transformation from source to destination.

**Header:** `#include <dtl/algorithms/modifying/transform.hpp>`

```cpp
// Unary: src → dst
dtl::transform(dtl::par{}, src, dst, [](int x) { return x * 2; });

// In-place
dtl::transform(dtl::par{}, vec, [](int x) { return x * 2; });
```

---

### `reduce`

Combine all elements using a binary operation. Supports local-only and distributed (with communicator) variants.

**Header:** `#include <dtl/algorithms/reductions/reduce.hpp>`

```cpp
// Local reduce (no communication)
auto result = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{});
int sum = result.value();

// Distributed reduce with communicator
auto result = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{}, comm);
int global_sum = result.global_value;
```

Returns `reduce_result<T>` with `.local_value`, `.global_value`, and `.value()`.

---

### `sort` / `stable_sort_global`

Sort elements locally or globally.

**Header:** `#include <dtl/algorithms/sorting/sort.hpp>`

```cpp
// Local sort (single-rank or standalone)
dtl::sort(dtl::par{}, vec);
dtl::sort(dtl::par{}, vec, std::greater<>{});

// Distributed sort with communicator (sample sort)
dtl::sort(dtl::par{}, vec, std::less<>{}, comm);
```

---

### `inclusive_scan` / `exclusive_scan`

Prefix scan operations.

**Header:** `#include <dtl/algorithms/reductions/scan.hpp>`

```cpp
dtl::inclusive_scan(dtl::par{}, input, output, 0, std::plus<>{});
dtl::exclusive_scan(dtl::par{}, input, output, 0, std::plus<>{});
```

---

### `find` / `count`

Search and counting algorithms.

**Header:** `#include <dtl/algorithms/non_modifying/find.hpp>`, `#include <dtl/algorithms/non_modifying/count.hpp>`

```cpp
auto result = dtl::find(vec, target_value);
auto n = dtl::count(vec, target_value);
auto n = dtl::count_if(vec, [](int x) { return x > 0; });
```

---

### `copy` / `fill`

Data initialization and copying.

**Header:** `#include <dtl/algorithms/modifying/copy.hpp>`, `#include <dtl/algorithms/modifying/fill.hpp>`

```cpp
dtl::copy(dtl::par{}, src, dst);
dtl::fill(dtl::par{}, vec, 42);
```

---

## Views

### `local_view<T>`

Non-communicating view of local data. Safe for STL algorithms.

**Header:** `#include <dtl/views/local_view.hpp>`

```cpp
auto local = vec.local_view();
for (auto& elem : local) { /* ... */ }
local.begin(); local.end(); local.size();
local[i];  // Direct access by local index

static_assert(dtl::is_stl_safe_v<decltype(local)>);
```

---

### `global_view<Container>`

View providing access to all elements across ranks. Returns `remote_ref<T>` for remote elements.

**Header:** `#include <dtl/views/global_view.hpp>`

```cpp
auto global = vec.global_view();
auto ref = global[500];     // Returns remote_ref<T>
if (ref.is_local()) {
    int val = ref.get().value();
}

static_assert(dtl::may_communicate_v<decltype(global)>);
```

---

### `segmented_view<Container>`

Iteration over per-rank segments for efficient distributed algorithms.

**Header:** `#include <dtl/views/segmented_view.hpp>`

```cpp
auto seg = vec.segmented_view();
// Iterate segments (one per rank)
```

---

### `strided_view<Range>`

View with stride between elements.

**Header:** `#include <dtl/views/strided_view.hpp>`

---

### `remote_ref<T>`

Proxy reference to a potentially remote element.

**Header:** `#include <dtl/views/remote_ref.hpp>`

```cpp
auto ref = global_view[i];
ref.is_local();             // True if element is on this rank
ref.owner();                // Rank that owns the element
ref.get();                  // result<T> — may communicate
ref.put(value);             // Write to remote element
```

---

## Policies

**Header:** `#include <dtl/policies/policies.hpp>`

### Execution Policies

| Policy | Header | Description |
|--------|--------|-------------|
| `dtl::seq` | `execution/seq.hpp` | Sequential (single-threaded) |
| `dtl::par` | `execution/par.hpp` | Parallel (multi-threaded) |
| `dtl::async` | `execution/async.hpp` | Asynchronous (returns future) |
| `dtl::on_stream` | `execution/on_stream.hpp` | GPU stream execution |

```cpp
dtl::for_each(dtl::seq{}, vec, f);    // Sequential
dtl::for_each(dtl::par{}, vec, f);    // Parallel
auto fut = dtl::sort(dtl::async{}, vec);  // Async
```

### Partition Policies

| Policy | Header | Description |
|--------|--------|-------------|
| `dtl::block_partition<>` | `partition/block_partition.hpp` | Contiguous blocks (default) |
| `dtl::cyclic_partition` | `partition/cyclic_partition.hpp` | Round-robin distribution |
| `dtl::hash_partition` | `partition/hash_partition.hpp` | Hash-based distribution |
| `dtl::replicated` | `partition/replicated.hpp` | Full replication on all ranks |
| `dtl::custom_partition` | `partition/custom_partition.hpp` | User-defined partitioning |
| `dtl::dynamic_block` | `partition/dynamic_block.hpp` | Dynamic load balancing |

```cpp
dtl::distributed_vector<int, dtl::cyclic_partition> vec(1000, ctx);
```

### Placement Policies

| Policy | Header | Description |
|--------|--------|-------------|
| `dtl::host_only` | `placement/host_only.hpp` | CPU memory (default) |
| `dtl::device_only<N>` | `placement/device_only.hpp` | GPU N device memory |
| `dtl::device_only_runtime` | `placement/device_only_runtime.hpp` | Runtime GPU selection |
| `dtl::unified_memory` | `placement/unified_memory.hpp` | CUDA/HIP managed memory |
| `dtl::device_preferred` | `placement/device_preferred.hpp` | GPU-preferred unified memory |
| `dtl::explicit_placement` | `placement/explicit_placement.hpp` | Manual placement control |

```cpp
dtl::distributed_vector<float, dtl::device_only<0>> gpu_vec(1000, ctx);
dtl::distributed_vector<float, dtl::unified_memory> unified_vec(1000, ctx);
```

---

## Communication

**Header:** `#include <dtl/communication/communication.hpp>`

### Point-to-Point

```cpp
dtl::send(comm, data, count, dest, tag);
dtl::recv(comm, data, count, source, tag);
dtl::ssend(comm, data, count, dest, tag);      // Synchronous send
dtl::sendrecv(comm, ...);                       // Combined send/recv

// Non-blocking
auto req = dtl::isend(comm, data, count, dest, tag);
auto req = dtl::irecv(comm, data, count, source, tag);
dtl::wait(comm, req);
dtl::test(comm, req);
dtl::wait_all(comm, requests);
```

### Collectives

```cpp
dtl::barrier(comm);
dtl::broadcast(comm, data, count, root);
dtl::scatter(comm, send, recv, root);
dtl::gather(comm, send, recv, root);
dtl::allgather(comm, send, recv);
dtl::alltoall(comm, send, recv);
```

### Reductions

```cpp
dtl::reduce(comm, send, recv, op, root);
dtl::allreduce(comm, send, recv, op);
dtl::allreduce_inplace(comm, data, op);
dtl::scan(comm, send, recv, op);               // Inclusive prefix scan
dtl::exscan(comm, send, recv, op);             // Exclusive prefix scan

// Convenience
dtl::allsum(comm, send, recv);
dtl::allmax(comm, send, recv);
dtl::allmin(comm, send, recv);
```

### Reduction Operations

| Operation | Type | Identity |
|-----------|------|----------|
| `reduce_sum<T>` | Sum | 0 |
| `reduce_product<T>` | Product | 1 |
| `reduce_min<T>` | Minimum | max |
| `reduce_max<T>` | Maximum | lowest |
| `reduce_land<T>` | Logical AND | true |
| `reduce_lor<T>` | Logical OR | false |
| `reduce_band<T>` | Bitwise AND | ~0 |
| `reduce_bor<T>` | Bitwise OR | 0 |
| `reduce_bxor<T>` | Bitwise XOR | 0 |

---

## Context and Environment

### `environment`

RAII reference-counted environment manager. Initializes/finalizes backends.

**Header:** `#include <dtl/core/environment.hpp>`

```cpp
// With argc/argv (preferred for MPI)
dtl::environment env(argc, argv);

// Without argc/argv
dtl::environment env;

// With options
auto opts = dtl::environment_options::defaults();
opts.mpi.thread_level = dtl::thread_support_level::multiple;
dtl::environment env(argc, argv, opts);
```

### `world_comm()`

Get the default communicator.

**Header:** `#include <dtl/communication/default_communicator.hpp>`

```cpp
auto comm = dtl::world_comm();
// Returns mpi::mpi_comm_adapter (MPI enabled) or null_communicator
```

---

## Error Handling

### `result<T>`

Monadic result type (similar to `std::expected<T, status>`).

**Header:** `#include <dtl/error/result.hpp>`

```cpp
dtl::result<int> r = compute();

r.has_value();              // True if success
r.has_error();              // True if error
r.value();                  // Get value (throws if error)
r.error();                  // Get status (throws if success)

// Monadic operations
r.and_then([](int v) -> result<int> { return v * 2; });
r.or_else([](status s) -> result<int> { return 0; });
r.transform([](int v) { return v * 2; });

// Boolean context
if (r) { /* success */ }
```

### `status`

Error status with code and message.

**Header:** `#include <dtl/error/status.hpp>`

```cpp
dtl::status s(dtl::status_code::ok);
s.is_ok();                  // True if success
s.is_error();               // True if error
s.code();                   // status_code enum
s.message();                // Error message string
```

### Factory Functions

```cpp
auto ok = dtl::make_ok_result();                    // result<void> success
auto err = dtl::make_error_result<int>(status_code::out_of_bounds);
auto err = dtl::make_error<int>(status_code::cuda_error, rank, "msg");
```

---

## Umbrella Headers

| Header | Contents |
|--------|----------|
| `<dtl/dtl.hpp>` | Everything (all modules) |
| `<dtl/algorithms/algorithms.hpp>` | All algorithms |
| `<dtl/containers/containers.hpp>` | All containers |
| `<dtl/views/views.hpp>` | All views |
| `<dtl/policies/policies.hpp>` | All policies |
| `<dtl/communication/communication.hpp>` | All communication ops |
| `<dtl/error/error.hpp>` | All error handling |
| `<dtl/memory/memory.hpp>` | All memory spaces |

---

## Serialization

**Header:** `#include <dtl/serialization/serialization.hpp>`

DTL automatically serializes trivially-copyable types via `memcpy`. For other types, provide a `serializer<T>` specialization or use the `DTL_SERIALIZABLE` macro.

### Built-in Support

| Type | Header | Notes |
|------|--------|-------|
| Trivial types (`int`, `double`, PODs) | `serializer.hpp` | `memcpy`-based, zero overhead |
| `std::array<T,N>` | `serializer.hpp` | Trivial elements only |
| `std::pair<T1,T2>` | `serializer.hpp` | Trivial elements only |
| `std::string` | `stl_serializers.hpp` | Length-prefixed |
| `std::vector<T>` | `stl_serializers.hpp` | Count-prefixed, per-element |
| `std::optional<T>` | `stl_serializers.hpp` | Flag byte + value |

### `DTL_SERIALIZABLE` Macro

**Header:** `#include <dtl/serialization/aggregate_serializer.hpp>`

```cpp
struct my_message {
    int id;
    std::string name;
    std::vector<double> data;
};
DTL_SERIALIZABLE(my_message, id, name, data)
// Generates dtl::serializer<my_message> using field-by-field serialization
```

### Custom Serializer (Manual)

```cpp
template <>
struct dtl::serializer<MyType> {
    static dtl::size_type serialized_size(const MyType& v);
    static dtl::size_type serialize(const MyType& v, std::byte* buf);
    static MyType deserialize(const std::byte* buf, dtl::size_type size);
};
```

### Helper Functions

```cpp
dtl::serialize_field(value, buffer);        // Serialize one field
dtl::deserialize_field<T>(buffer, size);    // Deserialize one field
dtl::field_serialized_size(value);          // Size of one field
```

---

## Executor and `parallel_for`

**Header:** `#include <backends/cpu/cpu_executor.hpp>`

### `cpu_executor`

```cpp
dtl::cpu::cpu_executor exec;                // Hardware concurrency threads
dtl::cpu::cpu_executor exec(8);             // 8 threads

// Execute tasks
exec.execute(func);                         // Fire-and-forget
auto fut = exec.async_execute(func);        // Returns std::future
exec.sync_execute(func);                    // Blocks until done

// Parallel loops
exec.parallel_for(0, N, [](dtl::index_t i) { /* ... */ });
exec.parallel_for(count, [](dtl::size_type i) { /* ... */ });
exec.parallel_for_chunked(0, N, chunk, func);

// Parallel reduce
T result = exec.parallel_reduce<T>(0, N, identity, map_fn, reduce_fn);

// Queries
exec.num_threads();
exec.max_parallelism();
exec.synchronize();                         // Wait for all tasks
```

### Free Functions (Default Executor)

```cpp
dtl::cpu::parallel_for(0, N, func);
dtl::cpu::parallel_reduce<T>(0, N, identity, map_fn, reduce_fn);
```

---

## See Also

- [C API Reference](c_api_reference.md)
- [Python API Reference](python_api_reference.md)
- [Backend Comparison](../backends/comparison.md)
