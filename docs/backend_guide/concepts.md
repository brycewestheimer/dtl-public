# Backend Concepts

DTL uses C++20 concepts to define backend requirements. This ensures compile-time verification that backend implementations satisfy required interfaces.

## Overview

DTL's backend architecture has three primary concept hierarchies:

| Concept Family | Purpose | Key Implementations |
|----------------|---------|---------------------|
| **Communicator** | Point-to-point and collective communication | `mpi_comm_adapter` |
| **MemorySpace** | Memory allocation and properties | `host_memory_space`, `cuda_memory_space` |
| **Executor** | Computation dispatch | `cpu_executor`, `inline_executor` |

## Communicator Concepts

Communicators provide distributed communication capabilities across ranks.

### Communicator (Base)

The core concept for point-to-point communication:

```cpp
template <typename T>
concept Communicator = requires(T& comm, const T& ccomm,
                                void* buf, const void* cbuf,
                                size_type count, rank_t rank,
                                int tag, request_handle& req) {
    typename T::size_type;

    // Query operations
    { ccomm.rank() } -> std::same_as<rank_t>;
    { ccomm.size() } -> std::same_as<rank_t>;

    // Blocking point-to-point
    { comm.send(cbuf, count, rank, tag) } -> std::same_as<void>;
    { comm.recv(buf, count, rank, tag) } -> std::same_as<void>;

    // Non-blocking point-to-point
    { comm.isend(cbuf, count, rank, tag) } -> std::same_as<request_handle>;
    { comm.irecv(buf, count, rank, tag) } -> std::same_as<request_handle>;

    // Request completion
    { comm.wait(req) } -> std::same_as<void>;
    { comm.test(req) } -> std::same_as<bool>;
};
```

**Required Types:**
- `size_type`: Type for counts and sizes

**Required Operations:**
| Operation | Description |
|-----------|-------------|
| `rank()` | Returns this process's rank (0 to size-1) |
| `size()` | Returns total number of ranks |
| `send()` | Blocking send to destination rank |
| `recv()` | Blocking receive from source rank |
| `isend()` | Non-blocking send, returns request handle |
| `irecv()` | Non-blocking receive, returns request handle |
| `wait()` | Wait for non-blocking operation to complete |
| `test()` | Test if non-blocking operation completed |

### CollectiveCommunicator

Extends `Communicator` with collective operations:

```cpp
template <typename T>
concept CollectiveCommunicator = Communicator<T> &&
    requires(T& comm, void* buf, const void* cbuf,
             size_type count, rank_t root) {
    { comm.barrier() } -> std::same_as<void>;
    { comm.broadcast(buf, count, root) } -> std::same_as<void>;
    { comm.scatter(cbuf, buf, count, root) } -> std::same_as<void>;
    { comm.gather(cbuf, buf, count, root) } -> std::same_as<void>;
    { comm.allgather(cbuf, buf, count) } -> std::same_as<void>;
    { comm.alltoall(cbuf, buf, count) } -> std::same_as<void>;
};
```

**Additional Operations:**
| Operation | Description |
|-----------|-------------|
| `barrier()` | Synchronize all ranks |
| `broadcast()` | Root sends data to all ranks |
| `scatter()` | Root distributes data to all ranks |
| `gather()` | All ranks send data to root |
| `allgather()` | Gather data to all ranks |
| `alltoall()` | All-to-all exchange |

### ReducingCommunicator

Extends `CollectiveCommunicator` with reduction operations:

```cpp
template <typename T>
concept ReducingCommunicator = CollectiveCommunicator<T> &&
    requires(T& comm, const void* sendbuf, void* recvbuf,
             size_type count, rank_t root) {
    { comm.reduce_sum(sendbuf, recvbuf, count, root) } -> std::same_as<void>;
    { comm.allreduce_sum(sendbuf, recvbuf, count) } -> std::same_as<void>;
};
```

**Additional Operations:**
| Operation | Description |
|-----------|-------------|
| `reduce_sum()` | Sum reduction to root rank |
| `allreduce_sum()` | Sum reduction to all ranks |

### Communicator Tags

Tag types identify communicator implementations:

```cpp
struct mpi_communicator_tag {};
struct shared_memory_communicator_tag {};
struct gpu_communicator_tag {};
```

## MemorySpace Concepts

Memory spaces abstract memory allocation across different address spaces.

### MemorySpace (Base)

```cpp
template <typename T>
concept MemorySpace = requires(T& space, const T& cspace,
                               size_type size, size_type alignment,
                               void* ptr) {
    typename T::pointer;
    typename T::size_type;

    // Allocation
    { space.allocate(size) } -> std::same_as<void*>;
    { space.allocate(size, alignment) } -> std::same_as<void*>;
    { space.deallocate(ptr, size) } -> std::same_as<void>;

    // Properties
    { cspace.properties() } -> std::same_as<memory_space_properties>;
    { cspace.name() } -> std::convertible_to<const char*>;
};
```

**Required Types:**
- `pointer`: Raw pointer type (typically `void*`)
- `size_type`: Size type for counts

**Required Operations:**
| Operation | Description |
|-----------|-------------|
| `allocate(size)` | Allocate `size` bytes |
| `allocate(size, alignment)` | Allocate with alignment requirement |
| `deallocate(ptr, size)` | Free allocation |
| `properties()` | Returns `memory_space_properties` |
| `name()` | Returns human-readable name |

### Memory Space Properties

```cpp
struct memory_space_properties {
    bool host_accessible = true;    // CPU can access
    bool device_accessible = false; // GPU can access
    bool unified = false;           // Coherent across host/device
    bool supports_atomics = true;   // Atomic operations work
    bool pageable = true;           // Pageable (vs pinned)
    size_type alignment = alignof(std::max_align_t);
};
```

### TypedMemorySpace

Adds typed allocation support:

```cpp
template <typename Space, typename T>
concept TypedMemorySpace = MemorySpace<Space> &&
    requires(Space& space, size_type count, T* ptr) {
    { space.template allocate_typed<T>(count) } -> std::same_as<T*>;
    { space.deallocate_typed(ptr, count) } -> std::same_as<void>;
    { space.template construct<T>(ptr) } -> std::same_as<void>;
    { space.destroy(ptr) } -> std::same_as<void>;
};
```

### Memory Space Tags

```cpp
struct host_memory_space_tag {};    // CPU memory
struct device_memory_space_tag {};  // GPU memory
struct unified_memory_space_tag {}; // Managed memory
struct pinned_memory_space_tag {};  // Page-locked host memory
```

## Executor Concepts

Executors dispatch computation to processing resources.

### Executor (Base)

```cpp
template <typename T>
concept Executor = requires(T& exec, const T& cexec,
                            std::function<void()> f) {
    { exec.execute(f) } -> std::same_as<void>;
    { cexec.name() } -> std::convertible_to<const char*>;
};
```

**Required Operations:**
| Operation | Description |
|-----------|-------------|
| `execute(f)` | Execute callable `f` |
| `name()` | Returns executor name |

### ParallelExecutor

Adds parallel execution capabilities:

```cpp
template <typename T>
concept ParallelExecutor = Executor<T> &&
    requires(T& exec, const T& cexec,
             size_type count, std::function<void(size_type)> f) {
    { exec.parallel_for(count, f) } -> std::same_as<void>;
    { cexec.max_parallelism() } -> std::same_as<size_type>;
    { cexec.suggested_parallelism() } -> std::same_as<size_type>;
};
```

**Additional Operations:**
| Operation | Description |
|-----------|-------------|
| `parallel_for(count, f)` | Execute `f(i)` for i in [0, count) |
| `max_parallelism()` | Maximum concurrent work items |
| `suggested_parallelism()` | Recommended parallelism level |

### BulkExecutor

Optimized for bulk operations with chunking:

```cpp
template <typename T>
concept BulkExecutor = ParallelExecutor<T> &&
    requires(T& exec, size_type count,
             std::function<void(size_type, size_type)> f) {
    { exec.bulk_execute(count, f) } -> std::same_as<void>;
};
```

### Executor Properties

```cpp
struct executor_properties {
    size_type max_concurrency = 1;    // Max concurrent work items
    bool in_order = true;              // Ordered execution
    bool owns_threads = false;         // Manages thread pool
    bool supports_work_stealing = false;
};
```

### Executor Tags

```cpp
struct inline_executor_tag {};
struct thread_pool_executor_tag {};
struct single_thread_executor_tag {};
struct gpu_executor_tag {};
```

## Standard Executors

DTL provides these built-in executors:

### inline_executor

Executes work immediately in the calling thread:

```cpp
class inline_executor {
public:
    template <typename F>
    void execute(F&& f) { std::forward<F>(f)(); }

    static constexpr const char* name() noexcept { return "inline"; }
    static constexpr bool is_synchronous() noexcept { return true; }
};
```

### sequential_executor

Sequential execution with parallel interface:

```cpp
class sequential_executor {
public:
    template <typename F>
    void parallel_for(size_type count, F&& f) {
        for (size_type i = 0; i < count; ++i) { f(i); }
    }

    static constexpr size_type max_parallelism() noexcept { return 1; }
};
```

## Concept Verification

Implementations use `static_assert` to verify concept satisfaction:

```cpp
// In mpi_comm_adapter.hpp
static_assert(Communicator<mpi_comm_adapter>,
              "mpi_comm_adapter must satisfy Communicator concept");
static_assert(CollectiveCommunicator<mpi_comm_adapter>,
              "mpi_comm_adapter must satisfy CollectiveCommunicator concept");
static_assert(ReducingCommunicator<mpi_comm_adapter>,
              "mpi_comm_adapter must satisfy ReducingCommunicator concept");

// In cuda_memory_space.hpp
static_assert(MemorySpace<cuda_memory_space>,
              "cuda_memory_space must satisfy MemorySpace concept");

// In cpu_executor.hpp
static_assert(Executor<cpu_executor>,
              "cpu_executor must satisfy Executor concept");
static_assert(ParallelExecutor<cpu_executor>,
              "cpu_executor must satisfy ParallelExecutor concept");
```

## Concept Header Locations

| Concept | Header |
|---------|--------|
| Communicator | `include/dtl/backend/concepts/communicator.hpp` |
| MemorySpace | `include/dtl/backend/concepts/memory_space.hpp` |
| Executor | `include/dtl/backend/concepts/executor.hpp` |
| MemoryTransfer | `include/dtl/backend/concepts/memory_transfer.hpp` |
| Serializer | `include/dtl/backend/concepts/serializer.hpp` |

## See Also

- [Backend Selection](backend_selection.md) - When to use each backend
- [Implementing a Backend](implementing_backend.md) - Step-by-step guide
