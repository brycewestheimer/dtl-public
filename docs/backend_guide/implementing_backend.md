# Implementing a Backend

This guide walks through implementing a new DTL backend, using examples from existing implementations.

## Backend Architecture Overview

A complete DTL backend typically includes:

1. **Communicator** - For distributed communication
2. **Memory Space** - For memory allocation
3. **Executor** - For computation dispatch
4. **Memory Transfer** - For cross-space copies (optional)

Not all backends need all components. For example:
- A pure communication backend (like SHMEM) only needs a Communicator
- A pure acceleration backend (like OpenCL) only needs Memory Space and Executor

## Directory Structure

Place backend implementations in `backends/<name>/`:

```
backends/
└── my_backend/
    ├── my_communicator.hpp
    ├── my_memory_space.hpp
    ├── my_executor.hpp
    └── my_memory_transfer.hpp
```

## Step 1: Implement a Memory Space

Memory spaces manage allocation in a specific address space.

### Required Interface

```cpp
// backends/my_backend/my_memory_space.hpp

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/memory_space.hpp>

namespace dtl {
namespace my_backend {

class my_memory_space {
public:
    // Required type aliases
    using pointer = void*;
    using size_type = dtl::size_type;

    // Constructor
    my_memory_space() = default;

    // Copy/move semantics (choose appropriate semantics)
    my_memory_space(const my_memory_space&) = delete;
    my_memory_space& operator=(const my_memory_space&) = delete;
    my_memory_space(my_memory_space&&) = default;
    my_memory_space& operator=(my_memory_space&&) = default;

    // Required: Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "my_backend";
    }

    // Required: Get memory space properties
    [[nodiscard]] memory_space_properties properties() const noexcept {
        return memory_space_properties{
            .host_accessible = true,   // Can CPU access this memory?
            .device_accessible = false, // Can GPU access this memory?
            .unified = false,           // Is it coherent?
            .supports_atomics = true,
            .pageable = true,
            .alignment = 64             // Alignment guarantee
        };
    }

    // Required: Allocate memory
    [[nodiscard]] void* allocate(size_type size) {
        // Your allocation logic here
        return std::malloc(size);
    }

    // Required: Allocate aligned memory
    [[nodiscard]] void* allocate(size_type size, size_type alignment) {
        return std::aligned_alloc(alignment, size);
    }

    // Required: Deallocate memory
    void deallocate(void* ptr, size_type size) noexcept {
        (void)size;  // May be unused
        std::free(ptr);
    }

    // Optional: Additional functionality
    [[nodiscard]] size_type available_memory() const noexcept {
        // Return available memory, or 0 if unknown
        return 0;
    }
};

// Concept verification (compile-time check)
static_assert(MemorySpace<my_memory_space>,
              "my_memory_space must satisfy MemorySpace concept");

}  // namespace my_backend
}  // namespace dtl
```

### Example: CUDA Memory Space

Reference `backends/cuda/cuda_memory_space.hpp` for a complete GPU implementation:

```cpp
void* allocate(size_type size) {
#ifdef DTL_ENABLE_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) return nullptr;
    return ptr;
#else
    return nullptr;
#endif
}
```

Key patterns:
- Guard with `#ifdef DTL_ENABLE_CUDA` for optional backends
- Return `nullptr` on allocation failure (don't throw in allocate)
- Track allocations for debugging (`total_allocated_`, `peak_allocated_`)

## Step 2: Implement an Executor

Executors dispatch computation to processing resources.

### Required Interface

```cpp
// backends/my_backend/my_executor.hpp

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/executor.hpp>

#include <functional>

namespace dtl {
namespace my_backend {

class my_executor {
public:
    // Constructor
    my_executor() = default;

    // Required: Execute a callable
    template <typename F>
    void execute(F&& f) {
        std::forward<F>(f)();
    }

    // Required: Get executor name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "my_executor";
    }

    // For ParallelExecutor: Parallel for loop
    template <typename F>
    void parallel_for(size_type count, F&& f) {
        for (size_type i = 0; i < count; ++i) {
            f(i);
        }
    }

    // For ParallelExecutor: Query parallelism
    [[nodiscard]] size_type max_parallelism() const noexcept {
        return 1;  // Or actual concurrency
    }

    [[nodiscard]] size_type suggested_parallelism() const noexcept {
        return max_parallelism();
    }
};

// Concept verification
static_assert(Executor<my_executor>,
              "my_executor must satisfy Executor concept");

}  // namespace my_backend
}  // namespace dtl
```

### Example: CPU Thread Pool Executor

Reference `backends/cpu/cpu_executor.hpp` for a parallel implementation:

```cpp
template <typename F>
void parallel_for(index_t begin, index_t end, F&& func) {
    size_type num_threads = pool_->size();
    index_t chunk_size = (end - begin + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_type t = 0; t < num_threads; ++t) {
        index_t chunk_begin = begin + t * chunk_size;
        index_t chunk_end = std::min(chunk_begin + chunk_size, end);

        if (chunk_begin >= end) break;

        futures.push_back(pool_->submit([=, &func]() {
            for (index_t i = chunk_begin; i < chunk_end; ++i) {
                func(i);
            }
        }));
    }

    for (auto& f : futures) {
        f.wait();
    }
}
```

Key patterns:
- Use a thread pool for concurrent execution
- Chunk work to minimize task overhead
- Wait for all tasks before returning

## Step 3: Implement a Communicator

Communicators provide distributed communication.

### Required Interface

```cpp
// backends/my_backend/my_communicator.hpp

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/communicator.hpp>

namespace dtl {
namespace my_backend {

class my_communicator {
public:
    using size_type = dtl::size_type;

    // Constructor
    my_communicator() : rank_(0), size_(1) {}

    // Required: Query operations
    [[nodiscard]] rank_t rank() const noexcept { return rank_; }
    [[nodiscard]] rank_t size() const noexcept { return size_; }

    // Required: Blocking point-to-point
    void send(const void* buf, size_type count, rank_t dest, int tag) {
        // Send data to dest rank
    }

    void recv(void* buf, size_type count, rank_t source, int tag) {
        // Receive data from source rank
    }

    // Required: Non-blocking point-to-point
    [[nodiscard]] request_handle isend(const void* buf, size_type count,
                                        rank_t dest, int tag) {
        // Initiate non-blocking send
        return request_handle{/* implementation */};
    }

    [[nodiscard]] request_handle irecv(void* buf, size_type count,
                                        rank_t source, int tag) {
        // Initiate non-blocking receive
        return request_handle{/* implementation */};
    }

    // Required: Request completion
    void wait(request_handle& req) {
        // Wait for operation to complete
    }

    bool test(request_handle& req) {
        // Check if operation completed
        return true;
    }

    // For CollectiveCommunicator:
    void barrier() { /* synchronize all ranks */ }
    void broadcast(void* buf, size_type count, rank_t root) { /* ... */ }
    void scatter(const void* sendbuf, void* recvbuf, size_type count, rank_t root) { /* ... */ }
    void gather(const void* sendbuf, void* recvbuf, size_type count, rank_t root) { /* ... */ }
    void allgather(const void* sendbuf, void* recvbuf, size_type count) { /* ... */ }
    void alltoall(const void* sendbuf, void* recvbuf, size_type count) { /* ... */ }

    // For ReducingCommunicator:
    void reduce_sum(const void* sendbuf, void* recvbuf, size_type count, rank_t root) { /* ... */ }
    void allreduce_sum(const void* sendbuf, void* recvbuf, size_type count) { /* ... */ }

private:
    rank_t rank_;
    rank_t size_;
};

// Concept verification
static_assert(Communicator<my_communicator>,
              "my_communicator must satisfy Communicator concept");

}  // namespace my_backend
}  // namespace dtl
```

### Example: MPI Adapter

Reference `backends/mpi/mpi_comm_adapter.hpp` for a complete implementation:

```cpp
void send(const void* buf, size_type count, rank_t dest, int tag) {
    auto result = impl_->send_impl(buf, count, 1, dest, tag);
    if (!result) {
        throw communication_error("MPI send failed to rank " + std::to_string(dest));
    }
}
```

Key patterns:
- Wrap native API (MPI, NCCL) with concept-compliant interface
- Use result types internally, throw on error (for concept compliance)
- Provide both blocking and non-blocking variants

## Step 4: Add CMake Support

### Create CMakeLists.txt

```cmake
# backends/my_backend/CMakeLists.txt

# Find dependencies
find_package(MyBackendSDK REQUIRED)

# Add to DTL
target_sources(dtl INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/my_memory_space.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/my_executor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/my_communicator.hpp
)

target_link_libraries(dtl INTERFACE MyBackendSDK::MyBackendSDK)
```

### Add CMake Option

In root `CMakeLists.txt`:

```cmake
option(DTL_ENABLE_MY_BACKEND "Enable MyBackend support" OFF)

if(DTL_ENABLE_MY_BACKEND)
    add_subdirectory(backends/my_backend)
    target_compile_definitions(dtl INTERFACE DTL_ENABLE_MY_BACKEND)
endif()
```

## Step 5: Write Tests

### Unit Tests

Create tests in `tests/unit/backend/`:

```cpp
// tests/unit/backend/my_backend_test.cpp

#include <gtest/gtest.h>
#include <backends/my_backend/my_memory_space.hpp>
#include <backends/my_backend/my_executor.hpp>

namespace dtl::test {

class MyBackendTest : public ::testing::Test {
protected:
    my_backend::my_memory_space space_;
    my_backend::my_executor exec_;
};

TEST_F(MyBackendTest, MemorySpaceAllocate) {
    void* ptr = space_.allocate(1024);
    ASSERT_NE(ptr, nullptr);
    space_.deallocate(ptr, 1024);
}

TEST_F(MyBackendTest, ExecutorParallelFor) {
    std::atomic<int> counter{0};
    exec_.parallel_for(100, [&](size_t) {
        counter++;
    });
    EXPECT_EQ(counter.load(), 100);
}

}  // namespace dtl::test
```

### Concept Compliance Tests

```cpp
TEST(ConceptTest, MemorySpaceConcept) {
    static_assert(MemorySpace<my_backend::my_memory_space>);
}

TEST(ConceptTest, ExecutorConcept) {
    static_assert(Executor<my_backend::my_executor>);
    static_assert(ParallelExecutor<my_backend::my_executor>);
}
```

## Step 6: Document the Backend

### Add to Backend Selection Guide

Update `docs/backend_guide/backend_selection.md`:

```markdown
### MyBackend

**CMake Configuration:**
```bash
cmake .. -DDTL_ENABLE_MY_BACKEND=ON
```

**Use Case:** [When to use this backend]
```

### Create ADR (if architectural)

If the backend has significant design decisions, create an ADR:

```markdown
# ADR-00XX: MyBackend Integration Contract

**Status:** Proposed
**Date:** YYYY-MM-DD

## Context
[Why this backend is needed]

## Decision
[Key design decisions]

## Consequences
[Impact on DTL]
```

## Checklist

Before submitting a new backend:

- [ ] Memory space implements `MemorySpace` concept
- [ ] Executor implements `Executor` (and optionally `ParallelExecutor`) concept
- [ ] Communicator implements appropriate concept hierarchy
- [ ] `static_assert` verifies concept compliance
- [ ] CMake option and configuration added
- [ ] Unit tests pass
- [ ] Multi-rank tests pass (for communicators)
- [ ] Documentation updated
- [ ] No compiler warnings

## See Also

- [Backend Concepts](concepts.md) - Concept definitions
- [Backend Selection](backend_selection.md) - When to use each backend
