# OpenSHMEM Backend

**Status:** Production-Ready
**Since:** DTL 0.1.0-alpha.1
**Last Updated:** 2026-02-04

## Overview

The OpenSHMEM backend enables DTL to leverage PGAS (Partitioned Global Address Space) programming with symmetric memory and one-sided communication operations.

## Key Features

- **Symmetric Memory Allocation**: Automatic symmetric heap management
- **One-Sided Communication**: Put/get operations without target participation
- **Atomic Operations**: Fetch-add, compare-swap, swap, fetch, set
- **Synchronization**: Fence, quiet, barrier operations
- **DTL Integration**: Full `memory_window_impl` implementation

## Build Configuration

Enable the SHMEM backend with CMake:

```bash
cmake -DDTL_ENABLE_SHMEM=ON ..
```

### Requirements

- OpenSHMEM 1.4+ implementation (e.g., OpenSHMEM Reference, Sandia OpenSHMEM, OSHMEM)
- C++20 compiler
- Runtime: `oshrun` or equivalent launcher

### Supported Implementations

| Implementation | Status | Notes |
|----------------|--------|-------|
| OpenSHMEM Reference | ✅ Tested | Recommended for development |
| Sandia OpenSHMEM (SOS) | ✅ Tested | High-performance production |
| Open MPI OSHMEM | ✅ Tested | MPI + SHMEM hybrid |
| Cray SHMEM | ✅ Tested | HPC systems |

## Quick Start

### Basic Setup

```cpp
#include <dtl/core/config.hpp>

#if DTL_ENABLE_SHMEM
#include <backends/shmem/shmem_communicator.hpp>
#include <backends/shmem/shmem_memory_window_impl.hpp>

int main() {
    // Initialize SHMEM (RAII)
    dtl::shmem::scoped_shmem_environment env;

    // Get communicator
    auto& comm = dtl::shmem::world_communicator();
    
    printf("PE %d of %d\n", comm.rank(), comm.size());
    
    // Create memory window with symmetric allocation
    auto window_result = dtl::shmem::make_shmem_window(1024);
    if (!window_result) return 1;
    auto& window = *window_result.value();

    // Use window for RMA operations...
    
    return 0;
}
#endif
```

### Running SHMEM Programs

```bash
# Compile
cmake -DDTL_ENABLE_SHMEM=ON ..
make my_shmem_program

# Run with 4 PEs
oshrun -np 4 ./my_shmem_program
```

## API Reference

### Initialization

```cpp
namespace dtl::shmem {

// Manual init/finalize
result<void> init();
void finalize();

// RAII wrapper (recommended)
class scoped_shmem_environment {
    scoped_shmem_environment();   // Calls init()
    ~scoped_shmem_environment();  // Calls finalize()
};

}
```

### Domain and Context

```cpp
// SHMEM domain (rank/size/barrier)
class shmem_domain {
    rank_t rank() const noexcept;
    rank_t size() const noexcept;
    bool valid() const noexcept;
    bool is_root() const noexcept;
    void barrier();
};

// SHMEM context type alias
using shmem_context = context<shmem_domain, cpu_domain>;
```

### Symmetric Memory Allocation

```cpp
namespace dtl::shmem {

// Allocate symmetric memory
result<void*> symmetric_alloc(size_type size);
void symmetric_free(void* ptr);

// Memory space class
class shmem_symmetric_memory_space {
    static void* allocate(size_type size);
    static void* allocate(size_type size, size_type alignment);
    static void deallocate(void* ptr, size_type size) noexcept;
    static void* reallocate(void* ptr, size_type size);
    static void* calloc(size_type count, size_type size);
};

}
```

### Memory Window

```cpp
namespace dtl::shmem {

// Create SHMEM-backed memory window
result<std::unique_ptr<shmem_memory_window_impl>>
make_shmem_window(size_type size);

result<std::unique_ptr<shmem_memory_window_impl>>
make_shmem_window(void* base, size_type size);

class shmem_memory_window_impl : public memory_window_impl {
    // Properties
    void* base() const noexcept override;
    size_type size() const noexcept override;
    bool valid() const noexcept override;

    // Data transfer
    result<void> put(const void* origin, size_type size,
                     rank_t target, size_type target_offset) override;
    result<void> get(void* origin, size_type size,
                     rank_t target, size_type target_offset) override;

    // Non-blocking transfer
    result<void> async_put(const void* origin, size_type size,
                           rank_t target, size_type target_offset,
                           rma_request_handle& request) override;
    result<void> async_get(void* origin, size_type size,
                           rank_t target, size_type target_offset,
                           rma_request_handle& request) override;

    // Atomics
    result<void> fetch_and_op(const void* origin, void* result_buf,
                              size_type size, rank_t target,
                              size_type target_offset, rma_reduce_op op) override;
    result<void> compare_and_swap(const void* origin, const void* compare,
                                  void* result_buf, size_type size,
                                  rank_t target, size_type target_offset) override;

    // Synchronization
    result<void> fence(int assert_flags = 0) override;
    result<void> flush(rank_t target) override;
    result<void> flush_all() override;

    // SHMEM-specific
    void barrier();
    rank_t rank() const noexcept;
    rank_t num_pes() const noexcept;
};

}
```

### RMA Adapter (Low-Level)

```cpp
namespace dtl::shmem {

class shmem_rma_adapter {
    // Communication
    void put(rank_t target, void* dest, const void* source, size_type size);
    void get(rank_t source, void* dest, const void* source, size_type size);
    void put_nbi(rank_t target, void* dest, const void* source, size_type size);
    void get_nbi(rank_t source, void* dest, const void* source, size_type size);

    // Typed operations (int, long, double)
    void put(rank_t target, int* dest, const int* source, size_type count);
    void get(rank_t source, int* dest, const int* source, size_type count);

    // Atomics
    int fetch_add(int* target, int value, rank_t pe);
    int compare_swap(int* target, int cond, int value, rank_t pe);
    int atomic_swap(int* target, int value, rank_t pe);
    int atomic_fetch(const int* target, rank_t pe);
    void atomic_set(int* target, int value, rank_t pe);

    // Synchronization
    void fence();
    void quiet();
    void barrier();
};

shmem_rma_adapter& global_rma_adapter();

}
```

## Supported Operations

### Data Types for Atomics

| Type | fetch_add | compare_swap | swap | fetch | set |
|------|-----------|--------------|------|-------|-----|
| `int` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `long` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `unsigned int` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `unsigned long` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `float` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `double` | ❌ | ❌ | ❌ | ❌ | ❌ |

### RMA Reduce Operations

| Operation | Description | Supported Types |
|-----------|-------------|-----------------|
| `sum` | Add to remote value | int, long |
| `replace` | Swap/replace value | int, long |
| `no_op` | Just fetch | int, long |
| `max` | Maximum (collective only) | int, long |
| `min` | Minimum (collective only) | int, long |

## Synchronization Model

### Ordering Guarantees

```cpp
// fence() - Orders operations before with those after to same target
window.put(data1, size, target, offset1);
window.fence();
window.put(data2, size, target, offset2);  // Guaranteed after data1

// flush_all() / quiet() - Ensure all operations complete
window.async_put(data, size, target, offset, req);
window.flush_all();  // All previous operations complete

// barrier() - Full synchronization across all PEs
window.barrier();    // All PEs reach this point
```

### Comparison with MPI RMA

| Feature | MPI RMA | SHMEM |
|---------|---------|-------|
| Window creation | Explicit `MPI_Win_create` | Implicit (symmetric memory) |
| Epoch management | Required | Not required |
| Default mode | Active or passive | Passive (always accessible) |
| Ordering | Via epochs | Via fence/quiet |
| Lock/unlock | Required for passive | No-op (always valid) |

## Examples

### Put/Get Communication

```cpp
// Create window
auto window = dtl::shmem::make_shmem_window(1024).value();

// Initialize data
int* data = static_cast<int*>(window->base());
data[0] = rank;

window->barrier();

// PE 0 reads from PE 1
if (rank == 0) {
    int remote_value;
    window->get(&remote_value, sizeof(int), 1, 0);
    printf("Read %d from PE 1\n", remote_value);
}
```

### Atomic Counter

```cpp
auto window = dtl::shmem::make_shmem_window(sizeof(int)).value();
int* counter = static_cast<int*>(window->base());
*counter = 0;
window->barrier();

// All PEs increment counter on PE 0
int one = 1;
int old;
window->fetch_and_op(&one, &old, sizeof(int), 0, 0, rma_reduce_op::sum);

window->flush_all();
window->barrier();

if (rank == 0) {
    printf("Final count: %d\n", *counter);  // Equals number of PEs
}
```

### Compare-and-Swap Lock

```cpp
int* lock = static_cast<int*>(window->base());
*lock = 0;  // 0 = unlocked
window->barrier();

// Try to acquire lock
int unlocked = 0, my_id = rank + 1;
int prev;
window->compare_and_swap(&my_id, &unlocked, &prev, sizeof(int), 0, 0);

if (prev == 0) {
    printf("PE %d acquired lock\n", rank);
    // ... critical section ...
    // Release lock
    int zero = 0;
    window->put(&zero, sizeof(int), 0, 0);
    window->flush(0);
}
```

## Testing

### Running SHMEM Tests

```bash
# Build tests
cmake -DDTL_ENABLE_SHMEM=ON -DDTL_BUILD_INTEGRATION_TESTS=ON ..
make dtl_shmem_tests

# Run with oshrun
oshrun -np 2 ./dtl_shmem_tests
oshrun -np 4 ./dtl_shmem_tests
```

### Test Labels

```bash
# Run all SHMEM tests via CTest (if oshrun is detected)
ctest -L shmem

# Specific test sets
ctest -R shmem_integration_2pes
ctest -R shmem_integration_4pes
```

## Troubleshooting

### Common Issues

1. **"SHMEM not initialized"**
   - Ensure `scoped_shmem_environment` is created before any SHMEM operations
   - Or manually call `dtl::shmem::init()` at program start

2. **"Symmetric memory allocation failed"**
   - Increase symmetric heap size: `oshrun -np 4 -e SHMEM_SYMMETRIC_SIZE=256M ./prog`
   - Check available memory

3. **"Invalid PE" errors**
   - Verify target PE is in range `[0, size())`
   - Ensure consistent number of PEs across runs

4. **Segfault on put/get**
   - Verify memory is symmetric (allocated via `shmem_malloc` or `make_shmem_window`)
   - Check offset doesn't exceed window size

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SHMEM_SYMMETRIC_SIZE` | Symmetric heap size | Implementation-specific |
| `SHMEM_DEBUG` | Enable debug output | Off |

## See Also

- [SHMEM Backend MVP Specification](shmem_backend_mvp.md)
- [RMA Operations Guide](../rma/overview.md)
- [Backend Selection Guide](backend_selection.md)
