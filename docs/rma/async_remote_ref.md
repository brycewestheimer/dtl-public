# Async remote_ref Operations

**Since:** v1.4.0
**API Status:** Stable

This document describes the async API contract for `remote_ref<T>::async_get()` and `async_put()`.

---

## Overview

`remote_ref<T>` provides explicit handles for accessing elements that may be stored on remote ranks. The async methods allow overlapping communication with computation by returning futures instead of blocking.

---

## API Contract

### Return Types

```cpp
template <typename T>
class remote_ref {
public:
    /// @brief Asynchronously read value from remote element
    /// @return A distributed_future that resolves to the value
    [[nodiscard]] distributed_future<T> async_get() const;

    /// @brief Asynchronously write value to remote element
    /// @param value The value to write
    /// @return A distributed_future that resolves when the write is complete
    distributed_future<void> async_put(const T& value);
};
```

### Semantics

| Property | Guarantee |
|----------|-----------|
| **Non-blocking** | Both methods return immediately without waiting for RMA completion |
| **Progress requirement** | Completion requires either explicit `poll()` or background progress mode |
| **Ordering** | Operations to the same target within the same epoch complete in issue order |
| **Error propagation** | Errors are captured in the future's result (not thrown immediately) |
| **Local elements** | Local accesses complete immediately (future is already ready) |

### Progress Requirements

Async remote_ref operations integrate with the DTL futures progress engine. Completion requires one of:

1. **Explicit polling:** Call `dtl::futures::poll()` in your main loop
2. **Background mode:** Enable `environment_options::enable_background_progress`

```cpp
// Option 1: Explicit polling
auto future = ref.async_get();
while (!future.ready()) {
    dtl::futures::poll();
}
T value = future.get().value();

// Option 2: Poll-until helper
auto future = ref.async_get();
dtl::futures::poll_until([&] { return future.ready(); });
T value = future.get().value();

// Option 3: Background progress (configured at environment setup)
dtl::environment_options opts;
opts.enable_background_progress = true;
dtl::environment env(opts);

auto future = ref.async_get();
T value = future.get().value();  // Progress happens automatically
```

### Ordering Guarantees

Within a single `memory_window` and to the same target rank:

- Operations are completed in issue order
- A `flush()` ensures remote visibility
- No ordering between operations to different targets unless synchronized

---

## Backend Implementation

### MPI Backend

The MPI backend uses MPI-3 request-based RMA operations:

| DTL Operation | MPI Call |
|---------------|----------|
| `async_get()` | `MPI_Rget` |
| `async_put()` | `MPI_Rput` |

Completion is tested via `MPI_Test()` during progress engine polls.

### Local Optimization

When `is_local() == true`:
- No MPI call is issued
- The future is immediately ready
- Direct memory access is used

---

## Error Handling

Errors are captured in the future's result:

```cpp
auto future = ref.async_get();
auto result = future.get();

if (result.has_error()) {
    // Handle error
    auto status = result.error();
    // status.code() returns status_code::...
}
```

Possible errors:

| Error Code | Description |
|------------|-------------|
| `status_code::invalid_state` | Window is invalid or not locked |
| `status_code::invalid_rank` | Target rank is invalid |
| `status_code::out_of_range` | Offset exceeds window bounds |
| `status_code::not_implemented` | No window available and element is remote |

---

## Best Practices

1. **Batch operations:** Issue multiple async ops before polling for better overlap
2. **Use local_view:** For elements known to be local, use `local_view()` for direct access
3. **Poll regularly:** In event loops, call `poll()` at least once per iteration
4. **Consider background mode:** For fire-and-forget patterns, enable background progress

---

## Example

```cpp
#include <dtl/views/remote_ref.hpp>
#include <dtl/futures/progress.hpp>

void example(dtl::distributed_vector<int>& vec) {
    auto gview = vec.global_view();
    
    // Issue multiple async operations
    std::vector<dtl::distributed_future<int>> futures;
    for (size_t i = 0; i < 100; i += 10) {
        auto ref = gview[i];
        if (ref.is_remote()) {
            futures.push_back(ref.async_get());
        }
    }
    
    // Poll until all complete
    while (std::any_of(futures.begin(), futures.end(),
                       [](auto& f) { return !f.ready(); })) {
        dtl::futures::poll();
    }
    
    // Collect results
    for (auto& f : futures) {
        int value = f.get().value();
        // process value...
    }
}
```

---

## See Also

- [Progress Engine Documentation](../futures/progress_engine.md)
- [Memory Window API](memory_window.md)
- [RMA Operations](rma_operations.md)
