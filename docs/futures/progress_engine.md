# Progress Engine

The DTL futures system uses a **progress engine** to drive asynchronous operations to completion. This document explains how to use the progress engine effectively.

## Overview

The progress engine is a poll-based system that advances pending async operations when polled. Unlike thread-per-operation models, the progress engine gives applications control over when and how progress is made.

## When to Call poll()

### Explicit Polling (Default Mode)

By default, DTL uses explicit polling mode. You need to call `poll()` periodically to advance async operations:

```cpp
#include <dtl/futures/futures.hpp>

// Create an async operation
auto future = async_operation();

// Poll until ready
while (!future.is_ready()) {
    dtl::poll();  // Drive progress
    // Optionally do other work here
}

auto result = future.get();
```

### Polling in Application Loops

For applications with main loops (game engines, servers, etc.), integrate polling:

```cpp
void main_loop() {
    while (running) {
        // Application work
        process_input();
        update_state();
        render();

        // Drive DTL progress
        dtl::poll();
    }
}
```

### Bounded Polling

Use `poll_for()` to dedicate a time slice to progress:

```cpp
// Poll for up to 10ms
dtl::poll_for(std::chrono::milliseconds(10));
```

### Conditional Polling

Use `poll_until()` to poll until a condition is met:

```cpp
std::atomic<bool> data_ready{false};

// Poll until data is ready or timeout
bool success = dtl::poll_until(
    [&data_ready] { return data_ready.load(); },
    std::chrono::seconds(5)  // Optional timeout
);
```

## Background Progress Mode

For applications that prefer automatic progress, enable background mode:

```cpp
#include <dtl/futures/background_progress.hpp>

// Enable background progress
dtl::start_background_progress();

// Now futures complete without explicit polling
auto future = async_operation();
future.wait();  // Works without explicit poll()

// Disable when done
dtl::stop_background_progress();
```

### Scoped Background Progress

Use RAII for automatic cleanup:

```cpp
{
    dtl::scoped_background_progress guard;
    
    // Background progress is active within this scope
    auto f1 = async_op1();
    auto f2 = async_op2();
    
    f1.wait();
    f2.wait();
}  // Background progress stops here
```

### Configuration

Configure background polling behavior:

```cpp
auto config = dtl::background_progress_config::background_mode();
config.poll_interval = std::chrono::microseconds(50);  // Poll frequency
config.adaptive_polling = true;  // Reduce frequency when idle

dtl::start_background_progress(config);
```

## Callback Executor (Isolation)

Long-running callbacks can block progress on unrelated futures. The callback executor provides isolation:

```cpp
#include <dtl/futures/callback_executor.hpp>

// The global callback executor isolates callbacks
auto& executor = dtl::global_callback_executor();

// Long callbacks don't block progress
auto future = compute().then([](Result r) {
    // This callback runs on the executor, not the progress thread
    expensive_processing(r);
    return processed_result;
});
```

### Custom Executor Configuration

```cpp
// Use thread pool for parallel callback execution
auto config = dtl::executor_config::thread_pool_execution(4);
dtl::callback_executor executor(config);
```

## Timeout Configuration

Configure wait timeouts to prevent indefinite hangs:

```cpp
#include <dtl/futures/diagnostics.hpp>

// Set custom timeout
auto config = dtl::timeout_config::defaults();
config.default_wait_timeout = std::chrono::seconds(60);
config.enable_timeout_diagnostics = true;

dtl::set_global_timeout_config(config);
```

### Timeout Handling

```cpp
try {
    future.wait();
} catch (const dtl::timeout_exception& ex) {
    // Get diagnostic information
    const auto& diag = ex.diagnostics();
    std::cerr << diag.to_string();
}
```

### CI Mode

Set `DTL_CI_MODE` environment variable for CI-specific timeouts:

```bash
export DTL_CI_MODE=1
```

## Diagnostics

Get diagnostic information about the progress engine:

```cpp
auto& collector = dtl::diagnostic_collector::instance();
auto diag = collector.get_diagnostics();

std::cout << "Pending callbacks: " << diag.pending_callback_count << "\n";
std::cout << "Total polls: " << diag.total_polls << "\n";
std::cout << diag.to_string();  // Full diagnostic dump
```

## Best Practices

### 1. Poll Regularly

In explicit mode, poll at least every 10-100ms to ensure timely completion:

```cpp
// In your main loop
constexpr auto POLL_INTERVAL = std::chrono::milliseconds(10);
auto last_poll = std::chrono::steady_clock::now();

void tick() {
    auto now = std::chrono::steady_clock::now();
    if (now - last_poll > POLL_INTERVAL) {
        dtl::poll();
        last_poll = now;
    }
}
```

### 2. Use Background Mode for Simple Applications

If you don't need fine-grained control, background mode is simpler:

```cpp
int main() {
    dtl::scoped_background_progress bg;
    
    // Just use futures normally
    auto result = async_compute().get();
    return 0;
}
```

### 3. Avoid Long Callbacks

Keep continuation callbacks short. Offload heavy work:

```cpp
// Bad: Long callback blocks progress
auto future = fetch_data().then([](Data d) {
    return expensive_computation(d);  // Blocks other futures!
});

// Good: Offload to thread pool
auto future = fetch_data().then([](Data d) {
    return std::async(std::launch::async, expensive_computation, d);
});
```

### 4. Set Reasonable Timeouts

Configure timeouts appropriate for your workload:

```cpp
// For quick operations
config.default_wait_timeout = std::chrono::seconds(10);

// For batch jobs
config.default_wait_timeout = std::chrono::minutes(30);

// For development/debugging
config.default_wait_timeout = std::chrono::milliseconds(0);  // No timeout
```

## Thread Safety

- `poll()` is thread-safe (only one thread polls at a time)
- Multiple threads can call `poll()` concurrently (extras return immediately)
- Callbacks are invoked under a mutex to prevent double-execution
- Background progress and explicit polling can coexist

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DTL_CI_MODE` | Enables CI-specific timeout (30s default) |

## API Reference

### Polling Functions

- `poll()` - Poll once, return progress count
- `poll_one()` - Complete at least one operation
- `poll_for(duration)` - Poll for specified time
- `poll_until(predicate, timeout)` - Poll until predicate is true
- `drain_progress(max_iterations)` - Poll until empty

### Background Progress

- `start_background_progress(config)` - Enable background polling
- `stop_background_progress()` - Disable background polling
- `is_background_progress_enabled()` - Check if active
- `scoped_background_progress` - RAII guard

### Configuration

- `timeout_config` - Timeout settings
- `background_progress_config` - Background polling settings
- `executor_config` - Callback executor settings

### Diagnostics

- `diagnostic_collector::instance()` - Get collector
- `progress_diagnostics` - Snapshot of engine state
- `timeout_exception` - Exception with diagnostics

---

## RMA Integration (v1.4.0+)

The progress engine now drives async RMA operations via `remote_ref`:

```cpp
#include <dtl/views/remote_ref.hpp>
#include <dtl/futures/progress.hpp>

// Create a remote reference
remote_ref<int> ref(target_rank, index, nullptr, window_impl, offset);

// Async get returns a distributed_future
auto future = ref.async_get();

// Poll until ready
while (!future.is_ready()) {
    dtl::futures::poll();
}

// Get the result
int value = future.get_result().value();
```

### Multiple RMA Operations

Issue multiple async ops for better overlap:

```cpp
std::vector<distributed_future<int>> futures;

// Issue all operations first
for (auto& ref : remote_refs) {
    futures.push_back(ref.async_get());
}

// Then poll until all complete
while (std::any_of(futures.begin(), futures.end(),
                   [](auto& f) { return !f.is_ready(); })) {
    dtl::futures::poll();
}

// Collect results
for (auto& f : futures) {
    int val = f.get_result().value();
    // process...
}
```

See [Async remote_ref Documentation](../rma/async_remote_ref.md) for full API contract.

