# RPC Dynamic Handlers

**Since:** v0.2.1 (Phase 09)

This guide explains how to use dynamic RPC handlers in DTL for runtime-registered action dispatch.

## Overview

DTL provides two mechanisms for RPC action dispatch:

| Method | Use Case | Registration | Performance |
|--------|----------|--------------|-------------|
| **Static dispatch** (`static_action_table`) | Known actions at compile time | Compile-time | Fastest |
| **Dynamic dispatch** (`action_registry`) | Runtime action lookup | Runtime via `registry_builder` | Fast with O(n) lookup |

Both methods are fully functional. Choose based on your requirements.

## Quick Start

### 1. Define Actions

```cpp
#include <dtl/remote/action.hpp>

// Define your functions
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
void notify(int value) { /* side effect */ }

// Register as actions
DTL_REGISTER_ACTION(add);
DTL_REGISTER_ACTION(multiply);
DTL_REGISTER_ACTION(notify);
```

### 2. Build a Registry

```cpp
#include <dtl/remote/action_registry.hpp>

auto registry = dtl::remote::registry_builder<64>{}
    .add<dtl::remote::action<&add>>()
    .add<dtl::remote::action<&multiply>>()
    .add<dtl::remote::action<&notify>>()
    .build();
```

### 3. Invoke Dynamically

```cpp
#include <dtl/remote/dynamic_handler.hpp>
#include <dtl/remote/argument_pack.hpp>

// Look up handler
if (auto handler = registry.find(action_add.id())) {
    // Serialize arguments
    std::array<std::byte, 64> request_buf{};
    dtl::remote::argument_pack<int, int>::serialize(10, 32, request_buf.data());
    
    // Invoke
    std::array<std::byte, 64> response_buf{};
    size_t written = handler->invoke(
        request_buf.data(), sizeof(int) * 2,
        response_buf.data(), response_buf.size());
    
    // Deserialize result
    int result = dtl::deserialize<int>(response_buf.data(), written);
    // result == 42
}
```

## High-Level API

For simpler usage without manual serialization:

```cpp
#include <dtl/remote/dynamic_handler.hpp>

auto handler = dtl::remote::make_dynamic_handler<dtl::remote::action<&add>>();

// Invoke with typed arguments and result
auto result = dtl::remote::invoke_handler<int>(handler, 10, 32);
if (result.has_value()) {
    std::cout << "Result: " << result.value() << "\n";  // 42
}
```

## Supported Types

Dynamic handlers support any type with a valid `dtl::serializer<T>` specialization:

| Type Category | Examples | Requirements |
|---------------|----------|--------------|
| Trivially copyable | `int`, `float`, `double`, POD structs | None (automatic) |
| `std::array<T, N>` | `std::array<int, 4>` | T must be trivially serializable |
| `std::pair<T1, T2>` | `std::pair<int, double>` | Both types must be trivially serializable |
| Custom types | User-defined | Specialize `dtl::serializer<T>` |
| Cereal types | Cereal-serializable | `DTL_ENABLE_CEREAL=ON` + `use_cereal_adapter<T>` |
| Boost types | Boost.Serialization | `DTL_ENABLE_BOOST_SERIALIZATION=ON` + `use_boost_adapter<T>` |

### Custom Serializer Example

```cpp
struct MyData {
    int id;
    double value;
};

template <>
struct dtl::serializer<MyData> {
    static size_type serialized_size(const MyData&) { 
        return sizeof(int) + sizeof(double); 
    }
    
    static size_type serialize(const MyData& data, std::byte* buffer) {
        std::memcpy(buffer, &data.id, sizeof(int));
        std::memcpy(buffer + sizeof(int), &data.value, sizeof(double));
        return sizeof(int) + sizeof(double);
    }
    
    static MyData deserialize(const std::byte* buffer, size_type) {
        MyData data;
        std::memcpy(&data.id, buffer, sizeof(int));
        std::memcpy(&data.value, buffer + sizeof(int), sizeof(double));
        return data;
    }
};

// Now MyData works with dynamic handlers
int process_data(MyData data) { return data.id; }
DTL_REGISTER_ACTION(process_data);
```

## Remote Invocation

For cross-rank RPC, use the serialization helpers:

```cpp
#include <dtl/remote/rpc_serialization.hpp>

// On sender
auto request = dtl::remote::serialize_request<dtl::remote::action<&add>>(
    my_rank, request_id, 10, 32);

// Send via MPI or other transport
MPI_Send(request.data(), request.size(), MPI_BYTE, target, tag, comm);

// On receiver
// ... receive message into buffer ...
auto header = dtl::remote::message_header::deserialize(buffer.data());
auto handler = registry.find(header.action);

if (handler) {
    const std::byte* payload = buffer.data() + dtl::remote::message_header::serialized_size();
    std::vector<std::byte> result(256);
    
    size_t written = handler->invoke(
        payload, header.payload_size,
        result.data(), result.size());
    
    // Send response back
    auto response = dtl::remote::serialize_response(
        header.request, my_rank, dtl::deserialize<int>(result.data(), written));
    MPI_Send(response.data(), response.size(), MPI_BYTE, header.source_rank, resp_tag, comm);
}
```

## Error Handling

### Handler Validity

```cpp
auto handler = registry.find(action_id);
if (!handler) {
    // Unknown action ID
}
if (!handler->valid()) {
    // Handler is invalid (should not happen with registry_builder)
}
```

### Extended Error Reporting

```cpp
auto extended = dtl::remote::make_extended_handler<dtl::remote::action<&add>>();

auto result = extended.invoke_with_result(
    request.data(), request.size(),
    response.data(), response.size());

switch (result.error_code) {
    case dtl::remote::handler_result::success:
        // Use result.bytes_written
        break;
    case dtl::remote::handler_result::buffer_too_small:
        // Retry with larger buffer
        break;
    case dtl::remote::handler_result::deserialization_failed:
        // Malformed request
        break;
    case dtl::remote::handler_result::invocation_failed:
        // Exception during function call
        break;
}
```

## Static vs Dynamic Dispatch

### When to Use Static Dispatch

- All actions are known at compile time
- Maximum performance is critical
- Plugin/runtime action registration not needed

```cpp
using ActionTable = dtl::remote::static_action_table<
    dtl::remote::action<&add>,
    dtl::remote::action<&multiply>
>;

ActionTable::dispatch(action_id, [](auto action) {
    // action is the correct type
    using Action = decltype(action);
    // ...
});
```

### When to Use Dynamic Dispatch

- Actions are registered at runtime
- Plugin-based architecture
- Need to iterate over registered actions
- Action set varies between builds

```cpp
auto registry = dtl::remote::registry_builder<64>{}
    .add<dtl::remote::action<&add>>()
    .build();

auto handler = registry.find(action_id);
if (handler) {
    handler->invoke(...);
}
```

## Thread Safety

- `action_registry` is immutable after construction → thread-safe reads
- `registry_builder` is NOT thread-safe during construction
- Handler invocation is thread-safe if the underlying function is thread-safe

## See Also

- [Dynamic Handlers Design](dynamic_handlers_design.md) - Architecture details
- [Serialization](../serialization/external_adapters.md) - External serializer adapters
- [RPC Overview](../api_reference/remote.md) - Complete RPC API reference
