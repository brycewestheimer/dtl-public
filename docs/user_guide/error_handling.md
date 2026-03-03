# Legacy Deep-Dive: Error Handling

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [09-error-handling-and-reliability.md](09-error-handling-and-reliability.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


DTL provides two error handling modes: result-based (default) and exception-based. This guide covers both approaches and how to handle distributed errors.

---

## Table of Contents

- [Overview](#overview)
- [Error Categories](#error-categories)
- [Result-Based Error Handling](#result-based-error-handling)
- [Exception-Based Error Handling](#exception-based-error-handling)
- [Collective Errors](#collective-errors)
- [Structural Invalidation](#structural-invalidation)
- [Best Practices](#best-practices)

---

## Overview

Distributed systems introduce error scenarios not present in single-process code:
- Communication failures
- Rank failures
- Collective operation mismatches
- Structural invalidation across ranks

DTL handles these through configurable error policies.

### Error Policy Selection

```cpp
// Result-based (default)
dtl::distributed_vector<double, ..., dtl::expected> vec(1000, size, rank);

// Exception-based
dtl::distributed_vector<double, ..., dtl::throwing> vec(1000, size, rank);
```

---

## Error Categories

DTL defines standard error codes for all operations:

| Error Code | Description |
|------------|-------------|
| `ok` | Operation succeeded |
| `invalid_argument` | Invalid parameter |
| `out_of_range` | Index out of bounds |
| `not_supported` | Operation not supported |
| `backend_failure` | Backend (MPI, CUDA) error |
| `collective_failure` | Collective operation failed |
| `serialization_error` | Serialization failed |
| `consistency_violation` | Consistency policy violated |
| `structural_invalidation` | View/iterator invalidated |
| `timeout` | Operation timed out |
| `canceled` | Operation canceled |

### Checking Error Codes

```cpp
dtl::error err = some_operation();

switch (err.code()) {
    case dtl::error_code::ok:
        // Success
        break;
    case dtl::error_code::out_of_range:
        // Handle bounds error
        break;
    case dtl::error_code::collective_failure:
        // Handle distributed failure
        break;
    default:
        // Handle other errors
        break;
}
```

---

## Result-Based Error Handling

The default mode uses `result<T>` (similar to `std::expected`).

### Basic Usage

```cpp
dtl::distributed_vector<double> vec(1000, size, rank);  // default: expected policy
auto global = vec.global_view();

// get() returns result<T>
auto result = global[500].get();

if (result.has_value()) {
    double val = result.value();
    std::cout << "Value: " << val << "\n";
} else {
    dtl::error err = result.error();
    std::cerr << "Error: " << err.message() << "\n";
}
```

### Result Type API

```cpp
dtl::result<T> result = operation();

// Check success
if (result) { ... }           // Explicit bool conversion
if (result.has_value()) { ... }

// Access value (precondition: has_value())
T val = result.value();
T val = *result;              // Same as value()

// Access value with default
T val = result.value_or(default_value);

// Access error (precondition: !has_value())
dtl::error err = result.error();
```

### Chaining Operations

```cpp
auto result = global[idx].get()
    .and_then([](double x) -> dtl::result<double> {
        return x * 2.0;
    })
    .or_else([](dtl::error e) -> dtl::result<double> {
        std::cerr << "Error: " << e.message() << "\n";
        return 0.0;  // Default value
    });
```

### Void Results

Operations that don't return a value use `result<void>`:

```cpp
dtl::result<void> result = global[500].put(42.0);

if (!result) {
    std::cerr << "Put failed: " << result.error().message() << "\n";
}
```

---

## Exception-Based Error Handling

Enable exceptions with the `throwing` error policy.

### Configuration

```cpp
// Container with throwing policy
dtl::distributed_vector<double, dtl::block_partition<>, dtl::host_only,
                        dtl::seq, dtl::bulk_synchronous, dtl::throwing> vec(1000, size, rank);
```

### Basic Usage

```cpp
try {
    auto global = vec.global_view();
    double val = global[500].get();  // Throws on error
    global[500].put(42.0);           // Throws on error
} catch (const dtl::dtl_exception& e) {
    std::cerr << "DTL error: " << e.what() << "\n";
    std::cerr << "Error code: " << static_cast<int>(e.code()) << "\n";
}
```

### Exception Hierarchy

```
std::exception
└── dtl::dtl_exception
    ├── dtl::invalid_argument_error
    ├── dtl::out_of_range_error
    ├── dtl::communication_error
    │   └── dtl::collective_error
    ├── dtl::serialization_error
    ├── dtl::consistency_error
    └── dtl::invalidation_error
```

### Catching Specific Exceptions

```cpp
try {
    // ... operations ...
} catch (const dtl::collective_error& e) {
    // Handle collective failure specifically
    std::cerr << "Collective failed: " << e.what() << "\n";
    // e.failing_ranks() may be available
} catch (const dtl::communication_error& e) {
    // Handle any communication error
    std::cerr << "Communication error: " << e.what() << "\n";
} catch (const dtl::dtl_exception& e) {
    // Handle any DTL error
    std::cerr << "DTL error: " << e.what() << "\n";
}
```

---

## Collective Errors

Distributed operations may fail on subsets of ranks. DTL provides `collective_error` to aggregate failure information.

### Collective Error Structure

```cpp
struct collective_error {
    error_code code;
    std::string message;

    // Did any rank fail?
    bool any_failed() const;

    // Get representative error
    error representative_error() const;

    // Get failing ranks (if available)
    std::optional<std::vector<rank_t>> failing_ranks() const;
};
```

### Handling Collective Errors

```cpp
// Result-based
auto result = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
if (!result) {
    auto& err = result.error();
    if (err.code() == dtl::error_code::collective_failure) {
        // Some ranks failed
        if (auto ranks = err.failing_ranks()) {
            std::cerr << "Failing ranks: ";
            for (auto r : *ranks) std::cerr << r << " ";
            std::cerr << "\n";
        }
    }
}

// Exception-based
try {
    double sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
} catch (const dtl::collective_error& e) {
    std::cerr << "Collective failed: " << e.what() << "\n";
    // Access failing rank information
}
```

### Collective Semantics

All ranks must participate in collective operations:

```cpp
// WRONG: Only some ranks call reduce
if (rank < 2) {
    auto sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});  // DEADLOCK
}

// CORRECT: All ranks call
auto sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
```

---

## Structural Invalidation

Views and iterators are invalidated by structural operations. Using invalidated views produces deterministic errors.

### What Causes Invalidation

| Operation | Invalidates Views? |
|-----------|-------------------|
| `resize()` | Yes |
| `redistribute()` | Yes |
| Element modification | No |
| Obtaining new views | No |

### Detection

```cpp
auto local = vec.local_view();
// ... use local ...

vec.resize(2000);  // Structural operation

// local is now invalid
// Result-based:
auto result = local[0];  // Returns error with structural_invalidation

// Exception-based:
try {
    auto val = local[0];  // Throws dtl::invalidation_error
} catch (const dtl::invalidation_error& e) {
    std::cerr << "View invalidated: " << e.what() << "\n";
}
```

### Safe Pattern

```cpp
void process(dtl::distributed_vector<double>& vec) {
    auto local = vec.local_view();

    // Phase 1: Use view
    for (double& x : local) {
        x *= 2.0;
    }

    // Phase 2: Structural change
    if (needs_resize()) {
        vec.resize(new_size);
        local = vec.local_view();  // Get fresh view
    }

    // Phase 3: Continue with valid view
    for (double& x : local) {
        x += 1.0;
    }
}
```

---

## Best Practices

### 1. Choose Error Policy Based on Use Case

```cpp
// For production/library code: result-based
// - Forces explicit error handling
// - No unexpected stack unwinding
dtl::distributed_vector<double, ..., dtl::expected> prod_vec(1000, size, rank);

// For development/debugging: exception-based
// - Stack traces on error
// - Easier to debug
dtl::distributed_vector<double, ..., dtl::throwing> debug_vec(1000, size, rank);
```

### 2. Always Handle Collective Errors

```cpp
auto result = dtl::distributed_reduce(vec, 0.0, std::plus<>{});

// Don't ignore collective failures
if (!result) {
    // Log and potentially abort
    MPI_Abort(MPI_COMM_WORLD, 1);
}
```

### 3. Use RAII for View Management

```cpp
class view_scope {
    distributed_vector<double>& vec_;
    decltype(vec_.local_view()) view_;

public:
    view_scope(distributed_vector<double>& vec)
        : vec_(vec), view_(vec.local_view()) {}

    auto& get() { return view_; }

    void refresh() { view_ = vec_.local_view(); }
};
```

### 4. Propagate Errors in Helper Functions

```cpp
// Result-based
dtl::result<double> compute_sum(const dtl::distributed_vector<double>& vec) {
    auto result = dtl::distributed_reduce(vec, 0.0, std::plus<>{});
    if (!result) {
        return result.error();  // Propagate error
    }
    return result.value();
}

// Exception-based: errors propagate automatically
double compute_sum(const dtl::distributed_vector<double>& vec) {
    return dtl::distributed_reduce(vec, 0.0, std::plus<>{});
}
```

### 5. Document Error Conditions

```cpp
/// @brief Computes global average
/// @param vec Input vector
/// @return Global average value
/// @throws dtl::collective_error If collective operation fails
/// @throws dtl::invalid_argument_error If vector is empty
double global_average(const dtl::distributed_vector<double>& vec);
```

### 6. Use Error Policy Consistently

```cpp
// All containers in a program should typically use the same policy
template<typename T>
using my_vector = dtl::distributed_vector<T, dtl::block_partition<>,
                                          dtl::host_only, dtl::seq,
                                          dtl::bulk_synchronous,
                                          MY_ERROR_POLICY>;
```

---

## Error Information

### Error Object API

```cpp
dtl::error err = ...;

// Error code
dtl::error_code code = err.code();

// Human-readable message
std::string msg = err.message();

// Check specific codes
bool is_timeout = (err.code() == dtl::error_code::timeout);

// Optional: backend-specific info
if (auto* mpi_err = err.backend_error<mpi_error>()) {
    int mpi_code = mpi_err->code;
}
```

### Custom Error Messages

```cpp
dtl::error custom_err(dtl::error_code::invalid_argument,
                      "Custom error message: value out of range");
```

---

## See Also

- [Policies Guide](policies.md) - Error policy configuration
- [Views Guide](views.md) - View invalidation details
