# Legacy Deep-Dive: Environment

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [03-environment-context-and-backends.md](03-environment-context-and-backends.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


DTL provides `dtl::environment` as the recommended way to manage backend lifecycle and context creation.

---

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Backend Ownership](#backend-ownership)
- [Context Factory Methods](#context-factory-methods)
- [Multi-Domain Contexts](#multi-domain-contexts)
- [Best Practices](#best-practices)

---

## Overview

The `dtl::environment` class:

- Initializes MPI and other backends automatically
- Manages backend ownership (owned vs borrowed)
- Provides factory methods for context creation
- Handles finalization on destruction (RAII)
- Is the recommended entry point for DTL programs

---

## Basic Usage

```cpp
#include <dtl/dtl.hpp>

int main(int argc, char** argv) {
    // Environment initializes MPI if not already initialized
    dtl::environment env(argc, argv);

    // Create a context for distributed operations
    auto ctx = env.make_world_context();

    std::cout << "Rank " << ctx.rank() << " of " << ctx.size() << "\n";

    // Use context with containers
    dtl::distributed_vector<double> vec(ctx, 1000);

    // Fill local partition
    auto local = vec.local_view();
    for (std::size_t i = 0; i < local.size(); ++i) {
        local[i] = ctx.rank() * 1000.0 + i;
    }

    // Collective operation
    double sum = dtl::distributed_reduce(vec, 0.0, std::plus<>{});

    if (ctx.rank() == 0) {
        std::cout << "Global sum: " << sum << "\n";
    }

    // Environment automatically finalizes MPI on destruction
    return 0;
}
```

---

## Backend Ownership

### Owned Mode (Default)

In owned mode, DTL takes responsibility for initializing and finalizing backends:

```cpp
// DTL owns MPI - will call MPI_Init and MPI_Finalize
dtl::environment env(argc, argv);

// ... use DTL ...

// MPI_Finalize called automatically when env goes out of scope
```

### Borrowed Mode

If your application already manages MPI lifecycle, use borrowed mode:

```cpp
// User manages MPI lifecycle
MPI_Init(&argc, &argv);

// Tell DTL that MPI is already initialized
dtl::environment_options opts;
opts.mpi_ownership = dtl::backend_ownership::borrowed;
dtl::environment env(argc, argv, opts);

// ... use DTL ...

// Environment destructor does NOT call MPI_Finalize

// User responsible for finalization
MPI_Finalize();
```

### Checking Backend State

```cpp
dtl::environment env(argc, argv);

// Check what backends are available
if (env.has_mpi()) {
    std::cout << "MPI initialized\n";
}
if (env.has_cuda()) {
    std::cout << "CUDA available\n";
}
```

---

## Context Factory Methods

The environment provides factory methods to create different types of contexts:

### MPI World Context

```cpp
dtl::environment env(argc, argv);

// Create context spanning all MPI ranks
auto ctx = env.make_world_context();
// ctx.size() == total MPI ranks
// ctx.rank() == this process's rank
```

### MPI+CUDA Context

```cpp
// Create context with GPU support (if CUDA enabled)
auto cuda_ctx = env.make_world_context(/*device_id=*/0);
```

### CPU-Only Context

```cpp
// Create a local-only context (no MPI communication)
auto cpu_ctx = env.make_cpu_context();
// cpu_ctx.size() == 1
// cpu_ctx.rank() == 0
```

### Custom Communicator Context

```cpp
// Create context from a custom MPI communicator
MPI_Comm sub_comm;
MPI_Comm_split(MPI_COMM_WORLD, color, key, &sub_comm);

auto sub_ctx = env.make_context(sub_comm);
```

---

## Multi-Domain Contexts

DTL V1.3.0 introduces multi-domain contexts for heterogeneous computing. A domain represents a specific execution environment (MPI, CUDA, etc.).

### Basic Multi-Domain Context

```cpp
dtl::environment env(argc, argv);

// Context with MPI + CUDA domains (if both enabled)
auto ctx = env.make_world_context(0);  // 0 = CUDA device ID

// The context provides access to both domains
auto& mpi_domain = ctx.get<dtl::mpi_domain>();
auto& cuda_domain = ctx.get<dtl::cuda_domain>();
```

### Querying Domain Support

```cpp
dtl::environment env(argc, argv);
auto ctx = env.make_world_context();

// Check if context has CUDA domain
if constexpr (ctx.has_domain<dtl::cuda_domain>()) {
    // Use CUDA features
}
```

---

## Best Practices

### 1. One Environment Per Program

Create the environment once in `main()`:

```cpp
int main(int argc, char** argv) {
    dtl::environment env(argc, argv);  // Single instance

    run_simulation(env);
    run_analysis(env);

    return 0;
}
```

### 2. Use RAII

Let the environment destructor handle cleanup:

```cpp
// GOOD: RAII handles cleanup
{
    dtl::environment env(argc, argv);
    // ... use DTL ...
}  // Automatic cleanup here

// BAD: Manual finalization
dtl::environment* env = new dtl::environment(argc, argv);
// ... use DTL ...
delete env;  // Error-prone, may leak on exception
```

### 3. Check Backend Availability

Before using optional features:

```cpp
dtl::environment env(argc, argv);

if (!env.has_mpi()) {
    std::cerr << "Error: MPI required for distributed execution\n";
    return 1;
}

if (!env.has_cuda()) {
    std::cout << "Warning: CUDA not available, using CPU fallback\n";
}
```

### 4. Pass Context, Not Environment

Functions should accept contexts, not environments:

```cpp
// GOOD: Function takes context
void compute(const dtl::context& ctx) {
    dtl::distributed_vector<double> vec(ctx, 1000);
    // ...
}

// LESS GOOD: Function takes environment
void compute(dtl::environment& env) {
    auto ctx = env.make_world_context();  // Creates new context each call
    // ...
}

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    compute(ctx);  // Pass context
}
```

---

## Error Handling

### Initialization Failures

```cpp
try {
    dtl::environment env(argc, argv);
} catch (const dtl::backend_error& e) {
    std::cerr << "Backend initialization failed: " << e.what() << "\n";
    return 1;
}
```

### Missing Requirements

```cpp
dtl::environment_options opts;
opts.require_mpi = true;  // Throw if MPI unavailable

try {
    dtl::environment env(argc, argv, opts);
} catch (const dtl::missing_backend_error& e) {
    std::cerr << "Required backend not available: " << e.what() << "\n";
    return 1;
}
```

---

## See Also

- [Getting Started](../getting_started.md) - Installation and first program
- [Containers Guide](containers.md) - Using distributed containers with contexts
