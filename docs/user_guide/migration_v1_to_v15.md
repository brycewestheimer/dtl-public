# Legacy Deep-Dive: Migration (V1.0 to V1.5)

> This page is retained as a **detailed reference**.
> The canonical user path is now the chaptered handbook.

**Primary chapter**: [12-migration-and-upgrade-guidance.md](12-migration-and-upgrade-guidance.md)

**Runtime and handles**: [Runtime and Handle Model](13-runtime-and-handle-model.md)

---

## Detailed Reference (Legacy)


This guide covers all breaking and deprecating changes from DTL V1.0.0 through V1.5.0, with code examples showing how to update your application.

---

## Table of Contents

- [Overview of Changes](#overview-of-changes)
- [Deprecated APIs and Replacements](#deprecated-apis-and-replacements)
  - [environment_base to environment](#environment_base-to-environment)
  - [world_comm to make_world_context](#world_comm-to-make_world_context)
  - [Static Environment Queries to Instance Methods](#static-environment-queries-to-instance-methods)
- [Namespace Changes](#namespace-changes)
  - [Futures Namespace](#futures-namespace)
  - [MPMD Namespace](#mpmd-namespace)
  - [Subsystem Namespaces](#subsystem-namespaces)
  - [Legacy Namespace](#legacy-namespace)
- [Runtime Registry Changes](#runtime-registry-changes)
  - [Environment is Now a Handle](#environment-is-now-a-handle)
  - [Move Semantics](#move-semantics)
  - [Communicator Injection for Libraries](#communicator-injection-for-libraries)
- [Runtime DSO Extraction](#runtime-dso-extraction)
  - [New Runtime Library Dependency](#new-runtime-library-dependency)
  - [Link Flag Changes](#link-flag-changes)
- [Runtime Subproject](#runtime-subproject)
  - [New Runtime Namespace](#new-runtime-namespace)
- [New Convenience Constructors](#new-convenience-constructors)
  - [Single-Rank distributed_vector](#single-rank-distributed_vector)
- [New View Fast Paths](#new-view-fast-paths)
  - [Single-Rank Short Circuit](#single-rank-short-circuit)
  - [Offset Cache in segmented_view](#offset-cache-in-segmented_view)
- [Device Abstraction](#device-abstraction)
- [Build System Changes](#build-system-changes)
- [Quick Migration Checklist](#quick-migration-checklist)

---

## Overview of Changes

| Version | Key Changes |
|---------|-------------|
| V1.0.0 | Initial release. `environment_guard`, `world_comm()`, flat `dtl::` namespace. |
| V1.1.0 | Dirty-state model, STL parity contract, `sync()` API. |
| V1.2.0 | `environment_guard` renamed to `environment`, full backend init, `dtl::legacy::` namespace. |
| V1.2.1 | Namespace organization, complete C ABI and Python bindings. |
| V1.3.0 | Context factory methods (`make_world_context()`), `world_comm()` deprecated. |
| V1.4.0 | Runtime registry, DSO extraction, `from_comm()`, per-instance communicators. |
| V1.5.0 | Runtime subproject, `dtl::runtime::` namespace, capability scaffolds. |

**Backward compatibility:** Most changes are additive. Deprecated APIs remain functional with `[[deprecated]]` warnings. No source-breaking changes are required for V1.0.0 code to compile against V1.5.0, but you will see deprecation warnings.

---

## Deprecated APIs and Replacements

### environment_base to environment

**V1.0.0 (deprecated):**

```cpp
#include <dtl/utility/environment.hpp>

dtl::environment_base env;
dtl::single_environment env(argc, argv);
dtl::scoped_environment env(argc, argv, opts);
```

**V1.5.0 (current):**

```cpp
#include <dtl/core/environment.hpp>

dtl::environment env(argc, argv);
dtl::environment env(argc, argv, opts);
```

The old `environment_base`, `single_environment`, and `scoped_environment` classes are moved to `dtl::legacy::` and marked `[[deprecated]]`. They remain functional but will be removed in a future major version.

### world_comm to make_world_context

**V1.0.0 (deprecated):**

```cpp
dtl::environment env(argc, argv);
auto comm = env.world_comm();  // Returns raw MPI communicator wrapper
// Use comm directly for operations
```

**V1.5.0 (current):**

```cpp
dtl::environment env(argc, argv);
auto ctx = env.make_world_context();  // Returns context<mpi_domain, cpu_domain>
// Use ctx with containers and algorithms
dtl::distributed_vector<double> vec(1000, ctx);
```

`world_comm()` is deprecated because:
- It exposes raw communicator details instead of the type-safe context
- It does not support multi-domain contexts (MPI + CUDA)
- `make_world_context()` provides communicator isolation via per-instance dup'd communicators

For GPU contexts:

```cpp
auto gpu_ctx = env.make_world_context(/*device_id=*/0);
// Returns context<mpi_domain, cpu_domain, cuda_domain>
```

### Static Environment Queries to Instance Methods

**V1.0.0 (deprecated):**

```cpp
// Static methods -- deprecated
if (dtl::environment::has_mpi()) { ... }
if (dtl::environment::is_initialized()) { ... }
auto count = dtl::environment::ref_count();
```

**V1.5.0 (current):**

```cpp
dtl::environment env(argc, argv);

// Instance methods
if (env.has_mpi()) { ... }
if (env.has_cuda()) { ... }
if (env.has_nccl()) { ... }
auto level = env.mpi_thread_level();
auto name = env.domain();  // Named domain for diagnostics
```

The static methods still compile but emit `[[deprecated]]` warnings. They delegate to the runtime registry internally.

---

## Namespace Changes

V1.2.1 formalized the namespace organization into a two-tier model. Most changes are transparent because master headers re-export types into `dtl::`.

### Futures Namespace

Types in the futures subsystem now live in `dtl::futures::`:

```cpp
// Both of these work (master header re-exports into dtl::)
dtl::distributed_future<int> f1;
dtl::futures::distributed_future<int> f2;

// New types in dtl::futures::
dtl::futures::progress_engine::instance().poll();
dtl::futures::when_all(f1, f2);
dtl::futures::when_any(f1, f2);
```

### MPMD Namespace

MPMD types now live in `dtl::mpmd::`:

```cpp
// Both work
dtl::role_manager rm;
dtl::mpmd::role_manager rm2;

dtl::mpmd::rank_group group;
dtl::mpmd::inter_group_communicator igc;
```

### Subsystem Namespaces

| Subsystem | Namespace | Example Types |
|-----------|-----------|---------------|
| RMA | `dtl::rma::` | `rma_guard`, `async_rma_handle` |
| Remote/RPC | `dtl::remote::` | `action`, `action_registry` |
| Topology | `dtl::topology::` | `hardware_topology`, `affinity_mask` |
| Futures | `dtl::futures::` | `distributed_future`, `progress_engine` |
| MPMD | `dtl::mpmd::` | `role_manager`, `rank_group` |
| Device | `dtl::device::` | `device_guard`, `stream_handle`, `device_buffer` |
| Runtime | `dtl::runtime::` | `runtime_registry`, `available_backends()` |

**No action required:** All key types are re-exported into `dtl::` via master headers. Existing `dtl::distributed_future<T>` code continues to compile.

### Legacy Namespace

Deprecated Phase 0 classes are moved to `dtl::legacy::`:

```cpp
// These compile but emit deprecation warnings
dtl::legacy::environment_base env;        // [[deprecated]]
dtl::legacy::single_environment env;      // [[deprecated]]
dtl::legacy::scoped_environment env;      // [[deprecated]]
dtl::legacy::init_options opts;           // [[deprecated]]
```

**Action:** Replace with `dtl::environment` and `dtl::environment_options`.

---

## Runtime Registry Changes

V1.4.0 introduced the runtime registry pattern, fundamentally changing how DTL manages backend state.

### Environment is Now a Handle

Previously, `environment` was a heavyweight object that directly held all backend state. Now it is a lightweight handle/view:

- **Runtime registry** (`dtl::runtime::runtime_registry`): A Meyer's singleton that holds all process-global backend state. Not user-facing.
- **Environment handle** (`dtl::environment`): A lightweight, move-only RAII handle that references the registry.

Each environment instance:
- Holds a reference count to the registry
- Owns a per-instance MPI communicator (dup'd from `MPI_COMM_WORLD`)
- Provides instance-based queries that delegate to the registry

**Impact on existing code:** None for basic usage. The constructor and destructor signatures are unchanged.

### Move Semantics

`environment` is now move-only (it was non-copyable and non-movable in V1.0):

```cpp
// V1.0: Could not move or copy -- had to be stack-allocated
dtl::environment env(argc, argv);

// V1.4+: Move-only (copy is still deleted)
dtl::environment env1(argc, argv);
dtl::environment env2 = std::move(env1);  // OK: transfers comm ownership
// env1 is now in a moved-from state
```

### Communicator Injection for Libraries

V1.4.0 adds the `from_comm()` factory for library authors:

```cpp
// Library code: create an isolated DTL environment from an application's communicator
void my_library_init(MPI_Comm app_comm) {
    auto env = dtl::environment::from_comm(app_comm);
    // env has its own dup'd communicator -- no interference with the app
    auto ctx = env.make_world_context();
    // ... use ctx for library-internal distributed operations
}
```

This enables safe multi-library composition where each library creates its own environment without cross-talk.

---

## Runtime DSO Extraction

V1.4.0 also extracted the runtime registry implementation into a shared library (`libdtl_runtime.so`).

### New Runtime Library Dependency

DTL is no longer purely header-only for the environment subsystem. The runtime registry singleton lives in `libdtl_runtime.so` to ensure exactly one copy of the global state process-wide (Rule 9.1 compliance).

**Templates, containers, policies, algorithms, and views remain header-only.** Only the environment/runtime lifecycle requires the DSO.

### Link Flag Changes

**CMake users:** No changes needed. The `DTL::dtl` target transitively links `DTL::runtime`:

```cmake
find_package(DTL REQUIRED)
target_link_libraries(my_app PRIVATE DTL::dtl)
# libdtl_runtime.so is linked automatically
```

**pkg-config users:** The `Libs` field now includes `-ldtl_runtime`:

```
# Before (V1.0-V1.3)
Libs: -L${libdir}

# After (V1.4+)
Libs: -L${libdir} -ldtl_runtime
```

**Manual linking:**

```bash
# Before
g++ -std=c++20 -I/path/to/dtl/include main.cpp -lmpi

# After
g++ -std=c++20 -I/path/to/dtl/include main.cpp -ldtl_runtime -lmpi
```

---

## Runtime Subproject

V1.5.0 promotes the runtime to a top-level subproject with its own namespace.

### New Runtime Namespace

Runtime services now live in `dtl::runtime::`:

```cpp
#include <dtl/runtime/runtime_registry.hpp>
#include <dtl/runtime/backend_discovery.hpp>
#include <dtl/runtime/plugin_loader.hpp>
#include <dtl/runtime/connection_pool.hpp>

// Registry access (usually not needed by applications)
auto& reg = dtl::runtime::runtime_registry::instance();

// Backend discovery (V1.5.0 scaffolded)
auto backends = dtl::runtime::available_backends();

// Plugin loader (V1.5.0 scaffolded -- returns not_implemented)
auto& plugins = dtl::runtime::plugin_registry::instance();

// Connection pool (V1.5.0 scaffolded -- returns not_implemented)
auto pool = dtl::runtime::make_communicator_pool(config);
```

**Note:** The plugin loader and connection pool are scaffolded in V1.5.0 with stub implementations. They will be fully implemented in future versions.

**Impact on existing code:** The `runtime_registry` was previously in `dtl::detail::`. It is now in `dtl::runtime::`. If you referenced it directly (unlikely for application code), update the namespace.

---

## New Convenience Constructors

### Single-Rank distributed_vector

V1.2+ adds convenience constructors that do not require a context for single-process use:

```cpp
// V1.0: Required context even for single-rank
// auto ctx = env.make_world_context();
// dtl::distributed_vector<int> vec(1000, ctx);

// V1.2+: Single-rank convenience (no context needed)
dtl::distributed_vector<int> vec(1000);         // 1000 default-initialized elements
dtl::distributed_vector<int> vec(1000, 42);     // 1000 elements, all set to 42
```

These constructors create a single-rank vector with all data local. Useful for testing, prototyping, and non-distributed use cases.

---

## New View Fast Paths

### Single-Rank Short Circuit

V1.3+ adds single-rank short circuits to `segmented_view::for_each_local()` and `segmented_view::for_each_segment()`. When `num_ranks() == 1`, the segment machinery is bypassed entirely:

```cpp
auto segv = vec.segmented_view();
// When num_ranks == 1, this directly iterates local data
// without constructing segment descriptors
segv.for_each_local([](double& x) { x *= 2.0; });
```

### Offset Cache in segmented_view

V1.3+ adds an offset cache to `segmented_view` that is built once at construction (O(p)) and enables O(1) offset lookups during iteration. Previously, each segment access required O(p) offset computation.

**Impact:** `segmented_view` iteration is now O(p) total instead of O(p^2) for full traversal. No code changes needed -- the improvement is automatic.

---

## Device Abstraction

V1.5.0 introduces the `dtl::device::` namespace for vendor-agnostic device abstractions:

```cpp
#include <dtl/device/device.hpp>

// Vendor-agnostic types
dtl::device::device_guard guard(0);          // CUDA or HIP device guard
dtl::device::stream_handle stream;           // CUDA or HIP stream
dtl::device::device_buffer<float> buf(1024); // Device memory allocation
dtl::device::event_handle event;             // Synchronization event
```

These types select the underlying backend at compile time:
- With `DTL_ENABLE_CUDA`: maps to CUDA types
- With `DTL_ENABLE_HIP`: maps to HIP types
- Without either: stub implementations

**Migration:** If you were using `dtl::cuda::` types directly, consider migrating to `dtl::device::` for portability. The `dtl::cuda::` types still work but are backend-specific.

---

## Build System Changes

### CMake Minimum Version

No change -- CMake 3.18+ is still required.

### New CMake Targets

| Target | Version | Purpose |
|--------|---------|---------|
| `DTL::dtl` | V1.0+ | Header-only interface library (transitively links runtime since V1.4) |
| `DTL::runtime` | V1.4+ | Runtime shared library (`libdtl_runtime.so`) |
| `DTL::c` | V1.0+ | C ABI library (`libdtl_c.so`) |

### New CMake Options

| Option | Default | Since | Description |
|--------|---------|-------|-------------|
| `DTL_ENABLE_SHMEM` | OFF | V1.2 | Enable OpenSHMEM backend |
| `DTL_BUILD_BENCHMARKS` | OFF | V1.3 | Build performance benchmarks |

### pkg-config Changes

```bash
# V1.0-V1.3
pkg-config --libs dtl
# Output: -L/usr/local/lib

# V1.4+
pkg-config --libs dtl
# Output: -L/usr/local/lib -ldtl_runtime
```

---

## Quick Migration Checklist

For V1.0 code migrating to V1.5:

- [ ] Replace `environment_base`/`single_environment`/`scoped_environment` with `dtl::environment`
- [ ] Replace `world_comm()` calls with `make_world_context()`
- [ ] Replace static `environment::has_mpi()` with instance `env.has_mpi()`
- [ ] Add `-ldtl_runtime` to manual link commands (CMake users: automatic)
- [ ] Ensure `libdtl_runtime.so` is in the library search path at runtime
- [ ] (Optional) Replace `dtl::cuda::` types with `dtl::device::` for portability
- [ ] (Optional) Use single-rank convenience constructors where appropriate

**Compile check:** Build with `-Wall -Wdeprecated-declarations` to find all deprecated API usage.

---

## See Also

- [Environment Guide](environment.md) -- Current environment usage
- [Troubleshooting Guide](troubleshooting.md) -- Common issues
