# C Bindings Guide

This guide covers the C ABI bindings for DTL, providing a stable interface for C programs and as the foundation for bindings in other languages.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [Context Operations](#context-operations)
  - [Communicator Operations](#communicator-operations)
  - [Container Operations](#container-operations)
  - [Collective Operations](#collective-operations)
  - [Algorithm Operations](#algorithm-operations)
  - [Policy Selection](#policy-selection)
  - [RMA Operations](#rma-operations)
- [Error Handling](#error-handling)
- [Thread Safety](#thread-safety)
- [Building FFI for Other Languages](#building-ffi-for-other-languages)
- [Complete Example](#complete-example)

---

## Overview

The C bindings provide:

- **ABI Stability**: Binary compatibility across minor versions
- **Language Neutrality**: Works with C99 and C11 compilers
- **Clear Ownership**: Explicit memory management via naming conventions
- **Error Transparency**: All errors surfaced via status codes

### Design Goals

1. Enable C programs to use DTL
2. Provide a stable foundation for bindings in other languages (Python, Fortran, Julia, etc.)
3. Maintain minimal overhead over native C++ calls

---

## Installation

### Building the C Library

```bash
cmake .. -DDTL_BUILD_C_BINDINGS=ON
make dtl_c
```

This produces `libdtl_c.so` (Linux/macOS) or `dtl_c.dll` (Windows).

### Linking

```bash
# Compile C program
gcc -std=c99 -o my_program my_program.c -ldtl_c -lmpi

# With explicit paths
gcc -std=c99 -I/path/to/dtl/include -L/path/to/dtl/lib \
    -o my_program my_program.c -ldtl_c -lmpi
```

### Headers

```c
// Master header includes everything
#include <dtl/bindings/c/dtl.h>

// Or include specific headers
#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_vector.h>
```

---

## Core Concepts

### Opaque Handles

All DTL objects are accessed through opaque handle types:

```c
typedef struct dtl_context_s*      dtl_context_t;
typedef struct dtl_vector_s*       dtl_vector_t;
typedef struct dtl_tensor_s*       dtl_tensor_t;
typedef struct dtl_request_s*      dtl_request_t;
```

### Memory Ownership Conventions

Function naming indicates ownership semantics:

| Suffix | Meaning | Caller Responsibility |
|--------|---------|----------------------|
| `_create` | Creates new owned object | MUST call `_destroy` |
| `_destroy` | Releases owned object | Object becomes invalid |
| `_get` | Returns borrowed pointer | MUST NOT free |
| `_take` | Transfers ownership to caller | Caller MUST free/destroy |
| `_give` | Transfers ownership from caller | Caller MUST NOT use after |

### Basic Types

```c
typedef int32_t  dtl_rank_t;    // MPI rank
typedef uint64_t dtl_size_t;    // Size type
typedef int64_t  dtl_index_t;   // Index type (signed for offsets)
typedef int32_t  dtl_status;    // Status code
```

### Data Types

```c
typedef enum dtl_dtype {
    DTL_DTYPE_INT8    = 0,
    DTL_DTYPE_INT16   = 1,
    DTL_DTYPE_INT32   = 2,
    DTL_DTYPE_INT64   = 3,
    DTL_DTYPE_UINT8   = 4,
    DTL_DTYPE_UINT16  = 5,
    DTL_DTYPE_UINT32  = 6,
    DTL_DTYPE_UINT64  = 7,
    DTL_DTYPE_FLOAT32 = 8,
    DTL_DTYPE_FLOAT64 = 9,
    DTL_DTYPE_BYTE    = 10
} dtl_dtype;
```

---

## API Reference

### Environment Operations

The environment manages backend lifecycle (MPI, CUDA, HIP, NCCL, SHMEM) using reference-counted RAII semantics. The first `create` call initializes backends; the last `destroy` finalizes them.

```c
// Lifecycle
dtl_status dtl_environment_create(dtl_environment_t* env);
dtl_status dtl_environment_create_with_args(dtl_environment_t* env, int* argc, char*** argv);
void       dtl_environment_destroy(dtl_environment_t env);

// State queries
int        dtl_environment_is_initialized(void);
dtl_size_t dtl_environment_ref_count(void);

// Backend availability
int dtl_environment_has_mpi(void);
int dtl_environment_has_cuda(void);
int dtl_environment_has_hip(void);
int dtl_environment_has_nccl(void);
int dtl_environment_has_shmem(void);
int dtl_environment_mpi_thread_level(void);

// Context factories
dtl_status dtl_environment_make_world_context(dtl_environment_t env, dtl_context_t* ctx);
dtl_status dtl_environment_make_world_context_gpu(dtl_environment_t env, int device_id, dtl_context_t* ctx);
dtl_status dtl_environment_make_cpu_context(dtl_environment_t env, dtl_context_t* ctx);
```

**Example:**

```c
int main(int argc, char** argv) {
    dtl_environment_t env;
    dtl_status status = dtl_environment_create_with_args(&env, &argc, &argv);
    if (status != DTL_SUCCESS) {
        fprintf(stderr, "Init failed: %s\n", dtl_status_message(status));
        return 1;
    }

    dtl_context_t ctx;
    status = dtl_environment_make_world_context(env, &ctx);
    printf("Rank %d of %d\n", dtl_context_rank(ctx), dtl_context_size(ctx));

    dtl_context_destroy(ctx);
    dtl_environment_destroy(env);
    return 0;
}
```

### Context Operations

The context encapsulates MPI communicator and device selection. Contexts can be created directly or via environment factory methods (preferred).

```c
// Create context with default options (MPI_COMM_WORLD, no GPU)
dtl_status dtl_context_create_default(dtl_context_t* ctx);

// Create context with options
typedef struct {
    int device_id;       // GPU device ID (-1 for CPU only)
    int init_mpi;        // Whether to initialize MPI (default: 1)
    int finalize_mpi;    // Whether to finalize MPI on destruction (default: 0)
    int reserved[4];     // ABI-stable extension fields
} dtl_context_options;

dtl_status dtl_context_create(dtl_context_t* ctx, const dtl_context_options* opts);

// Destroy context
void dtl_context_destroy(dtl_context_t ctx);

// Query properties
dtl_rank_t dtl_context_rank(dtl_context_t ctx);  // Current rank
dtl_rank_t dtl_context_size(dtl_context_t ctx);  // Total ranks

// Synchronization
dtl_status dtl_context_barrier(dtl_context_t ctx);
```

#### Mode-Aware CUDA/NCCL Context APIs

```c
// Add CUDA/NCCL domains
dtl_status dtl_context_with_cuda(dtl_context_t ctx, int device_id, dtl_context_t* out);
dtl_status dtl_context_with_nccl(dtl_context_t ctx, int device_id, dtl_context_t* out);
dtl_status dtl_context_with_nccl_ex(
    dtl_context_t ctx, int device_id, dtl_nccl_operation_mode mode, dtl_context_t* out);

// Split with NCCL domain
dtl_status dtl_context_split_nccl(dtl_context_t ctx, int color, int key, dtl_context_t* out);
dtl_status dtl_context_split_nccl_ex(
    dtl_context_t ctx, int color, int key, int device_id,
    dtl_nccl_operation_mode mode, dtl_context_t* out);

// NCCL mode/capability introspection
int dtl_context_nccl_mode(dtl_context_t ctx);
int dtl_context_nccl_supports_native(dtl_context_t ctx, dtl_nccl_operation op);
int dtl_context_nccl_supports_hybrid(dtl_context_t ctx, dtl_nccl_operation op);
```

`DTL_NCCL_MODE_NATIVE_ONLY` rejects non-native NCCL operation families.
`DTL_NCCL_MODE_HYBRID_PARITY` enables explicit hybrid parity paths where
available.

**Example:**

```c
dtl_context_t ctx;
dtl_status status = dtl_context_create_default(&ctx);
if (status != DTL_SUCCESS) {
    fprintf(stderr, "Error: %s\n", dtl_status_message(status));
    return 1;
}

printf("Rank %d of %d\n", dtl_context_rank(ctx), dtl_context_size(ctx));
dtl_context_barrier(ctx);
dtl_context_destroy(ctx);
```

### Container Operations

#### Distributed Vector

```c
// Create vector
dtl_status dtl_vector_create(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t global_size,
    dtl_vector_t* vec
);

// Create with fill value
dtl_status dtl_vector_create_fill(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t global_size,
    const void* fill_value,
    dtl_vector_t* vec
);

// Destroy vector
void dtl_vector_destroy(dtl_vector_t vec);

// Size queries
dtl_size_t dtl_vector_global_size(dtl_vector_t vec);
dtl_size_t dtl_vector_local_size(dtl_vector_t vec);
dtl_size_t dtl_vector_local_offset(dtl_vector_t vec);

// Data access (borrowed pointers - do not free)
const void* dtl_vector_local_data(dtl_vector_t vec);
void* dtl_vector_local_data_mut(dtl_vector_t vec);

// Index queries
dtl_rank_t dtl_vector_owner(dtl_vector_t vec, dtl_size_t global_idx);
int dtl_vector_is_local(dtl_vector_t vec, dtl_size_t global_idx);
```

**Example:**

```c
dtl_vector_t vec;
dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 1000, &vec);
if (status != DTL_SUCCESS) {
    fprintf(stderr, "Failed to create vector: %s\n", dtl_status_message(status));
    return 1;
}

// Get local data pointer
double* data = (double*)dtl_vector_local_data_mut(vec);
dtl_size_t local_size = dtl_vector_local_size(vec);

// Fill with values
for (size_t i = 0; i < local_size; i++) {
    data[i] = (double)(dtl_vector_local_offset(vec) + i);
}

dtl_vector_destroy(vec);
```

#### Distributed Span (`dtl_span_t`)

The C ABI exposes a first-class non-owning distributed span handle:

- create from containers: `dtl_span_from_vector`, `dtl_span_from_array`, `dtl_span_from_tensor`
- create from raw local buffer + metadata: `dtl_span_create`
- subspan operations: `dtl_span_first`, `dtl_span_last`, `dtl_span_subspan`
- local access: `dtl_span_data(_mut)`, `dtl_span_get_local`, `dtl_span_set_local`
- metadata: `dtl_span_size`, `dtl_span_local_size`, `dtl_span_rank`, `dtl_span_num_ranks`

`dtl_span_t` is explicitly non-owning. The backing container/storage must outlive every span created from it.

```c
dtl_vector_t vec = NULL;
dtl_span_t span = NULL;
dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 1024, &vec);
if (status != DTL_SUCCESS) return 1;

status = dtl_span_from_vector(vec, &span);
if (status != DTL_SUCCESS) return 1;

double value = 3.14;
status = dtl_span_set_local(span, 0, &value);

dtl_span_destroy(span);
dtl_vector_destroy(vec);
```

#### Distributed Tensor

```c
// Create tensor
dtl_status dtl_tensor_create(
    dtl_context_t ctx,
    dtl_dtype dtype,
    const dtl_size_t* shape,
    int ndim,
    dtl_tensor_t* tensor
);

// Destroy tensor
void dtl_tensor_destroy(dtl_tensor_t tensor);

// Shape queries
int dtl_tensor_ndim(dtl_tensor_t tensor);
void dtl_tensor_shape(dtl_tensor_t tensor, dtl_size_t* shape);
void dtl_tensor_local_shape(dtl_tensor_t tensor, dtl_size_t* shape);
dtl_size_t dtl_tensor_global_size(dtl_tensor_t tensor);
dtl_size_t dtl_tensor_local_size(dtl_tensor_t tensor);

// Data access
const void* dtl_tensor_local_data(dtl_tensor_t tensor);
void* dtl_tensor_local_data_mut(dtl_tensor_t tensor);
```

**Example:**

```c
dtl_size_t shape[] = {100, 64, 64};  // 100x64x64 tensor
dtl_tensor_t tensor;
dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape, 3, &tensor);

float* data = (float*)dtl_tensor_local_data_mut(tensor);
// ... fill data ...

dtl_tensor_destroy(tensor);
```

#### Distributed Array

Fixed-size distributed array (size cannot be changed after creation).

```c
// Create array
dtl_status dtl_array_create(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t size,
    dtl_array_t* arr
);

// Create with fill value
dtl_status dtl_array_create_fill(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t size,
    const void* fill_value,
    dtl_array_t* arr
);

// Destroy array
void dtl_array_destroy(dtl_array_t arr);

// Size queries
dtl_size_t dtl_array_size(dtl_array_t arr);        // Global size (fixed)
dtl_size_t dtl_array_local_size(dtl_array_t arr);
dtl_index_t dtl_array_local_offset(dtl_array_t arr);

// Data access
const void* dtl_array_local_data(dtl_array_t arr);
void* dtl_array_local_data_mut(dtl_array_t arr);

// Index queries
dtl_rank_t dtl_array_owner(dtl_array_t arr, dtl_index_t global_idx);
int dtl_array_is_local(dtl_array_t arr, dtl_index_t global_idx);
```

**Example:**

```c
dtl_array_t arr;
dtl_status status = dtl_array_create(ctx, DTL_DTYPE_INT32, 1000, &arr);

int32_t* data = (int32_t*)dtl_array_local_data_mut(arr);
for (size_t i = 0; i < dtl_array_local_size(arr); i++) {
    data[i] = i;
}

// Note: No resize() method - arrays are fixed size
dtl_array_destroy(arr);
```

### Collective Operations

```c
// Reduction operations
typedef enum dtl_reduce_op {
    DTL_OP_SUM  = 0,
    DTL_OP_PROD = 1,
    DTL_OP_MIN  = 2,
    DTL_OP_MAX  = 3,
    DTL_OP_LAND = 4,  // Logical AND
    DTL_OP_LOR  = 5,  // Logical OR
    DTL_OP_BAND = 6,  // Bitwise AND
    DTL_OP_BOR  = 7   // Bitwise OR
} dtl_reduce_op;

// Broadcast
dtl_status dtl_broadcast(
    dtl_context_t ctx,
    void* buf,
    dtl_size_t count,
    dtl_dtype dtype,
    dtl_rank_t root
);

// Reduce to root
dtl_status dtl_reduce(
    dtl_context_t ctx,
    const void* sendbuf,
    void* recvbuf,
    dtl_size_t count,
    dtl_dtype dtype,
    dtl_reduce_op op,
    dtl_rank_t root
);

// Reduce to all ranks
dtl_status dtl_allreduce(
    dtl_context_t ctx,
    const void* sendbuf,
    void* recvbuf,
    dtl_size_t count,
    dtl_dtype dtype,
    dtl_reduce_op op
);

// Gather to root
dtl_status dtl_gather(
    dtl_context_t ctx,
    const void* sendbuf,
    dtl_size_t sendcount,
    dtl_dtype sendtype,
    void* recvbuf,
    dtl_size_t recvcount,
    dtl_dtype recvtype,
    dtl_rank_t root
);

// Scatter from root
dtl_status dtl_scatter(
    dtl_context_t ctx,
    const void* sendbuf,
    dtl_size_t sendcount,
    dtl_dtype sendtype,
    void* recvbuf,
    dtl_size_t recvcount,
    dtl_dtype recvtype,
    dtl_rank_t root
);

// Gather to all
dtl_status dtl_allgather(
    dtl_context_t ctx,
    const void* sendbuf,
    dtl_size_t sendcount,
    dtl_dtype sendtype,
    void* recvbuf,
    dtl_size_t recvcount,
    dtl_dtype recvtype
);
```

**Example:**

```c
double local_sum = compute_local_sum(vec);
double global_sum;

dtl_status status = dtl_allreduce(
    ctx,
    &local_sum,           // send buffer
    &global_sum,          // receive buffer
    1,                    // count
    DTL_DTYPE_FLOAT64,    // data type
    DTL_OP_SUM            // reduction operation
);

if (status == DTL_SUCCESS) {
    printf("Global sum: %f\n", global_sum);
}
```

### Algorithm Operations

DTL provides distributed algorithm operations on containers.

```c
// Callback types
typedef void (*dtl_unary_func)(void* element, dtl_size_t index, void* user_data);
typedef int (*dtl_predicate)(const void* element, void* user_data);

// for_each - apply function to each local element
dtl_status dtl_for_each_vector(dtl_vector_t vec, dtl_unary_func func, void* user_data);
dtl_status dtl_for_each_array(dtl_array_t arr, dtl_unary_func func, void* user_data);

// copy - copy data between containers
dtl_status dtl_copy_vector(dtl_vector_t src, dtl_vector_t dst);
dtl_status dtl_copy_array(dtl_array_t src, dtl_array_t dst);

// fill - fill container with value
dtl_status dtl_fill_vector(dtl_vector_t vec, const void* value);
dtl_status dtl_fill_array(dtl_array_t arr, const void* value);

// find - find first matching element
dtl_index_t dtl_find_vector(dtl_vector_t vec, const void* value);
dtl_index_t dtl_find_if_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data);
dtl_index_t dtl_find_array(dtl_array_t arr, const void* value);
dtl_index_t dtl_find_if_array(dtl_array_t arr, dtl_predicate pred, void* user_data);

// count - count matching elements
dtl_size_t dtl_count_vector(dtl_vector_t vec, const void* value);
dtl_size_t dtl_count_if_vector(dtl_vector_t vec, dtl_predicate pred, void* user_data);
dtl_size_t dtl_count_array(dtl_array_t arr, const void* value);
dtl_size_t dtl_count_if_array(dtl_array_t arr, dtl_predicate pred, void* user_data);

// reduce - local reduction
dtl_status dtl_reduce_local_vector(dtl_vector_t vec, dtl_reduce_op op, void* result);
dtl_status dtl_reduce_local_array(dtl_array_t arr, dtl_reduce_op op, void* result);

// sort - local sort
dtl_status dtl_sort_vector(dtl_vector_t vec);
dtl_status dtl_sort_vector_descending(dtl_vector_t vec);
dtl_status dtl_sort_array(dtl_array_t arr);
dtl_status dtl_sort_array_descending(dtl_array_t arr);

// minmax - find local min and max
dtl_status dtl_minmax_vector(dtl_vector_t vec, void* min_val, void* max_val);
dtl_status dtl_minmax_array(dtl_array_t arr, void* min_val, void* max_val);
```

**Example:**

```c
// Fill vector with value
double value = 42.0;
dtl_fill_vector(vec, &value);

// Count elements greater than 10
int predicate_gt_10(const void* elem, void* user_data) {
    return *(double*)elem > 10.0;
}
dtl_size_t count = dtl_count_if_vector(vec, predicate_gt_10, NULL);

// Local reduction
double local_sum;
dtl_reduce_local_vector(vec, DTL_OP_SUM, &local_sum);

// Sort ascending
dtl_sort_vector(vec);
```

### Policy Selection

DTL supports policy selection at container creation time.

```c
// Partition policies
typedef enum dtl_partition_policy {
    DTL_PARTITION_BLOCK = 0,
    DTL_PARTITION_CYCLIC = 1,
    DTL_PARTITION_BLOCK_CYCLIC = 2,
    DTL_PARTITION_HASH = 3,
    DTL_PARTITION_REPLICATED = 4
} dtl_partition_policy;

// Placement policies
typedef enum dtl_placement_policy {
    DTL_PLACEMENT_HOST = 0,
    DTL_PLACEMENT_DEVICE = 1,           // CUDA only
    DTL_PLACEMENT_UNIFIED = 2,          // CUDA only
    DTL_PLACEMENT_DEVICE_PREFERRED = 3  // CUDA only
} dtl_placement_policy;

// Execution policies
typedef enum dtl_execution_policy {
    DTL_EXEC_SEQ = 0,
    DTL_EXEC_PAR = 1,
    DTL_EXEC_ASYNC = 2
} dtl_execution_policy;

// Container options
typedef struct dtl_container_options {
    dtl_partition_policy partition;
    dtl_placement_policy placement;
    dtl_execution_policy execution;
} dtl_container_options;

// Initialize options to defaults
void dtl_container_options_init(dtl_container_options* opts);

// Create container with options
dtl_status dtl_vector_create_with_options(
    dtl_context_t ctx,
    dtl_dtype dtype,
    dtl_size_t size,
    const dtl_container_options* opts,
    dtl_vector_t* vec
);

// Query container policies
dtl_partition_policy dtl_vector_partition_policy(dtl_vector_t vec);
dtl_placement_policy dtl_vector_placement_policy(dtl_vector_t vec);

// Check policy availability
int dtl_placement_available(dtl_placement_policy placement);
```

**Example:**

```c
dtl_container_options opts;
dtl_container_options_init(&opts);
opts.partition = DTL_PARTITION_CYCLIC;

dtl_vector_t vec;
dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 10000, &opts, &vec);

// Query policy
dtl_partition_policy policy = dtl_vector_partition_policy(vec);
printf("Partition: %s\n", policy == DTL_PARTITION_CYCLIC ? "cyclic" : "other");
```

### RMA Operations

Remote Memory Access (one-sided communication) operations.

#### Window Management

```c
typedef struct dtl_window_s* dtl_window_t;

typedef enum dtl_lock_mode {
    DTL_LOCK_EXCLUSIVE = 0,
    DTL_LOCK_SHARED = 1
} dtl_lock_mode;

// Create window from existing memory
dtl_status dtl_window_create(
    dtl_context_t ctx,
    void* base,
    dtl_size_t size,
    dtl_window_t* win
);

// Allocate window with new memory
dtl_status dtl_window_allocate(
    dtl_context_t ctx,
    dtl_size_t size,
    dtl_window_t* win
);

// Destroy window
void dtl_window_destroy(dtl_window_t win);

// Query properties
void* dtl_window_base(dtl_window_t win);
dtl_size_t dtl_window_size(dtl_window_t win);
int dtl_window_is_valid(dtl_window_t win);
```

#### Synchronization

```c
// Active-target synchronization (collective)
dtl_status dtl_window_fence(dtl_window_t win);

// Passive-target synchronization (per-rank)
dtl_status dtl_window_lock(dtl_window_t win, dtl_rank_t target, dtl_lock_mode mode);
dtl_status dtl_window_unlock(dtl_window_t win, dtl_rank_t target);
dtl_status dtl_window_lock_all(dtl_window_t win);
dtl_status dtl_window_unlock_all(dtl_window_t win);

// Flush pending operations
dtl_status dtl_window_flush(dtl_window_t win, dtl_rank_t target);
dtl_status dtl_window_flush_all(dtl_window_t win);
dtl_status dtl_window_flush_local(dtl_window_t win, dtl_rank_t target);
dtl_status dtl_window_flush_local_all(dtl_window_t win);
```

#### Data Transfer

```c
// Put data to remote window
dtl_status dtl_rma_put(
    dtl_window_t win,
    dtl_rank_t target,
    dtl_size_t target_offset,
    const void* origin,
    dtl_size_t size
);

// Get data from remote window
dtl_status dtl_rma_get(
    dtl_window_t win,
    dtl_rank_t target,
    dtl_size_t target_offset,
    void* buffer,
    dtl_size_t size
);

// Async versions
dtl_status dtl_rma_put_async(dtl_window_t win, dtl_rank_t target,
                              dtl_size_t offset, const void* data,
                              dtl_size_t size, dtl_request_t* req);
dtl_status dtl_rma_get_async(dtl_window_t win, dtl_rank_t target,
                              dtl_size_t offset, void* buffer,
                              dtl_size_t size, dtl_request_t* req);
```

#### Atomic Operations

```c
// Atomic accumulate
dtl_status dtl_rma_accumulate(
    dtl_window_t win,
    dtl_rank_t target,
    dtl_size_t offset,
    const void* origin,
    dtl_size_t size,
    dtl_dtype dtype,
    dtl_reduce_op op
);

// Atomic fetch-and-op
dtl_status dtl_rma_fetch_and_op(
    dtl_window_t win,
    dtl_rank_t target,
    dtl_size_t offset,
    const void* origin,
    void* result,
    dtl_dtype dtype,
    dtl_reduce_op op
);

// Atomic compare-and-swap
dtl_status dtl_rma_compare_and_swap(
    dtl_window_t win,
    dtl_rank_t target,
    dtl_size_t offset,
    const void* compare,
    const void* swap,
    void* result,
    dtl_dtype dtype
);
```

**Example:**

```c
// Create window
dtl_window_t win;
dtl_window_allocate(ctx, 1024, &win);

// Active-target: use fence
dtl_window_fence(win);  // Start epoch

double data[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
dtl_rank_t my_rank = dtl_context_rank(ctx);
dtl_rma_put(win, my_rank, 0, data, sizeof(data));

dtl_window_fence(win);  // Complete epoch

// Passive-target: use lock/unlock
dtl_window_lock(win, my_rank, DTL_LOCK_EXCLUSIVE);

int64_t old_val;
int64_t addend = 1;
dtl_rma_fetch_and_op(win, my_rank, 0, &addend, &old_val,
                      DTL_DTYPE_INT64, DTL_OP_SUM);
dtl_window_flush(win, my_rank);

dtl_window_unlock(win, my_rank);

// Cleanup
dtl_window_destroy(win);
```

---

## Error Handling

### Status Codes

All functions that can fail return `dtl_status`:

```c
#define DTL_SUCCESS                    0

// Communication (100-199)
#define DTL_ERROR_COMMUNICATION      100
#define DTL_ERROR_SEND_FAILED        101
#define DTL_ERROR_RECV_FAILED        102
#define DTL_ERROR_BARRIER_FAILED     105
#define DTL_ERROR_TIMEOUT            106

// Memory (200-299)
#define DTL_ERROR_MEMORY             200
#define DTL_ERROR_ALLOCATION_FAILED  201
#define DTL_ERROR_OUT_OF_MEMORY      202

// Bounds (400-499)
#define DTL_ERROR_BOUNDS             400
#define DTL_ERROR_OUT_OF_BOUNDS      401
#define DTL_ERROR_INVALID_ARGUMENT   410

// Backend (500-599)
#define DTL_ERROR_BACKEND            500
#define DTL_ERROR_MPI                530

// Internal (900-999)
#define DTL_ERROR_NOT_IMPLEMENTED    901
```

### Error Messages

```c
// Get human-readable error message
const char* dtl_status_message(dtl_status status);

// Get error category name
const char* dtl_status_category(dtl_status status);

// Get error category code
int dtl_status_category_code(dtl_status status);
```

### Error Handling Pattern

```c
dtl_status status = dtl_some_operation(...);
if (status != DTL_SUCCESS) {
    fprintf(stderr, "[%s] Error %d: %s\n",
            dtl_status_category(status),
            status,
            dtl_status_message(status));
    // Handle error...
}
```

---

## Thread Safety

| Category | Guarantee |
|----------|-----------|
| Version queries | Thread-safe |
| Status messages | Thread-safe (static strings) |
| Context operations | NOT thread-safe within same context |
| Container operations | NOT thread-safe within same container |

**Multi-threaded usage**: Different contexts/containers MAY be used from different threads simultaneously. The same context/container MUST NOT be accessed from multiple threads without external synchronization.

---

## Building FFI for Other Languages

The C ABI is designed to be easily wrapped by other languages:

### Symbol Naming

All symbols use the `dtl_` prefix with predictable patterns:
- Types: `dtl_<name>_t`
- Functions: `dtl_<type>_<action>`
- Constants: `DTL_<NAME>`

### Handle Pattern

Opaque handles are pointers to forward-declared structs:

```c
// In header
typedef struct dtl_context_s* dtl_context_t;

// Actual implementation hidden in .cpp
struct dtl_context_s {
    // ... implementation details ...
};
```

This ensures:
- ABI stability (pointer size is fixed)
- No need to expose internal structure layout
- Safe to change internals without recompilation

### Feature Detection

```c
// Check available backends at runtime
int dtl_has_mpi(void);   // Returns 1 if MPI available
int dtl_has_cuda(void);  // Returns 1 if CUDA available
int dtl_has_hip(void);   // Returns 1 if HIP available
int dtl_has_nccl(void);  // Returns 1 if NCCL available
```

### Version Queries

```c
int dtl_version_major(void);
int dtl_version_minor(void);
int dtl_version_patch(void);
int dtl_abi_version(void);
const char* dtl_version_string(void);  // e.g., "1.0.0"
```

---

## Complete Example

```c
/**
 * DTL C Bindings Example: Distributed Sum
 *
 * Compile: gcc -std=c99 -o dist_sum dist_sum.c -ldtl_c -lmpi
 * Run: mpirun -np 4 ./dist_sum
 */

#include <dtl/bindings/c/dtl.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    dtl_context_t ctx;
    dtl_vector_t vec;
    dtl_status status;

    // Create context
    status = dtl_context_create_default(&ctx);
    if (status != DTL_SUCCESS) {
        fprintf(stderr, "Failed to create context: %s\n",
                dtl_status_message(status));
        return 1;
    }

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);
    printf("[Rank %d/%d] Started\n", rank, size);

    // Create distributed vector with 10000 elements
    const dtl_size_t global_size = 10000;
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, &vec);
    if (status != DTL_SUCCESS) {
        fprintf(stderr, "Failed to create vector: %s\n",
                dtl_status_message(status));
        dtl_context_destroy(ctx);
        return 1;
    }

    // Get local data pointer
    double* data = (double*)dtl_vector_local_data_mut(vec);
    dtl_size_t local_size = dtl_vector_local_size(vec);
    dtl_size_t local_offset = dtl_vector_local_offset(vec);

    printf("[Rank %d] Local size: %lu, offset: %lu\n",
           rank, (unsigned long)local_size, (unsigned long)local_offset);

    // Initialize with global indices
    for (dtl_size_t i = 0; i < local_size; i++) {
        data[i] = (double)(local_offset + i);
    }

    // Compute local sum
    double local_sum = 0.0;
    for (dtl_size_t i = 0; i < local_size; i++) {
        local_sum += data[i];
    }

    // Global sum via allreduce
    double global_sum;
    status = dtl_allreduce(ctx, &local_sum, &global_sum, 1,
                           DTL_DTYPE_FLOAT64, DTL_OP_SUM);
    if (status != DTL_SUCCESS) {
        fprintf(stderr, "Allreduce failed: %s\n", dtl_status_message(status));
        dtl_vector_destroy(vec);
        dtl_context_destroy(ctx);
        return 1;
    }

    // Verify result (sum of 0..N-1 = N*(N-1)/2)
    double expected = (double)(global_size * (global_size - 1) / 2);

    if (rank == 0) {
        printf("\nGlobal sum: %.0f\n", global_sum);
        printf("Expected:   %.0f\n", expected);
        printf("Result: %s\n",
               (global_sum == expected) ? "SUCCESS" : "FAILURE");
    }

    // Cleanup
    dtl_vector_destroy(vec);
    dtl_context_destroy(ctx);

    return (global_sum == expected) ? 0 : 1;
}
```

---

## References

- [Python Bindings Guide](python_bindings.md) (uses C ABI internally)
- [Fortran Bindings Guide](fortran_bindings.md) (via ISO_C_BINDING)

## Thread Safety

A `dtl_context_t` handle is **not safe for concurrent use** from multiple
threads. However, multi-threaded programs can use DTL safely by following
these patterns:

### Pattern 1: One context per thread

```c
void* worker(void* arg) {
    int thread_id = *(int*)arg;

    // Each thread creates its own context
    dtl_context_t ctx;
    dtl_context_create_default(&ctx);

    // Create and work with containers independently
    dtl_vector_t vec;
    dtl_vector_create_f64(&vec, ctx, 1000);

    // ... compute ...

    dtl_vector_destroy(vec);
    dtl_context_destroy(ctx);
    return NULL;
}
```

### Pattern 2: Duplicate context for threads

```c
// Main thread creates the primary context
dtl_context_t main_ctx;
dtl_context_create_default(&main_ctx);

// Before spawning threads, duplicate for each
dtl_context_t thread_ctx;
dtl_context_dup(main_ctx, &thread_ctx);

// Pass thread_ctx to the worker thread
// Each thread uses its own duplicated context
```

### Pattern 3: External synchronization

```c
pthread_mutex_t ctx_lock = PTHREAD_MUTEX_INITIALIZER;
dtl_context_t shared_ctx;

void* worker(void* arg) {
    // Serialize access to shared context
    pthread_mutex_lock(&ctx_lock);
    dtl_context_barrier(shared_ctx);
    pthread_mutex_unlock(&ctx_lock);
    return NULL;
}
```

### Recommended approach

**Pattern 1** (one context per thread) is recommended for most use cases.
It provides complete isolation with no synchronization overhead. Each
context duplicates the MPI communicator internally, ensuring message
isolation between threads.

For GPU applications with multiple CUDA streams, use Pattern 2 and set
different device IDs for each context:

```c
dtl_context_options opts;
dtl_context_options_init(&opts);
opts.device_id = thread_id % num_gpus;

dtl_context_t ctx;
dtl_context_create(&ctx, &opts);
```
