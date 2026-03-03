# C API Reference

**DTL Version:** 0.1.0-alpha.1
**Last Updated:** 2026-02-25

The DTL C bindings provide a C-compatible interface to DTL's core functionality. All functions use opaque handle types, `dtl_status` return codes, and callback-based algorithms.

**Master Header:** `#include <dtl/bindings/c/dtl.h>`

---

## Conventions

- All functions return `dtl_status` (integer) unless stated otherwise
- `DTL_SUCCESS` (0) indicates success; non-zero values are error codes
- Opaque handles (`dtl_context_t`, `dtl_vector_t`, etc.) are pointer-sized types
- Callback functions receive a `void* user_data` for user context
- `NULL` handles are safe to pass to destroy functions (no-op)
- All collective operations require all ranks to participate
- `dtl_version_string()` returns the full public release string (`0.1.0-alpha.1`)
- Unless documented otherwise, destroy functions own and release the underlying handle
- Host-only builds are supported; MPI-dependent operations must fail explicitly when MPI is unavailable

---

## Context Functions

**Header:** `#include <dtl/bindings/c/dtl_context.h>`

### `dtl_context_create`

Create a context with options.

```c
dtl_status dtl_context_create(dtl_context_t* ctx, const dtl_context_options* opts);
```

**Parameters:**
- `ctx` — Pointer to receive the context handle (must not be NULL)
- `opts` — Configuration options (NULL for defaults)

**Example:**
```c
dtl_context_t ctx;
dtl_context_options opts;
dtl_context_options_init(&opts);
opts.device_id = 0;  // Use GPU 0

dtl_status status = dtl_context_create(&ctx, &opts);
if (status != DTL_SUCCESS) {
    fprintf(stderr, "Failed: %s\n", dtl_status_message(status));
    return 1;
}
```

### `dtl_context_create_default`

Create a context with default options.

In this alpha release, callers should not assume MPI is auto-initialized on
their behalf. In MPI builds, create the context only after MPI is available in
the process and treat collective operations as collective over the active
communicator.

```c
dtl_status dtl_context_create_default(dtl_context_t* ctx);
```

### `dtl_context_destroy`

Destroy a context and release resources.

```c
void dtl_context_destroy(dtl_context_t ctx);
```

Safe to call with NULL.

### `dtl_context_rank` / `dtl_context_size`

Query rank and world size.

```c
dtl_rank_t dtl_context_rank(dtl_context_t ctx);
dtl_rank_t dtl_context_size(dtl_context_t ctx);
```

### `dtl_context_barrier`

Collective barrier synchronization.

```c
dtl_status dtl_context_barrier(dtl_context_t ctx);
```

### `dtl_context_options_init`

Initialize options struct with defaults.

```c
void dtl_context_options_init(dtl_context_options* opts);
```

**Full example:**
```c
#include <dtl/bindings/c/dtl.h>
#include <stdio.h>

int main(int argc, char** argv) {
    dtl_context_t ctx;
    dtl_status status = dtl_context_create_default(&ctx);
    if (status != DTL_SUCCESS) return 1;

    printf("Rank %d of %d\n", dtl_context_rank(ctx), dtl_context_size(ctx));

    dtl_context_destroy(ctx);
    return 0;
}
```

---

## Container Functions — Vector

**Header:** `#include <dtl/bindings/c/dtl_vector.h>`

### `dtl_vector_create`

Create a distributed vector with specified dtype and global size.

```c
dtl_status dtl_vector_create(dtl_context_t ctx, dtl_dtype dtype,
                              dtl_size_t global_size, dtl_vector_t* vec);
```

**Supported dtypes:** `DTL_DTYPE_INT8`, `DTL_DTYPE_INT32`, `DTL_DTYPE_INT64`, `DTL_DTYPE_UINT8`, `DTL_DTYPE_UINT32`, `DTL_DTYPE_UINT64`, `DTL_DTYPE_FLOAT32`, `DTL_DTYPE_FLOAT64`

### `dtl_vector_create_fill`

Create a vector initialized with a value.

```c
dtl_status dtl_vector_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                   dtl_size_t global_size, const void* value,
                                   dtl_vector_t* vec);
```

### `dtl_vector_destroy`

Destroy a vector and free its memory.

```c
void dtl_vector_destroy(dtl_vector_t vec);
```

### Size Queries

```c
dtl_size_t dtl_vector_global_size(dtl_vector_t vec);
dtl_size_t dtl_vector_local_size(dtl_vector_t vec);
```

### Data Access

```c
const void* dtl_vector_local_data(dtl_vector_t vec);
void*       dtl_vector_local_data_mut(dtl_vector_t vec);
```

**Example:**
```c
dtl_vector_t vec;
dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 1000, &vec);

double* data = (double*)dtl_vector_local_data(vec);
dtl_size_t local_n = dtl_vector_local_size(vec);
for (dtl_size_t i = 0; i < local_n; i++) {
    data[i] = (double)i;
}

dtl_vector_destroy(vec);
```

---

## Container Functions — Array

**Header:** `#include <dtl/bindings/c/dtl_array.h>`

Arrays have the same interface as vectors but cannot be resized.

```c
dtl_status dtl_array_create(dtl_context_t ctx, dtl_dtype dtype,
                             dtl_size_t global_size, dtl_array_t* arr);
dtl_status dtl_array_create_fill(dtl_context_t ctx, dtl_dtype dtype,
                                  dtl_size_t global_size, const void* value,
                                  dtl_array_t* arr);
void       dtl_array_destroy(dtl_array_t arr);

dtl_size_t dtl_array_global_size(dtl_array_t arr);
dtl_size_t dtl_array_local_size(dtl_array_t arr);
void*      dtl_array_local_data(dtl_array_t arr);
```

---

## Container Functions — Span

**Header:** `#include <dtl/bindings/c/dtl_span.h>`

```c
dtl_status dtl_span_from_vector(dtl_vector_t vec, dtl_span_t* span);
dtl_status dtl_span_from_array(dtl_array_t arr, dtl_span_t* span);
dtl_status dtl_span_from_tensor(dtl_tensor_t tensor, dtl_span_t* span);
void       dtl_span_destroy(dtl_span_t span);

dtl_size_t dtl_span_size(dtl_span_t span);
dtl_size_t dtl_span_local_size(dtl_span_t span);
dtl_size_t dtl_span_size_bytes(dtl_span_t span);
dtl_rank_t dtl_span_rank(dtl_span_t span);
dtl_rank_t dtl_span_num_ranks(dtl_span_t span);

const void* dtl_span_data(dtl_span_t span);
void*       dtl_span_data_mut(dtl_span_t span);

dtl_status dtl_span_first(dtl_span_t span, dtl_size_t count, dtl_span_t* out_span);
dtl_status dtl_span_last(dtl_span_t span, dtl_size_t count, dtl_span_t* out_span);
dtl_status dtl_span_subspan(dtl_span_t span, dtl_size_t offset, dtl_size_t count,
                            dtl_span_t* out_span);
```

`dtl_span_t` is a non-owning view. The backing container/storage must outlive all derived span handles.

---

## Container Functions — Tensor

**Header:** `#include <dtl/bindings/c/dtl_tensor.h>`

```c
dtl_status dtl_tensor_create(dtl_context_t ctx, dtl_dtype dtype,
                              dtl_size_t ndim, const dtl_size_t* shape,
                              dtl_tensor_t* tensor);
void       dtl_tensor_destroy(dtl_tensor_t tensor);

dtl_size_t dtl_tensor_ndim(dtl_tensor_t tensor);
dtl_size_t dtl_tensor_global_size(dtl_tensor_t tensor);
dtl_size_t dtl_tensor_local_size(dtl_tensor_t tensor);
void*      dtl_tensor_local_data(dtl_tensor_t tensor);
```

---

## Container Functions — Map

**Header:** `#include <dtl/bindings/c/dtl_map.h>`

```c
dtl_status dtl_map_create(dtl_context_t ctx, dtl_dtype key_dtype,
                           dtl_dtype value_dtype, dtl_map_t* map);
void       dtl_map_destroy(dtl_map_t map);

dtl_status dtl_map_insert(dtl_map_t map, const void* key, const void* value);
dtl_status dtl_map_find(dtl_map_t map, const void* key, void* value, int* found);
dtl_status dtl_map_erase(dtl_map_t map, const void* key);
dtl_size_t dtl_map_local_size(dtl_map_t map);
```

---

## Algorithm Functions

**Header:** `#include <dtl/bindings/c/dtl_algorithms.h>`

### Callback Types

```c
// Mutable element callback
typedef void (*dtl_unary_func)(void* element, dtl_size_t index, void* user_data);

// Const element callback
typedef void (*dtl_const_unary_func)(const void* element, dtl_size_t index, void* user_data);

// Transform callback
typedef void (*dtl_transform_func)(const void* input, void* output,
                                    dtl_size_t index, void* user_data);

// Predicate callback
typedef int (*dtl_predicate)(const void* element, void* user_data);

// Comparator callback
typedef int (*dtl_comparator)(const void* a, const void* b, void* user_data);

// Binary reduction callback
typedef void (*dtl_binary_func)(const void* a, const void* b,
                                 void* result, void* user_data);
```

### For-Each

```c
dtl_status dtl_for_each_vector(dtl_vector_t vec, dtl_unary_func func, void* user_data);
dtl_status dtl_for_each_vector_const(dtl_vector_t vec, dtl_const_unary_func func, void* user_data);
dtl_status dtl_for_each_array(dtl_array_t arr, dtl_unary_func func, void* user_data);
dtl_status dtl_for_each_array_const(dtl_array_t arr, dtl_const_unary_func func, void* user_data);
```

**Example:**
```c
void double_element(void* elem, dtl_size_t idx, void* user_data) {
    double* val = (double*)elem;
    *val *= 2.0;
}

dtl_for_each_vector(vec, double_element, NULL);
```

### Transform

```c
dtl_status dtl_transform_vector(dtl_vector_t src, dtl_vector_t dst,
                                 dtl_transform_func func, void* user_data);
dtl_status dtl_transform_array(dtl_array_t src, dtl_array_t dst,
                                dtl_transform_func func, void* user_data);
```

### Copy / Fill

```c
dtl_status dtl_copy_vector(dtl_vector_t src, dtl_vector_t dst);
dtl_status dtl_copy_array(dtl_array_t src, dtl_array_t dst);
dtl_status dtl_fill_vector(dtl_vector_t vec, const void* value);
dtl_status dtl_fill_array(dtl_array_t arr, const void* value);
```

### Find

```c
dtl_status dtl_find_vector(dtl_vector_t vec, const void* value, dtl_size_t* index);
dtl_status dtl_find_if_vector(dtl_vector_t vec, dtl_predicate pred,
                               void* user_data, dtl_size_t* index);
dtl_status dtl_find_array(dtl_array_t arr, const void* value, dtl_size_t* index);
```

### Count

```c
dtl_status dtl_count_vector(dtl_vector_t vec, const void* value, dtl_size_t* count);
dtl_status dtl_count_if_vector(dtl_vector_t vec, dtl_predicate pred,
                                void* user_data, dtl_size_t* count);
```

### Reduce

```c
dtl_status dtl_reduce_vector(dtl_vector_t vec, const void* init,
                              dtl_binary_func func, void* result, void* user_data);
dtl_status dtl_reduce_array(dtl_array_t arr, const void* init,
                             dtl_binary_func func, void* result, void* user_data);
```

### Sort

```c
dtl_status dtl_sort_vector(dtl_vector_t vec, dtl_comparator comp, void* user_data);
dtl_status dtl_sort_array(dtl_array_t arr, dtl_comparator comp, void* user_data);
```

**Example:**
```c
int compare_doubles(const void* a, const void* b, void* user_data) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

dtl_sort_vector(vec, compare_doubles, NULL);
```

---

## Communication Functions

**Header:** `#include <dtl/bindings/c/dtl_communicator.h>`

### Point-to-Point (Blocking)

```c
dtl_status dtl_send(dtl_context_t ctx, const void* buf, dtl_size_t count,
                     dtl_dtype dtype, dtl_rank_t dest, dtl_tag_t tag);

dtl_status dtl_recv(dtl_context_t ctx, void* buf, dtl_size_t count,
                     dtl_dtype dtype, dtl_rank_t source, dtl_tag_t tag);

dtl_status dtl_sendrecv(dtl_context_t ctx,
                         const void* sendbuf, dtl_size_t sendcount,
                         dtl_dtype senddtype, dtl_rank_t dest, dtl_tag_t sendtag,
                         void* recvbuf, dtl_size_t recvcount,
                         dtl_dtype recvdtype, dtl_rank_t source, dtl_tag_t recvtag);
```

### Point-to-Point (Non-Blocking)

```c
dtl_status dtl_isend(dtl_context_t ctx, const void* buf, dtl_size_t count,
                      dtl_dtype dtype, dtl_rank_t dest, dtl_tag_t tag,
                      dtl_request_t* request);

dtl_status dtl_irecv(dtl_context_t ctx, void* buf, dtl_size_t count,
                      dtl_dtype dtype, dtl_rank_t source, dtl_tag_t tag,
                      dtl_request_t* request);

dtl_status dtl_wait(dtl_request_t request);
dtl_status dtl_waitall(dtl_size_t count, dtl_request_t* requests);
dtl_status dtl_test(dtl_request_t request, int* completed);
void       dtl_request_free(dtl_request_t request);
```

### Barrier

```c
dtl_status dtl_barrier(dtl_context_t ctx);
```

### Broadcast

```c
dtl_status dtl_broadcast(dtl_context_t ctx, void* buf, dtl_size_t count,
                          dtl_dtype dtype, dtl_rank_t root);
```

### Reduce / Allreduce

```c
dtl_status dtl_reduce(dtl_context_t ctx,
                       const void* sendbuf, void* recvbuf,
                       dtl_size_t count, dtl_dtype dtype,
                       dtl_reduce_op op, dtl_rank_t root);

dtl_status dtl_allreduce(dtl_context_t ctx,
                          const void* sendbuf, void* recvbuf,
                          dtl_size_t count, dtl_dtype dtype,
                          dtl_reduce_op op);
```

**Reduction operations:** `DTL_REDUCE_SUM`, `DTL_REDUCE_PROD`, `DTL_REDUCE_MIN`, `DTL_REDUCE_MAX`, `DTL_REDUCE_LAND`, `DTL_REDUCE_LOR`, `DTL_REDUCE_BAND`, `DTL_REDUCE_BOR`

### Gather / Scatter

```c
dtl_status dtl_gather(dtl_context_t ctx,
                       const void* sendbuf, dtl_size_t sendcount, dtl_dtype senddtype,
                       void* recvbuf, dtl_size_t recvcount, dtl_dtype recvdtype,
                       dtl_rank_t root);

dtl_status dtl_scatter(dtl_context_t ctx,
                        const void* sendbuf, dtl_size_t sendcount, dtl_dtype senddtype,
                        void* recvbuf, dtl_size_t recvcount, dtl_dtype recvdtype,
                        dtl_rank_t root);

dtl_status dtl_allgather(dtl_context_t ctx,
                          const void* sendbuf, dtl_size_t sendcount, dtl_dtype senddtype,
                          void* recvbuf, dtl_size_t recvcount, dtl_dtype recvdtype);
```

---

## Status Codes

**Header:** `#include <dtl/bindings/c/dtl_status.h>`

| Constant | Value | Description |
|----------|-------|-------------|
| `DTL_SUCCESS` | 0 | Operation succeeded |
| `DTL_ERROR_COMMUNICATION` | 100 | Communication failure |
| `DTL_ERROR_MEMORY` | 200 | Memory allocation failure |
| `DTL_ERROR_BOUNDS` | 400 | Index out of bounds |
| `DTL_ERROR_BACKEND` | 500 | Backend error |
| `DTL_ERROR_INVALID_ARGUMENT` | 410 | Invalid argument |
| `DTL_ERROR_NULL_POINTER` | 411 | Null pointer |
| `DTL_ERROR_NOT_SUPPORTED` | 420 | Operation not supported |

```c
const char* dtl_status_message(dtl_status status);
int         dtl_status_is_error(dtl_status status);
```

---

## Data Types

**Header:** `#include <dtl/bindings/c/dtl_types.h>`

```c
typedef int64_t  dtl_size_t;     // Size type
typedef int32_t  dtl_rank_t;     // Rank identifier
typedef int32_t  dtl_tag_t;      // Message tag

// Opaque handles
typedef struct dtl_context_s*  dtl_context_t;
typedef struct dtl_vector_s*   dtl_vector_t;
typedef struct dtl_array_s*    dtl_array_t;
typedef struct dtl_tensor_s*   dtl_tensor_t;
typedef struct dtl_map_s*      dtl_map_t;
typedef struct dtl_request_s*  dtl_request_t;
```

---

## See Also

- [C++ Quick Reference](cpp_quick_reference.md)
- [Python API Reference](python_api_reference.md)
