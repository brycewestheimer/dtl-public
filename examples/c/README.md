# DTL C Examples

This directory contains example C programs demonstrating the DTL C bindings.

## Prerequisites

1. Build DTL with C bindings:
   ```bash
   cmake .. -DDTL_BUILD_C_BINDINGS=ON
   make
   ```

2. Set library path:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/dtl/build/src/bindings/c
   ```

## Examples

### hello_dtl.c
Basic introduction to DTL C bindings:
- Version and feature detection
- Creating a context
- Querying rank and size

```bash
gcc -I../../include -L../../build/src/bindings/c hello_dtl.c -ldtl_c -o hello_dtl
./hello_dtl
```

### distributed_vector.c
Working with distributed vectors:
- Creating vectors with different dtypes
- Accessing local data
- Fill operations
- Computing local statistics

```bash
gcc -I../../include -L../../build/src/bindings/c distributed_vector.c -ldtl_c -lm -o distributed_vector
./distributed_vector
mpirun -np 4 ./distributed_vector
```

### distributed_tensor.c
Working with multi-dimensional distributed arrays:
- Creating 2D and 3D tensors
- Understanding tensor partitioning
- N-D indexing operations

```bash
gcc -I../../include -L../../build/src/bindings/c distributed_tensor.c -ldtl_c -o distributed_tensor
./distributed_tensor
mpirun -np 4 ./distributed_tensor
```

### collective_ops.c
Collective communication operations:
- Broadcast
- Reduce and Allreduce
- Gather and Scatter
- Allgather

```bash
mpicc -I../../include -L../../build/src/bindings/c collective_ops.c -ldtl_c -o collective_ops
mpirun -np 4 ./collective_ops
```

### nccl_modes.c
Mode-aware NCCL context composition and capability queries:
- `DTL_NCCL_MODE_NATIVE_ONLY` vs `DTL_NCCL_MODE_HYBRID_PARITY`
- `dtl_context_with_nccl_ex` / `dtl_context_split_nccl_ex`
- native/hybrid capability checks per operation family

```bash
mpicc -I../../include -L../../build/src/bindings/c nccl_modes.c -ldtl_c -o nccl_modes
mpirun -np 2 ./nccl_modes
```

## API Summary

### Context Management
```c
dtl_status dtl_context_create_default(dtl_context_t* ctx);
dtl_status dtl_context_create(dtl_context_t* ctx, const dtl_context_options* opts);
void dtl_context_destroy(dtl_context_t ctx);
dtl_rank_t dtl_context_rank(dtl_context_t ctx);
dtl_rank_t dtl_context_size(dtl_context_t ctx);
```

### Distributed Vector
```c
dtl_status dtl_vector_create(dtl_context_t ctx, dtl_dtype dtype, dtl_size_t size, dtl_vector_t* vec);
void dtl_vector_destroy(dtl_vector_t vec);
dtl_size_t dtl_vector_global_size(dtl_vector_t vec);
dtl_size_t dtl_vector_local_size(dtl_vector_t vec);
void* dtl_vector_local_data_mut(dtl_vector_t vec);
dtl_status dtl_vector_fill(dtl_vector_t vec, const void* value);
```

### Distributed Tensor
```c
dtl_status dtl_tensor_create(dtl_context_t ctx, dtl_dtype dtype, const dtl_shape* shape, dtl_tensor_t* tensor);
void dtl_tensor_destroy(dtl_tensor_t tensor);
void dtl_tensor_shape(dtl_tensor_t tensor, dtl_shape* shape);
void dtl_tensor_local_shape(dtl_tensor_t tensor, dtl_shape* shape);
void* dtl_tensor_local_data_mut(dtl_tensor_t tensor);
```

### Collective Operations
```c
dtl_status dtl_barrier(dtl_context_t ctx);
dtl_status dtl_broadcast(dtl_context_t ctx, void* data, dtl_size_t count, dtl_dtype dtype, dtl_rank_t root);
dtl_status dtl_reduce(dtl_context_t ctx, const void* send, void* recv, dtl_size_t count, dtl_dtype dtype, dtl_reduce_op op, dtl_rank_t root);
dtl_status dtl_allreduce(dtl_context_t ctx, const void* send, void* recv, dtl_size_t count, dtl_dtype dtype, dtl_reduce_op op);
```

### NCCL Context/Mode
```c
dtl_status dtl_context_with_nccl_ex(dtl_context_t ctx, int device_id,
                                    dtl_nccl_operation_mode mode, dtl_context_t* out);
dtl_status dtl_context_split_nccl_ex(dtl_context_t ctx, int color, int key,
                                     int device_id, dtl_nccl_operation_mode mode,
                                     dtl_context_t* out);
int dtl_context_nccl_supports_native(dtl_context_t ctx, dtl_nccl_operation op);
int dtl_context_nccl_supports_hybrid(dtl_context_t ctx, dtl_nccl_operation op);
```

## Error Handling

All DTL functions return a `dtl_status`. Use these functions to check and handle errors:

```c
int dtl_status_ok(dtl_status status);          /* Returns true if success */
int dtl_status_is_error(dtl_status status);    /* Returns true if error */
const char* dtl_status_message(dtl_status status);  /* Human-readable message */
const char* dtl_status_name(dtl_status status);     /* Status code name */
```

## Supported Data Types

| DTL Dtype | C Type |
|-----------|--------|
| `DTL_DTYPE_INT8` | `int8_t` |
| `DTL_DTYPE_INT16` | `int16_t` |
| `DTL_DTYPE_INT32` | `int32_t` |
| `DTL_DTYPE_INT64` | `int64_t` |
| `DTL_DTYPE_UINT8` | `uint8_t` |
| `DTL_DTYPE_UINT16` | `uint16_t` |
| `DTL_DTYPE_UINT32` | `uint32_t` |
| `DTL_DTYPE_UINT64` | `uint64_t` |
| `DTL_DTYPE_FLOAT32` | `float` |
| `DTL_DTYPE_FLOAT64` | `double` |
