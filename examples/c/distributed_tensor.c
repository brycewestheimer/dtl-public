// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file distributed_tensor.c
 * @brief Distributed tensor operations with DTL C bindings
 *
 * Demonstrates:
 * - Creating distributed tensors (multi-dimensional)
 * - Understanding tensor partitioning
 * - N-D indexing operations
 *
 * Compile with:
 *   gcc -I../../include -L../../build/src/bindings/c distributed_tensor.c -ldtl_c -o distributed_tensor
 *
 * Run with:
 *   ./distributed_tensor
 *   mpirun -np 4 ./distributed_tensor
 */

#include <stdio.h>
#include <stdlib.h>
#include <dtl/bindings/c/dtl.h>

#define CHECK_STATUS(status, msg) \
    do { \
        if (!dtl_status_ok(status)) { \
            fprintf(stderr, "%s: %s\n", msg, dtl_status_message(status)); \
            exit(1); \
        } \
    } while(0)

void print_shape(const char* label, dtl_shape shape) {
    printf("%s: (", label);
    for (int i = 0; i < shape.ndim; i++) {
        if (i > 0) printf(", ");
        printf("%lu", (unsigned long)shape.dims[i]);
    }
    printf(")\n");
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    dtl_context_t ctx = NULL;
    dtl_tensor_t tensor = NULL;
    dtl_status status;

    /* Create context */
    status = dtl_context_create_default(&ctx);
    CHECK_STATUS(status, "Failed to create context");

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);

    printf("Rank %d of %d: Distributed Tensor Example\n", rank, size);
    printf("============================================\n");

    /* Create a 2D tensor (100 x 64) */
    dtl_shape shape = dtl_shape_2d(100, 64);
    status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shape, &tensor);
    CHECK_STATUS(status, "Failed to create tensor");

    /* Query tensor properties */
    dtl_shape global_shape = dtl_tensor_shape(tensor);
    dtl_shape local_shape = dtl_tensor_local_shape(tensor);

    printf("Rank %d:\n", rank);
    print_shape("  Global shape", global_shape);
    print_shape("  Local shape", local_shape);
    printf("  Global size: %lu\n", (unsigned long)dtl_tensor_global_size(tensor));
    printf("  Local size: %lu\n", (unsigned long)dtl_tensor_local_size(tensor));
    printf("  Distributed dim: %d\n", dtl_tensor_distributed_dim(tensor));

    /* Get local data */
    double* data = (double*)dtl_tensor_local_data_mut(tensor);

    /* Row-major strides */
    dtl_index_t stride0 = (dtl_index_t)local_shape.dims[1];
    dtl_index_t stride1 = 1;

    printf("  Strides: [%ld, %ld]\n", (long)stride0, (long)stride1);

    /* Initialize: each element is row*100 + col */
    for (dtl_size_t i = 0; i < local_shape.dims[0]; i++) {
        for (dtl_size_t j = 0; j < local_shape.dims[1]; j++) {
            /* Row-major indexing */
            data[i * stride0 + j * stride1] = (double)(i * 100 + j);
        }
    }

    /* Use N-D indexing API */
    dtl_index_t indices[2] = {0, 0};
    double value = 0.0;

    status = dtl_tensor_get_local_nd(tensor, indices, &value);
    CHECK_STATUS(status, "Get failed");
    printf("  Value at [0,0]: %.1f\n", value);

    /* Set a value */
    double new_value = 999.0;
    status = dtl_tensor_set_local_nd(tensor, indices, &new_value);
    CHECK_STATUS(status, "Set failed");

    status = dtl_tensor_get_local_nd(tensor, indices, &value);
    CHECK_STATUS(status, "Get failed");
    printf("  Value at [0,0] after set: %.1f\n", value);

    /* Fill with constant */
    double fill_val = 42.0;
    status = dtl_tensor_fill_local(tensor, &fill_val);
    CHECK_STATUS(status, "Fill failed");

    status = dtl_tensor_get_local_nd(tensor, indices, &value);
    CHECK_STATUS(status, "Get failed");
    printf("  Value at [0,0] after fill: %.1f\n", value);

    /* Cleanup */
    dtl_tensor_destroy(tensor);

    /* Create a 3D tensor */
    printf("\n3D Tensor Example:\n");
    dtl_shape shape3d = dtl_shape_3d(10, 20, 30);
    status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape3d, &tensor);
    CHECK_STATUS(status, "Failed to create 3D tensor");

    global_shape = dtl_tensor_shape(tensor);
    local_shape = dtl_tensor_local_shape(tensor);

    printf("Rank %d:\n", rank);
    print_shape("  Global shape", global_shape);
    print_shape("  Local shape", local_shape);
    printf("  ndim: %d\n", dtl_tensor_ndim(tensor));

    dtl_tensor_destroy(tensor);
    dtl_context_destroy(ctx);

    if (rank == 0) {
        printf("\nDone!\n");
    }

    return 0;
}
