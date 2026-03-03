// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file distributed_vector.c
 * @brief Distributed vector operations with DTL C bindings
 *
 * Demonstrates:
 * - Creating distributed vectors
 * - Accessing local data
 * - Fill operations
 * - Computing local statistics
 *
 * Compile with:
 *   gcc -I../../include -L../../build/src/bindings/c distributed_vector.c -ldtl_c -lm -o distributed_vector
 *
 * Run with:
 *   ./distributed_vector
 *   mpirun -np 4 ./distributed_vector
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dtl/bindings/c/dtl.h>

#define CHECK_STATUS(status, msg) \
    do { \
        if (!dtl_status_ok(status)) { \
            fprintf(stderr, "%s: %s\n", msg, dtl_status_message(status)); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    dtl_context_t ctx = NULL;
    dtl_vector_t vec = NULL;
    dtl_status status;

    /* Create context */
    status = dtl_context_create_default(&ctx);
    CHECK_STATUS(status, "Failed to create context");

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);

    printf("Rank %d of %d: Distributed Vector Example\n", rank, size);

    /* Create a distributed vector of 1000 doubles */
    dtl_size_t global_size = 1000;
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, &vec);
    CHECK_STATUS(status, "Failed to create vector");

    /* Query vector properties */
    printf("Rank %d: global_size=%lu, local_size=%lu, local_offset=%ld\n",
           rank,
           (unsigned long)dtl_vector_global_size(vec),
           (unsigned long)dtl_vector_local_size(vec),
           (long)dtl_vector_local_offset(vec));

    /* Get local data pointer */
    double* data = (double*)dtl_vector_local_data_mut(vec);
    dtl_size_t local_size = dtl_vector_local_size(vec);
    dtl_index_t offset = dtl_vector_local_offset(vec);

    /* Initialize: each element is its global index */
    for (dtl_size_t i = 0; i < local_size; i++) {
        data[i] = (double)(offset + (dtl_index_t)i);
    }

    /* Compute local statistics */
    double local_sum = 0.0;
    double local_min = data[0];
    double local_max = data[0];

    for (dtl_size_t i = 0; i < local_size; i++) {
        local_sum += data[i];
        if (data[i] < local_min) local_min = data[i];
        if (data[i] > local_max) local_max = data[i];
    }

    double local_mean = local_sum / (double)local_size;

    printf("Rank %d: sum=%.2f, min=%.2f, max=%.2f, mean=%.2f\n",
           rank, local_sum, local_min, local_max, local_mean);

    /* Synchronize */
    status = dtl_vector_barrier(vec);
    CHECK_STATUS(status, "Barrier failed");

    /* Fill with a constant value */
    double fill_value = 42.0;
    status = dtl_vector_fill_local(vec, &fill_value);
    CHECK_STATUS(status, "Fill failed");

    /* Verify fill */
    int all_correct = 1;
    for (dtl_size_t i = 0; i < local_size; i++) {
        if (fabs(data[i] - fill_value) > 1e-10) {
            all_correct = 0;
            break;
        }
    }
    printf("Rank %d: fill verification %s\n", rank, all_correct ? "PASSED" : "FAILED");

    /* Cleanup */
    dtl_vector_destroy(vec);
    dtl_context_destroy(ctx);

    if (rank == 0) {
        printf("\nDone!\n");
    }

    return 0;
}
