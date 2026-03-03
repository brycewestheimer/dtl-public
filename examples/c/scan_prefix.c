// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file scan_prefix.c
 * @brief Prefix scan operations with DTL C bindings
 *
 * Demonstrates:
 * - dtl_inclusive_scan_vector for inclusive prefix sum
 * - dtl_exclusive_scan_vector for exclusive prefix sum
 *
 * Run:
 *   ./c_scan_prefix
 *   mpirun -np 4 ./c_scan_prefix
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

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    dtl_context_t ctx = NULL;
    dtl_status status;

    status = dtl_context_create_default(&ctx);
    CHECK_STATUS(status, "Failed to create context");

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);

    if (rank == 0) {
        printf("DTL Prefix Scan (C)\n");
        printf("=====================\n");
        printf("Running with %d ranks\n\n", size);
    }
    dtl_barrier(ctx);

    /* Create a vector of 16 integers filled with 1s */
    dtl_vector_t vec = NULL;
    int32_t fill_val = 1;
    status = dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 16, &fill_val, &vec);
    CHECK_STATUS(status, "Failed to create vector");

    dtl_size_t local_size = dtl_vector_local_size(vec);
    int32_t* data = (int32_t*)dtl_vector_local_data_mut(vec);

    printf("Rank %d input: [", rank);
    for (dtl_size_t i = 0; i < local_size; i++) {
        if (i > 0) printf(", ");
        printf("%d", data[i]);
    }
    printf("]\n");
    dtl_barrier(ctx);

    /* Inclusive scan */
    status = dtl_inclusive_scan_vector(vec, DTL_OP_SUM);
    CHECK_STATUS(status, "Inclusive scan failed");

    printf("Rank %d inclusive scan: [", rank);
    for (dtl_size_t i = 0; i < local_size; i++) {
        if (i > 0) printf(", ");
        printf("%d", data[i]);
    }
    printf("]\n");
    dtl_barrier(ctx);

    /* Save last inclusive scan value before overwriting */
    int32_t last_inclusive = data[local_size - 1];

    /* Reset to 1s and do exclusive scan */
    for (dtl_size_t i = 0; i < local_size; i++) {
        data[i] = 1;
    }

    dtl_vector_t vec2 = NULL;
    status = dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 16, &fill_val, &vec2);
    CHECK_STATUS(status, "Failed to create vector2");

    status = dtl_exclusive_scan_vector(vec2, DTL_OP_SUM);
    CHECK_STATUS(status, "Exclusive scan failed");

    int32_t* data2 = (int32_t*)dtl_vector_local_data_mut(vec2);
    dtl_size_t local_size2 = dtl_vector_local_size(vec2);

    printf("Rank %d exclusive scan: [", rank);
    for (dtl_size_t i = 0; i < local_size2; i++) {
        if (i > 0) printf(", ");
        printf("%d", data2[i]);
    }
    printf("]\n");
    dtl_barrier(ctx);

    /* Verify: last element of inclusive scan on last rank should be 16 */
    if (rank == size - 1) {
        printf("\nLast inclusive scan value: %d (expected: 16)\n", last_inclusive);
        printf("%s\n", (last_inclusive == 16) ? "SUCCESS" : "FAILURE");
    }

    dtl_vector_destroy(vec2);
    dtl_vector_destroy(vec);
    dtl_context_destroy(ctx);

    return 0;
}
