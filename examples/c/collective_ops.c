// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file collective_ops.c
 * @brief Collective communication operations with DTL C bindings
 *
 * Demonstrates:
 * - Broadcast
 * - Reduce and Allreduce
 * - Gather and Scatter
 *
 * Compile with:
 *   mpicc -I../../include -L../../build/src/bindings/c collective_ops.c -ldtl_c -o collective_ops
 *
 * Run with:
 *   mpirun -np 4 ./collective_ops
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

    /* Create context */
    status = dtl_context_create_default(&ctx);
    CHECK_STATUS(status, "Failed to create context");

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);

    if (rank == 0) {
        printf("Collective Operations Example\n");
        printf("==============================\n");
        printf("Running with %d ranks\n\n", size);
    }

    /* Barrier to sync output */
    dtl_barrier(ctx);

    /*
     * 1. Broadcast
     */
    if (rank == 0) printf("1. Broadcast Example:\n");
    dtl_barrier(ctx);

    int32_t bcast_value = (rank == 0) ? 42 : 0;
    printf("  Rank %d before broadcast: %d\n", rank, bcast_value);

    status = dtl_broadcast(ctx, &bcast_value, 1, DTL_DTYPE_INT32, 0);
    CHECK_STATUS(status, "Broadcast failed");

    printf("  Rank %d after broadcast: %d\n", rank, bcast_value);
    dtl_barrier(ctx);

    /*
     * 2. Reduce (sum to root)
     */
    if (rank == 0) printf("\n2. Reduce Example (sum to root):\n");
    dtl_barrier(ctx);

    int32_t local_value = rank + 1;  /* 1, 2, 3, 4, ... */
    int32_t reduced_value = 0;

    printf("  Rank %d contributing: %d\n", rank, local_value);

    status = dtl_reduce(ctx, &local_value, &reduced_value, 1,
                        DTL_DTYPE_INT32, DTL_OP_SUM, 0);
    CHECK_STATUS(status, "Reduce failed");

    if (rank == 0) {
        int expected = size * (size + 1) / 2;  /* Sum of 1..size */
        printf("  Root received sum: %d (expected: %d)\n", reduced_value, expected);
    }
    dtl_barrier(ctx);

    /*
     * 3. Allreduce (sum to all)
     */
    if (rank == 0) printf("\n3. Allreduce Example (sum to all):\n");
    dtl_barrier(ctx);

    double local_double = (double)rank;
    double global_sum = 0.0;

    status = dtl_allreduce(ctx, &local_double, &global_sum, 1,
                           DTL_DTYPE_FLOAT64, DTL_OP_SUM);
    CHECK_STATUS(status, "Allreduce failed");

    printf("  Rank %d: local=%.1f, global_sum=%.1f\n",
           rank, local_double, global_sum);
    dtl_barrier(ctx);

    /*
     * 4. Allreduce with min/max
     */
    if (rank == 0) printf("\n4. Min/Max Allreduce:\n");
    dtl_barrier(ctx);

    double values[2] = {(double)rank, (double)(size - 1 - rank)};
    double min_max[2];

    status = dtl_allreduce(ctx, &values[0], &min_max[0], 1,
                           DTL_DTYPE_FLOAT64, DTL_OP_MIN);
    CHECK_STATUS(status, "Allreduce min failed");

    status = dtl_allreduce(ctx, &values[1], &min_max[1], 1,
                           DTL_DTYPE_FLOAT64, DTL_OP_MAX);
    CHECK_STATUS(status, "Allreduce max failed");

    printf("  Rank %d: min=%.1f, max=%.1f\n", rank, min_max[0], min_max[1]);
    dtl_barrier(ctx);

    /*
     * 5. Gather (collect to root)
     */
    if (rank == 0) printf("\n5. Gather Example:\n");
    dtl_barrier(ctx);

    int32_t send_val = rank * 10;
    int32_t* recv_buf = NULL;

    if (rank == 0) {
        recv_buf = (int32_t*)malloc(size * sizeof(int32_t));
    }

    printf("  Rank %d sending: %d\n", rank, send_val);

    status = dtl_gather(ctx, &send_val, 1, DTL_DTYPE_INT32, recv_buf, 1, DTL_DTYPE_INT32, 0);
    CHECK_STATUS(status, "Gather failed");

    if (rank == 0) {
        printf("  Root gathered: [");
        for (int i = 0; i < size; i++) {
            if (i > 0) printf(", ");
            printf("%d", recv_buf[i]);
        }
        printf("]\n");
        free(recv_buf);
    }
    dtl_barrier(ctx);

    /*
     * 6. Scatter (distribute from root)
     */
    if (rank == 0) printf("\n6. Scatter Example:\n");
    dtl_barrier(ctx);

    int32_t* scatter_buf = NULL;
    int32_t recv_val = 0;

    if (rank == 0) {
        scatter_buf = (int32_t*)malloc(size * sizeof(int32_t));
        for (int i = 0; i < size; i++) {
            scatter_buf[i] = (i + 1) * 100;
        }
        printf("  Root scattering: [");
        for (int i = 0; i < size; i++) {
            if (i > 0) printf(", ");
            printf("%d", scatter_buf[i]);
        }
        printf("]\n");
    }

    status = dtl_scatter(ctx, scatter_buf, 1, DTL_DTYPE_INT32, &recv_val, 1, DTL_DTYPE_INT32, 0);
    CHECK_STATUS(status, "Scatter failed");

    printf("  Rank %d received: %d\n", rank, recv_val);

    if (rank == 0) {
        free(scatter_buf);
    }
    dtl_barrier(ctx);

    /*
     * 7. Allgather
     */
    if (rank == 0) printf("\n7. Allgather Example:\n");
    dtl_barrier(ctx);

    int32_t my_val = rank;
    int32_t* all_vals = (int32_t*)malloc(size * sizeof(int32_t));

    status = dtl_allgather(ctx, &my_val, 1, DTL_DTYPE_INT32, all_vals, 1, DTL_DTYPE_INT32);
    CHECK_STATUS(status, "Allgather failed");

    printf("  Rank %d received: [", rank);
    for (int i = 0; i < size; i++) {
        if (i > 0) printf(", ");
        printf("%d", all_vals[i]);
    }
    printf("]\n");

    free(all_vals);
    dtl_barrier(ctx);

    /* Cleanup */
    dtl_context_destroy(ctx);

    if (rank == 0) {
        printf("\nDone!\n");
    }

    return 0;
}
