// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file rma_operations.c
 * @brief RMA Put/Get operations with DTL C bindings
 *
 * Demonstrates:
 * - dtl_window_create / dtl_window_destroy for window lifecycle
 * - dtl_window_fence for active-target synchronization
 * - dtl_rma_put / dtl_rma_get for one-sided data transfer
 *
 * Note: RMA may fail on WSL2 OpenMPI 4.1.6. Example includes error handling.
 *
 * Run:
 *   mpirun -np 2 ./c_rma_operations
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

    if (size < 2) {
        if (rank == 0) printf("This example requires at least 2 ranks.\n");
        dtl_context_destroy(ctx);
        return 1;
    }

    if (rank == 0) {
        printf("DTL RMA Operations (C)\n");
        printf("========================\n");
        printf("Running with %d ranks\n\n", size);
    }
    dtl_barrier(ctx);

    /* Create a local buffer and expose via RMA window */
    int32_t local_buf[4];
    for (int i = 0; i < 4; i++) {
        local_buf[i] = rank * 100 + i;
    }

    dtl_window_t win = NULL;
    status = dtl_window_create(ctx, local_buf, 4 * sizeof(int32_t), &win);
    if (!dtl_status_ok(status)) {
        fprintf(stderr, "Rank %d: Window creation failed (RMA not supported?)\n", rank);
        dtl_context_destroy(ctx);
        return 1;
    }

    /* Initial fence */
    status = dtl_window_fence(win);
    CHECK_STATUS(status, "Initial fence failed");

    /* Rank 0 puts value 999 into rank 1's buffer at offset 0 */
    if (rank == 0) {
        int32_t value = 999;
        status = dtl_rma_put(win, 1, 0, &value, sizeof(int32_t));
        if (dtl_status_ok(status)) {
            printf("Rank 0: Put value %d to rank 1 offset 0\n", value);
        } else {
            fprintf(stderr, "Rank 0: Put failed\n");
        }
    }

    /* Fence to complete the put */
    status = dtl_window_fence(win);
    CHECK_STATUS(status, "Fence after put failed");

    /* Rank 1 verifies the received value */
    if (rank == 1) {
        printf("Rank 1: buffer[0] = %d (expected 999)\n", local_buf[0]);
    }
    dtl_barrier(ctx);

    /* Rank 1 reads rank 0's buffer[2] via get */
    if (rank == 1) {
        int32_t read_val = -1;
        status = dtl_rma_get(win, 0, 2 * sizeof(int32_t), &read_val, sizeof(int32_t));
        if (dtl_status_ok(status)) {
            printf("Rank 1: Got value %d from rank 0 offset 2 (expected %d)\n",
                   read_val, 0 * 100 + 2);
        }
    }

    /* Final fence */
    status = dtl_window_fence(win);
    CHECK_STATUS(status, "Final fence failed");

    dtl_window_destroy(win);
    dtl_context_destroy(ctx);

    if (rank == 0) printf("\nDone!\n");
    return 0;
}
