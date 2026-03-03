// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file futures_rpc.c
 * @brief Futures with DTL C bindings
 *
 * Demonstrates:
 * - dtl_future_create / dtl_future_set / dtl_future_get
 * - dtl_when_all for synchronizing multiple futures
 *
 * Note: Futures are experimental features. The progress engine
 *       may have stability issues.
 *
 * Run:
 *   mpirun -np 4 ./c_futures_rpc
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

    status = dtl_context_create_default(&ctx);
    CHECK_STATUS(status, "Failed to create context");

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);

    if (rank == 0) {
        printf("DTL Futures & RPC (C)\n");
        printf("======================\n");
        printf("Running with %d ranks\n\n", size);
    }
    dtl_barrier(ctx);

    /* --- 1. Basic future: create, set, get --- */
    if (rank == 0) printf("1. Basic Future:\n");
    dtl_barrier(ctx);

    {
        dtl_future_t fut = NULL;
        status = dtl_future_create(&fut);
        CHECK_STATUS(status, "Future create failed");

        /* Set a value */
        int32_t value = (rank + 1) * 100;
        status = dtl_future_set(fut, &value, sizeof(int32_t));
        CHECK_STATUS(status, "Future set failed");

        /* Wait for it */
        status = dtl_future_wait(fut);
        CHECK_STATUS(status, "Future wait failed");

        /* Get the value */
        int32_t result = 0;
        status = dtl_future_get(fut, &result, sizeof(int32_t));
        CHECK_STATUS(status, "Future get failed");

        printf("  Rank %d: future value = %d\n", rank, result);

        dtl_future_destroy(fut);
    }
    dtl_barrier(ctx);

    /* --- 2. when_all: wait for multiple futures --- */
    if (rank == 0) printf("\n2. when_all:\n");
    dtl_barrier(ctx);

    {
        const int N = 3;
        dtl_future_t futures[3];

        for (int i = 0; i < N; i++) {
            status = dtl_future_create(&futures[i]);
            CHECK_STATUS(status, "Future create failed");

            int32_t val = rank * 10 + i;
            status = dtl_future_set(futures[i], &val, sizeof(int32_t));
            CHECK_STATUS(status, "Future set failed");
        }

        /* Wait for all */
        dtl_future_t all_fut = NULL;
        status = dtl_when_all(futures, N, &all_fut);
        if (dtl_status_ok(status)) {
            dtl_future_wait(all_fut);
            printf("  Rank %d: all %d futures completed\n", rank, N);
            dtl_future_destroy(all_fut);
        } else {
            /* Fallback: wait individually */
            for (int i = 0; i < N; i++) dtl_future_wait(futures[i]);
            printf("  Rank %d: waited individually for %d futures\n", rank, N);
        }

        /* Get sum of values */
        int32_t sum = 0;
        for (int i = 0; i < N; i++) {
            int32_t val = 0;
            dtl_future_get(futures[i], &val, sizeof(int32_t));
            sum += val;
            dtl_future_destroy(futures[i]);
        }
        printf("  Rank %d: sum = %d\n", rank, sum);
    }
    dtl_barrier(ctx);

    dtl_context_destroy(ctx);

    if (rank == 0) printf("\nDone!\n");
    return 0;
}
