// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file point_to_point.c
 * @brief Point-to-point communication with DTL C bindings
 *
 * Demonstrates:
 * - dtl_send / dtl_recv for blocking P2P
 * - dtl_sendrecv for combined send/receive
 * - Ring communication pattern
 *
 * Run:
 *   mpirun -np 4 ./c_point_to_point
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
        printf("DTL Point-to-Point Communication (C)\n");
        printf("======================================\n");
        printf("Running with %d ranks\n\n", size);
    }
    dtl_barrier(ctx);

    /* 1. Simple send/recv between rank 0 and rank 1 */
    if (rank == 0) printf("1. Send/Recv (rank 0 -> rank 1):\n");
    dtl_barrier(ctx);

    if (rank == 0) {
        int32_t value = 42;
        status = dtl_send(ctx, &value, 1, DTL_DTYPE_INT32, 1, 0);
        CHECK_STATUS(status, "Send failed");
        printf("  Rank 0 sent: %d\n", value);
    } else if (rank == 1) {
        int32_t value = 0;
        status = dtl_recv(ctx, &value, 1, DTL_DTYPE_INT32, 0, 0);
        CHECK_STATUS(status, "Recv failed");
        printf("  Rank 1 received: %d\n", value);
    }
    dtl_barrier(ctx);

    /* 2. Sendrecv ring: each rank exchanges with next/prev */
    if (rank == 0) printf("\n2. Sendrecv Ring:\n");
    dtl_barrier(ctx);

    {
        dtl_rank_t next = (rank + 1) % size;
        dtl_rank_t prev = (rank + size - 1) % size;

        int32_t send_val = rank * 100;
        int32_t recv_val = -1;

        status = dtl_sendrecv(ctx,
                              &send_val, 1, DTL_DTYPE_INT32, next, 10,
                              &recv_val, 1, DTL_DTYPE_INT32, prev, 10);
        CHECK_STATUS(status, "Sendrecv failed");

        printf("  Rank %d: sent %d to rank %d, received %d from rank %d\n",
               rank, send_val, next, recv_val, prev);
    }
    dtl_barrier(ctx);

    /* 3. Non-blocking send/recv */
    if (rank == 0) printf("\n3. Non-blocking (isend/irecv):\n");
    dtl_barrier(ctx);

    if (rank == 0) {
        int32_t send_val = 999;
        int32_t recv_val = 0;
        dtl_request_t send_req = NULL, recv_req = NULL;

        status = dtl_isend(ctx, &send_val, 1, DTL_DTYPE_INT32, 1, 20, &send_req);
        CHECK_STATUS(status, "Isend failed");

        status = dtl_irecv(ctx, &recv_val, 1, DTL_DTYPE_INT32, 1, 21, &recv_req);
        CHECK_STATUS(status, "Irecv failed");

        dtl_wait(send_req);
        dtl_wait(recv_req);

        printf("  Rank 0: async sent %d, async received %d\n", send_val, recv_val);
        dtl_request_free(send_req);
        dtl_request_free(recv_req);
    } else if (rank == 1) {
        int32_t send_val = 888;
        int32_t recv_val = 0;
        dtl_request_t send_req = NULL, recv_req = NULL;

        status = dtl_irecv(ctx, &recv_val, 1, DTL_DTYPE_INT32, 0, 20, &recv_req);
        CHECK_STATUS(status, "Irecv failed");

        status = dtl_isend(ctx, &send_val, 1, DTL_DTYPE_INT32, 0, 21, &send_req);
        CHECK_STATUS(status, "Isend failed");

        dtl_wait(recv_req);
        dtl_wait(send_req);

        printf("  Rank 1: async received %d, async sent %d\n", recv_val, send_val);
        dtl_request_free(send_req);
        dtl_request_free(recv_req);
    }
    dtl_barrier(ctx);

    dtl_context_destroy(ctx);

    if (rank == 0) printf("\nDone!\n");
    return 0;
}
