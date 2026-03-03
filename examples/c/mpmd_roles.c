// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpmd_roles.c
 * @brief MPMD Role Manager with DTL C bindings
 *
 * Demonstrates:
 * - dtl_role_manager_create / dtl_role_manager_destroy
 * - dtl_role_manager_add_role / dtl_role_manager_initialize
 * - dtl_intergroup_send / dtl_intergroup_recv for inter-role communication
 *
 * Run:
 *   mpirun -np 4 ./c_mpmd_roles
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
        printf("DTL MPMD Role Manager (C)\n");
        printf("==========================\n");
        printf("Running with %d ranks\n\n", size);
    }
    dtl_barrier(ctx);

    /* Create role manager */
    dtl_role_manager_t mgr = NULL;
    status = dtl_role_manager_create(ctx, &mgr);
    CHECK_STATUS(status, "Failed to create role manager");

    /* Define roles: 1 coordinator + (size-1) workers */
    status = dtl_role_manager_add_role(mgr, "coordinator", 1);
    CHECK_STATUS(status, "Failed to add coordinator role");

    status = dtl_role_manager_add_role(mgr, "worker", size - 1);
    CHECK_STATUS(status, "Failed to add worker role");

    /* Initialize roles (collective) */
    status = dtl_role_manager_initialize(mgr);
    CHECK_STATUS(status, "Failed to initialize roles");

    /* Query role membership */
    int is_coordinator = 0;
    dtl_role_manager_has_role(mgr, "coordinator", &is_coordinator);

    int is_worker = 0;
    dtl_role_manager_has_role(mgr, "worker", &is_worker);

    printf("Rank %d: coordinator=%d, worker=%d\n",
           rank, is_coordinator, is_worker);
    dtl_barrier(ctx);

    /* Inter-group communication: coordinator sends tasks to workers */
    if (is_coordinator) {
        dtl_size_t worker_size = 0;
        dtl_role_manager_role_size(mgr, "worker", &worker_size);

        printf("\nCoordinator: distributing tasks to %lu workers\n",
               (unsigned long)worker_size);

        for (dtl_rank_t w = 0; w < (dtl_rank_t)worker_size; w++) {
            int32_t task = (w + 1) * 10;
            status = dtl_intergroup_send(mgr, "worker", w, &task, 1,
                                          DTL_DTYPE_INT32, 0);
            if (dtl_status_ok(status)) {
                printf("  Sent task %d to worker %d\n", task, w);
            }
        }

        /* Receive results */
        printf("\nResults:\n");
        for (dtl_rank_t w = 0; w < (dtl_rank_t)worker_size; w++) {
            int32_t result = 0;
            status = dtl_intergroup_recv(mgr, "worker", w, &result, 1,
                                          DTL_DTYPE_INT32, 1);
            if (dtl_status_ok(status)) {
                printf("  Worker %d returned: %d\n", w, result);
            }
        }
    }

    if (is_worker) {
        /* Receive task */
        int32_t task = 0;
        status = dtl_intergroup_recv(mgr, "coordinator", 0, &task, 1,
                                      DTL_DTYPE_INT32, 0);
        CHECK_STATUS(status, "Worker recv failed");

        /* Compute: square the task value */
        int32_t result = task * task;

        dtl_rank_t my_worker_rank = 0;
        dtl_role_manager_role_rank(mgr, "worker", &my_worker_rank);

        printf("Worker %d (global rank %d): received %d, computed %d\n",
               my_worker_rank, rank, task, result);

        /* Send result back */
        status = dtl_intergroup_send(mgr, "coordinator", 0, &result, 1,
                                      DTL_DTYPE_INT32, 1);
        CHECK_STATUS(status, "Worker send failed");
    }

    dtl_barrier(ctx);
    dtl_role_manager_destroy(mgr);
    dtl_context_destroy(ctx);

    if (rank == 0) printf("\nDone!\n");
    return 0;
}
