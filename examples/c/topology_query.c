// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file topology_query.c
 * @brief Hardware topology queries with DTL C bindings
 *
 * Demonstrates:
 * - dtl_topology_num_cpus / dtl_topology_num_gpus
 * - dtl_topology_node_id for locality detection
 * - dtl_topology_is_local for co-location checking
 *
 * Run:
 *   mpirun -np 2 ./c_topology_query
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
        printf("DTL Topology Query (C)\n");
        printf("========================\n");
        printf("Running with %d ranks\n\n", size);
    }
    dtl_barrier(ctx);

    /* Query CPU topology */
    int num_cpus = 0;
    status = dtl_topology_num_cpus(&num_cpus);
    if (dtl_status_ok(status)) {
        printf("Rank %d: %d CPUs available\n", rank, num_cpus);
    } else {
        printf("Rank %d: CPU query failed\n", rank);
    }

    /* Query GPU topology */
    int num_gpus = 0;
    status = dtl_topology_num_gpus(&num_gpus);
    if (dtl_status_ok(status)) {
        printf("Rank %d: %d GPUs available\n", rank, num_gpus);
    } else {
        printf("Rank %d: GPU query failed\n", rank);
    }

    /* Query node ID */
    int node_id = -1;
    status = dtl_topology_node_id(rank, &node_id);
    if (dtl_status_ok(status)) {
        printf("Rank %d: node_id = %d\n", rank, node_id);
    }

    /* Query CPU affinity */
    int cpu_id = -1;
    status = dtl_topology_cpu_affinity(rank, &cpu_id);
    if (dtl_status_ok(status)) {
        printf("Rank %d: CPU affinity = %d\n", rank, cpu_id);
    }

    /* GPU ID query */
    if (num_gpus > 0) {
        int gpu_id = -1;
        status = dtl_topology_gpu_id(rank, &gpu_id);
        if (dtl_status_ok(status)) {
            printf("Rank %d: GPU ID = %d\n", rank, gpu_id);
        }
    }

    dtl_barrier(ctx);

    /* Check locality between pairs */
    if (rank == 0 && size > 1) {
        printf("\nLocality checks:\n");
        for (dtl_rank_t r = 1; r < size; r++) {
            int is_local = 0;
            status = dtl_topology_is_local(0, r, &is_local);
            if (dtl_status_ok(status)) {
                printf("  Rank 0 & Rank %d: %s\n", r,
                       is_local ? "same node" : "different nodes");
            }
        }
    }
    dtl_barrier(ctx);

    dtl_context_destroy(ctx);

    if (rank == 0) printf("\nDone!\n");
    return 0;
}
