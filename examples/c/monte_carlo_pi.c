// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file monte_carlo_pi.c
 * @brief Monte Carlo Pi estimation with DTL C bindings
 *
 * Demonstrates:
 * - dtl_allreduce for global summation
 * - Random sampling across distributed ranks
 * - Parallel random number generation with rank-specific seeds
 *
 * Run:
 *   mpirun -np 4 ./c_monte_carlo_pi
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
    dtl_status status;

    status = dtl_context_create_default(&ctx);
    CHECK_STATUS(status, "Failed to create context");

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);

    if (rank == 0) {
        printf("DTL Monte Carlo Pi Estimation (C)\n");
        printf("==================================\n");
        printf("Running with %d ranks\n\n", size);
    }
    dtl_barrier(ctx);

    /* Configuration */
    const int64_t samples_per_rank = 1000000;
    const int64_t total_samples = samples_per_rank * (int64_t)size;

    /* Rank-specific seed for independent random streams */
    srand((unsigned)(42 + rank * 12345));

    /* Sample random points and count hits inside unit circle */
    int64_t local_hits = 0;
    for (int64_t i = 0; i < samples_per_rank; ++i) {
        double x = (double)rand() / (double)RAND_MAX;
        double y = (double)rand() / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_hits++;
        }
    }

    double local_pi = 4.0 * (double)local_hits / (double)samples_per_rank;
    printf("  Rank %d: %ld / %ld hits (local pi ~ %.6f)\n",
           rank, (long)local_hits, (long)samples_per_rank, local_pi);
    dtl_barrier(ctx);

    /* Global reduction: sum hits across all ranks */
    int64_t global_hits = 0;
    status = dtl_allreduce(ctx, &local_hits, &global_hits, 1,
                           DTL_DTYPE_INT64, DTL_OP_SUM);
    CHECK_STATUS(status, "Allreduce failed");

    double pi_estimate = 4.0 * (double)global_hits / (double)total_samples;
    double error = fabs(pi_estimate - 3.14159265358979323846);

    if (rank == 0) {
        printf("\nTotal samples: %ld\n", (long)total_samples);
        printf("Total hits:    %ld\n", (long)global_hits);
        printf("Pi estimate:   %.8f\n", pi_estimate);
        printf("Actual pi:     %.8f\n", 3.14159265358979323846);
        printf("Error:         %.4e\n", error);

        if (error < 0.01) {
            printf("SUCCESS: Estimate within 0.01 tolerance\n");
        } else {
            printf("WARNING: Estimate outside 0.01 tolerance\n");
        }
    }

    dtl_context_destroy(ctx);
    return 0;
}
