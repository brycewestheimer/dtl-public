// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file jacobi_1d.c
 * @brief 1D Jacobi iterative solver with DTL C bindings
 *
 * Solves u''(x) = 0 with boundary conditions u(0) = 1, u(L) = 0.
 * Uses Jacobi iteration with halo exchange via dtl_send / dtl_recv.
 *
 * Demonstrates:
 * - Halo exchange using dtl_send / dtl_recv
 * - Convergence checking via dtl_allreduce with DTL_OP_MAX
 * - Distributed iterative solver pattern
 *
 * Run:
 *   mpirun -np 4 ./c_jacobi_1d
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
        printf("DTL 1D Jacobi Solver (C)\n");
        printf("=========================\n");
        printf("Ranks: %d\n", size);
    }
    dtl_barrier(ctx);

    /* Problem setup */
    const int global_n = 100;       /* Interior points */
    const int max_iter = 10000;
    const double tol = 1e-8;

    /* Partition interior points among ranks */
    int local_n = global_n / size;
    int remainder = global_n % size;
    if (rank < remainder) local_n += 1;

    /* Allocate arrays with halo cells: [left_halo | interior | right_halo] */
    double* u     = (double*)calloc((size_t)(local_n + 2), sizeof(double));
    double* u_new = (double*)calloc((size_t)(local_n + 2), sizeof(double));
    if (!u || !u_new) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    /* Apply boundary conditions: u(0) = 1.0 on leftmost rank */
    if (rank == 0) {
        u[0] = 1.0;
        u_new[0] = 1.0;
    }
    /* u(L) = 0.0 on rightmost rank's right boundary (already 0) */

    const int halo_tag = 10;

    if (rank == 0) {
        printf("Grid: %d interior points\n", global_n);
        printf("BCs: u(0)=1, u(L)=0\n");
        printf("Tolerance: %.0e\n\n", tol);
    }
    dtl_barrier(ctx);

    int iter = 0;
    double global_diff = 0.0;

    for (iter = 0; iter < max_iter; ++iter) {
        /* --- Halo exchange --- */
        /* Even ranks send right, then left; odd ranks receive first */
        /* Send right boundary to right neighbor */
        if (rank < size - 1) {
            status = dtl_send(ctx, &u[local_n], 1, DTL_DTYPE_FLOAT64,
                              rank + 1, halo_tag);
            CHECK_STATUS(status, "Send right failed");
        }
        if (rank > 0) {
            status = dtl_recv(ctx, &u[0], 1, DTL_DTYPE_FLOAT64,
                              rank - 1, halo_tag);
            CHECK_STATUS(status, "Recv left failed");
        }

        /* Send left boundary to left neighbor */
        if (rank > 0) {
            status = dtl_send(ctx, &u[1], 1, DTL_DTYPE_FLOAT64,
                              rank - 1, halo_tag + 1);
            CHECK_STATUS(status, "Send left failed");
        }
        if (rank < size - 1) {
            status = dtl_recv(ctx, &u[local_n + 1], 1, DTL_DTYPE_FLOAT64,
                              rank + 1, halo_tag + 1);
            CHECK_STATUS(status, "Recv right failed");
        }

        /* --- Jacobi update --- */
        double local_diff = 0.0;
        for (int i = 1; i <= local_n; ++i) {
            u_new[i] = 0.5 * (u[i - 1] + u[i + 1]);
            double d = fabs(u_new[i] - u[i]);
            if (d > local_diff) local_diff = d;
        }

        /* Copy new to old */
        for (int i = 1; i <= local_n; ++i) {
            u[i] = u_new[i];
        }

        /* Check convergence: global max diff */
        status = dtl_allreduce(ctx, &local_diff, &global_diff, 1,
                               DTL_DTYPE_FLOAT64, DTL_OP_MAX);
        CHECK_STATUS(status, "Allreduce max failed");

        if (global_diff < tol) {
            break;
        }
    }

    dtl_barrier(ctx);

    if (rank == 0) {
        printf("Converged after %d iterations\n", iter);
        printf("Final max diff: %.4e\n\n", global_diff);
    }

    /* Print solution samples */
    for (dtl_rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            printf("  Rank %d: u[first]=%.6f, u[last]=%.6f\n",
                   rank, u[1], u[local_n]);
        }
        dtl_barrier(ctx);
    }

    if (rank == 0) {
        printf("\nExpected: linear from 1.0 to 0.0\n");
        printf("%s: Solver %s\n",
               global_diff < tol ? "SUCCESS" : "FAILURE",
               global_diff < tol ? "converged" : "did not converge");
    }

    free(u);
    free(u_new);
    dtl_context_destroy(ctx);
    return 0;
}
