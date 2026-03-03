// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file hello_dtl.c
 * @brief Basic DTL C bindings example
 *
 * Demonstrates:
 * - Creating a DTL context
 * - Querying rank and size
 * - Feature detection
 *
 * Compile with:
 *   gcc -I../../include -L../../build/src/bindings/c hello_dtl.c -ldtl_c -o hello_dtl
 *
 * Run with:
 *   ./hello_dtl
 *   mpirun -np 4 ./hello_dtl
 */

#include <stdio.h>
#include <dtl/bindings/c/dtl.h>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    /* Print version info */
    printf("DTL C Bindings Example\n");
    printf("======================\n");
    printf("Version: %s\n", dtl_version_string());
    printf("Version: %d.%d.%d\n",
           dtl_version_major(), dtl_version_minor(), dtl_version_patch());
    printf("ABI Version: %d\n", dtl_abi_version());
    printf("\n");

    /* Check available backends */
    printf("Available backends:\n");
    printf("  MPI:   %s\n", dtl_has_mpi() ? "yes" : "no");
    printf("  CUDA:  %s\n", dtl_has_cuda() ? "yes" : "no");
    printf("  HIP:   %s\n", dtl_has_hip() ? "yes" : "no");
    printf("  NCCL:  %s\n", dtl_has_nccl() ? "yes" : "no");
    printf("  SHMEM: %s\n", dtl_has_shmem() ? "yes" : "no");
    printf("\n");

    /* Create context */
    dtl_context_t ctx = NULL;
    dtl_status status = dtl_context_create_default(&ctx);

    if (!dtl_status_ok(status)) {
        fprintf(stderr, "Failed to create context: %s\n",
                dtl_status_message(status));
        return 1;
    }

    /* Query context properties */
    printf("Context created:\n");
    printf("  Rank: %d\n", dtl_context_rank(ctx));
    printf("  Size: %d\n", dtl_context_size(ctx));
    printf("  Is root: %s\n", dtl_context_is_root(ctx) ? "yes" : "no");
    printf("  Device ID: %d\n", dtl_context_device_id(ctx));
    printf("  Has device: %s\n", dtl_context_has_device(ctx) ? "yes" : "no");

    /* Cleanup */
    dtl_context_destroy(ctx);

    printf("\nDone!\n");
    return 0;
}
