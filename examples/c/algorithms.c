// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file algorithms.c
 * @brief Algorithm operations with DTL C bindings
 *
 * Demonstrates:
 * - dtl_for_each_vector for element-wise operations
 * - dtl_transform_vector for transformations
 * - dtl_reduce_local_vector for local reduction
 * - dtl_sort_vector for sorting
 * - dtl_find_vector / dtl_count_if_vector for search/count
 * - dtl_minmax_vector for min/max
 *
 * Run:
 *   ./c_algorithms
 *   mpirun -np 4 ./c_algorithms
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

/* Callback: multiply each element by 2 */
static void double_value(void* elem, dtl_size_t index, void* user_data) {
    (void)index;
    (void)user_data;
    double* p = (double*)elem;
    *p *= 2.0;
}

/* Transform callback: square value */
static void square_value(const void* src, void* dst, dtl_size_t index, void* user_data) {
    (void)index;
    (void)user_data;
    const double* s = (const double*)src;
    double* d = (double*)dst;
    *d = (*s) * (*s);
}

/* Predicate: is element > 50? */
static int greater_than_50(const void* elem, void* user_data) {
    (void)user_data;
    const double* p = (const double*)elem;
    return (*p > 50.0) ? 1 : 0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    dtl_context_t ctx = NULL;
    dtl_status status;

    status = dtl_context_create_default(&ctx);
    CHECK_STATUS(status, "Failed to create context");

    dtl_rank_t rank = dtl_context_rank(ctx);

    if (rank == 0) {
        printf("DTL Algorithm Operations (C)\n");
        printf("==============================\n\n");
    }
    dtl_barrier(ctx);

    /* Create a vector of 20 doubles */
    dtl_vector_t vec = NULL;
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 20, &vec);
    CHECK_STATUS(status, "Failed to create vector");

    /* Initialize with values 1..local_size */
    double* data = (double*)dtl_vector_local_data_mut(vec);
    dtl_size_t local_size = dtl_vector_local_size(vec);
    dtl_index_t offset = dtl_vector_local_offset(vec);

    for (dtl_size_t i = 0; i < local_size; i++) {
        data[i] = (double)(offset + (dtl_index_t)i + 1);
    }

    if (rank == 0) printf("1. Initial values (first rank): ");
    dtl_barrier(ctx);
    if (rank == 0) {
        for (dtl_size_t i = 0; i < local_size && i < 10; i++) {
            printf("%.0f ", data[i]);
        }
        printf("...\n");
    }
    dtl_barrier(ctx);

    /* 2. For-each: double all values */
    status = dtl_for_each_vector(vec, double_value, NULL);
    CHECK_STATUS(status, "For-each failed");

    if (rank == 0) {
        printf("\n2. After doubling: ");
        for (dtl_size_t i = 0; i < local_size && i < 10; i++) {
            printf("%.0f ", data[i]);
        }
        printf("...\n");
    }
    dtl_barrier(ctx);

    /* 3. Reduce: sum all local elements */
    double local_sum = 0.0;
    status = dtl_reduce_local_vector(vec, DTL_OP_SUM, &local_sum);
    CHECK_STATUS(status, "Reduce failed");

    printf("Rank %d: local sum = %.2f\n", rank, local_sum);
    dtl_barrier(ctx);

    /* 4. Find: look for value 10.0 */
    dtl_index_t found_idx = dtl_find_vector(vec, &(double){10.0});
    if (found_idx >= 0) {
        printf("Rank %d: found 10.0 at local index %ld\n", rank, (long)found_idx);
    }
    dtl_barrier(ctx);

    /* 5. Count: elements > 50 */
    dtl_size_t count = dtl_count_if_vector(vec, greater_than_50, NULL);
    printf("Rank %d: %lu elements > 50\n", rank, (unsigned long)count);
    dtl_barrier(ctx);

    /* 6. Min/Max */
    double min_val = 0.0, max_val = 0.0;
    status = dtl_minmax_vector(vec, &min_val, &max_val);
    CHECK_STATUS(status, "Minmax failed");

    printf("Rank %d: min=%.2f, max=%.2f\n", rank, min_val, max_val);
    dtl_barrier(ctx);

    /* 7. Sort (ascending) */
    status = dtl_sort_vector(vec);
    CHECK_STATUS(status, "Sort failed");

    if (rank == 0) {
        printf("\n7. After sort: ");
        for (dtl_size_t i = 0; i < local_size && i < 10; i++) {
            printf("%.0f ", data[i]);
        }
        printf("...\n");
    }
    dtl_barrier(ctx);

    /* 8. Transform: create squared copy */
    dtl_vector_t vec2 = NULL;
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 20, &vec2);
    CHECK_STATUS(status, "Failed to create output vector");

    status = dtl_transform_vector(vec, vec2, square_value, NULL);
    CHECK_STATUS(status, "Transform failed");

    if (rank == 0) {
        double* data2 = (double*)dtl_vector_local_data_mut(vec2);
        dtl_size_t sz2 = dtl_vector_local_size(vec2);
        printf("\n8. Squared values: ");
        for (dtl_size_t i = 0; i < sz2 && i < 10; i++) {
            printf("%.0f ", data2[i]);
        }
        printf("...\n");
    }

    dtl_vector_destroy(vec2);
    dtl_vector_destroy(vec);
    dtl_context_destroy(ctx);

    if (rank == 0) printf("\nDone!\n");
    return 0;
}
