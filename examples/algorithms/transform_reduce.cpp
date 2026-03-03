// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file transform_reduce.cpp
/// @brief Combined transform and reduce operations on distributed data
/// @details Demonstrates dtl::transform_reduce for efficient fused operations
///          like computing dot products, norms, and weighted sums.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./transform_reduce
///
/// Run (multiple ranks):
///   mpirun -np 4 ./transform_reduce
///
/// Expected output:
///   DTL Transform-Reduce Example
///   ============================
///
///   Configuration:
///     Vector size: 1000
///     Number of ranks: 4
///
///   --- Example 1: Sum of Squares (L2 norm squared) ---
///   Vector contains: 0, 1, 2, ..., 999
///   Sum of x^2: 332833500
///   L2 norm: 18244.3
///
///   --- Example 2: Dot Product ---
///   vec_a: 0, 1, 2, ..., 999
///   vec_b: 999, 998, 997, ..., 0
///   Dot product: 166167000
///
///   --- Example 3: Weighted Sum ---
///   Values: 0, 1, 2, ..., 999
///   Weights: 1, 2, 3, ..., 1000
///   Weighted sum: 332833500
///
///   --- Example 4: Count matching predicate ---
///   Counting elements divisible by 7
///   Count: 143
///
///   SUCCESS: All transform-reduce operations completed!

#include <dtl/dtl.hpp>

#include <iostream>
#include <cmath>
#include <numeric>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    const dtl::size_type N = 1000;

    if (my_rank == 0) {
        std::cout << "DTL Transform-Reduce Example\n";
        std::cout << "============================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Vector size: " << N << "\n";
        std::cout << "  Number of ranks: " << num_ranks << "\n\n";
    }

    // Create vectors
    dtl::distributed_vector<long> vec(N, ctx);
    dtl::distributed_vector<long> vec_a(N, ctx);
    dtl::distributed_vector<long> vec_b(N, ctx);
    dtl::distributed_vector<long> weights(N, ctx);

    // Initialize
    auto local = vec.local_view();
    auto local_a = vec_a.local_view();
    auto local_b = vec_b.local_view();
    auto local_w = weights.local_view();
    dtl::index_t offset = vec.global_offset();

    for (dtl::size_type i = 0; i < local.size(); ++i) {
        long idx = static_cast<long>(offset) + static_cast<long>(i);
        local[i] = idx;                           // 0, 1, 2, ...
        local_a[i] = idx;                         // 0, 1, 2, ...
        local_b[i] = static_cast<long>(N) - 1 - idx;  // N-1, N-2, ...
        local_w[i] = idx + 1;                     // 1, 2, 3, ...
    }

    comm.barrier();

    // =========================================================================
    // Example 1: Sum of Squares (L2 norm squared)
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Example 1: Sum of Squares (L2 norm squared) ---\n";
        std::cout << "Vector contains: 0, 1, 2, ..., " << (N - 1) << "\n";
    }

    // Local transform-reduce: sum of x^2
    long local_sum_sq = 0;
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local_sum_sq += local[i] * local[i];
    }

    long global_sum_sq = comm.allreduce_sum_value<long>(local_sum_sq);

    if (my_rank == 0) {
        std::cout << "Sum of x^2: " << global_sum_sq << "\n";
        std::cout << "L2 norm: " << std::sqrt(static_cast<double>(global_sum_sq)) << "\n\n";
    }

    // =========================================================================
    // Example 2: Dot Product
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Example 2: Dot Product ---\n";
        std::cout << "vec_a: 0, 1, 2, ..., " << (N - 1) << "\n";
        std::cout << "vec_b: " << (N - 1) << ", " << (N - 2) << ", ..., 0\n";
    }

    // Compute local dot product
    long local_dot = 0;
    for (dtl::size_type i = 0; i < local_a.size(); ++i) {
        local_dot += local_a[i] * local_b[i];
    }

    long global_dot = comm.allreduce_sum_value<long>(local_dot);

    if (my_rank == 0) {
        std::cout << "Dot product: " << global_dot << "\n\n";
    }

    // =========================================================================
    // Example 3: Weighted Sum
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Example 3: Weighted Sum ---\n";
        std::cout << "Values: 0, 1, 2, ..., " << (N - 1) << "\n";
        std::cout << "Weights: 1, 2, 3, ..., " << N << "\n";
    }

    // Compute local weighted sum
    long local_weighted = 0;
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local_weighted += local[i] * local_w[i];
    }

    long global_weighted = comm.allreduce_sum_value<long>(local_weighted);

    if (my_rank == 0) {
        std::cout << "Weighted sum: " << global_weighted << "\n\n";
    }

    // =========================================================================
    // Example 4: Count with predicate (transform to 0/1, then reduce)
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Example 4: Count matching predicate ---\n";
        std::cout << "Counting elements divisible by 7\n";
    }

    // Count elements divisible by 7
    long local_count = 0;
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        if (local[i] % 7 == 0) {
            local_count++;
        }
    }

    long global_count = comm.allreduce_sum_value<long>(local_count);

    if (my_rank == 0) {
        std::cout << "Count: " << global_count << "\n\n";
        std::cout << "SUCCESS: All transform-reduce operations completed!\n";
    }

    return 0;
}
