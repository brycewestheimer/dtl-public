// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file scan_prefix.cpp
/// @brief Distributed prefix scan using DTL
///
/// Demonstrates:
/// - Local inclusive/exclusive scan on distributed_vector
/// - Cross-rank prefix computation via comm.exscan_sum_value<T>()
/// - Combining local and cross-rank results for global prefix sums
///
/// Run:
///   mpirun -np 4 ./scan_prefix

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Distributed Prefix Scan Example\n";
        std::cout << "=====================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }
    comm.barrier();

    // Create distributed vector filled with 1s
    const dtl::size_type global_size = 20;
    dtl::distributed_vector<long> vec(global_size, ctx);
    auto local = vec.local_view();

    // Fill with 1s (so inclusive scan should give 1, 2, 3, ..., N)
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = 1L;
    }
    comm.barrier();

    // --- Inclusive Scan ---
    // Step 1: Compute local prefix sum
    std::vector<long> inclusive_result(local.size());
    std::inclusive_scan(local.begin(), local.end(), inclusive_result.begin());

    // Step 2: Get the cross-rank prefix (sum of all local sums from prior ranks)
    long local_total = local.empty() ? 0L : inclusive_result.back();
    long cross_rank_prefix = comm.exscan_sum_value<long>(local_total);
    // exscan gives 0 for rank 0, sum of ranks 0..r-1 for rank r

    // Step 3: Add cross-rank prefix to each local element
    for (auto& val : inclusive_result) {
        val += cross_rank_prefix;
    }

    // Print results
    if (rank == 0) std::cout << "Inclusive scan (input: all 1s):\n";
    comm.barrier();

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << r << ": [";
            for (dtl::size_type i = 0; i < inclusive_result.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << inclusive_result[i];
            }
            std::cout << "]\n";
        }
        comm.barrier();
    }

    // --- Exclusive Scan ---
    std::vector<long> exclusive_result(local.size());
    std::exclusive_scan(local.begin(), local.end(), exclusive_result.begin(), 0L);

    // Add cross-rank prefix
    for (auto& val : exclusive_result) {
        val += cross_rank_prefix;
    }

    if (rank == 0) std::cout << "\nExclusive scan (input: all 1s):\n";
    comm.barrier();

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << r << ": [";
            for (dtl::size_type i = 0; i < exclusive_result.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << exclusive_result[i];
            }
            std::cout << "]\n";
        }
        comm.barrier();
    }

    // Verify: last element of inclusive scan on last rank should equal global_size
    long last_val = 0;
    if (rank == size - 1) {
        last_val = inclusive_result.back();
    }
    long global_last = comm.allreduce_max_value<long>(last_val);

    if (rank == 0) {
        std::cout << "\nLast inclusive scan value: " << global_last
                  << " (expected: " << global_size << ")\n";
        std::cout << (global_last == static_cast<long>(global_size) ? "SUCCESS" : "FAILURE") << "\n";
    }

    return 0;
}
