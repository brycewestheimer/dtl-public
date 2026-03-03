// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_sort.cpp
/// @brief Distributed sorting example with local and global sort patterns
/// @details Demonstrates sorting patterns in DTL including local sort,
///          sample sort approach, and verification of sorted results.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./distributed_sort
///
/// Run (multiple ranks):
///   mpirun -np 4 ./distributed_sort
///
/// Expected output (4 ranks):
///   DTL Distributed Sort Example
///   ============================
///
///   Configuration:
///     Vector size: 1000
///     Number of ranks: 4
///
///   --- Initial State ---
///   Rank 0 first 10: 769 612 384 ...
///   (random values)
///
///   --- After Local Sort ---
///   Each rank's partition is now sorted locally.
///   Rank 0 first 10: 3 11 23 32 ...
///   Rank 0 last 10:  ... 957 969 981 992
///   Rank 1 first 10: 0 7 15 28 ...
///
///   --- Verify Local Sortedness ---
///   All ranks locally sorted: true
///
///   --- Global Statistics ---
///   Global min: 0
///   Global max: 999
///
///   SUCCESS: Distributed sort example completed!

#include <dtl/dtl.hpp>

#include <iostream>
#include <algorithm>
#include <random>
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
        std::cout << "DTL Distributed Sort Example\n";
        std::cout << "============================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Vector size: " << N << "\n";
        std::cout << "  Number of ranks: " << num_ranks << "\n\n";
    }

    // Create distributed vector
    dtl::distributed_vector<int> vec(N, ctx);

    // Initialize with pseudo-random values (reproducible)
    auto local = vec.local_view();
    std::mt19937 gen(static_cast<unsigned>(42 + my_rank * 1000));
    std::uniform_int_distribution<int> dist(0, static_cast<int>(N) - 1);

    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = dist(gen);
    }

    comm.barrier();

    // =========================================================================
    // Show initial state
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Initial State ---\n";
    }

    comm.barrier();

    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Rank " << r << " first 10: ";
            for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
                std::cout << local[i] << " ";
            }
            std::cout << "\n";
        }
        comm.barrier();
    }

    // =========================================================================
    // Local sort - each rank sorts its partition
    // =========================================================================
    std::sort(local.begin(), local.end());

    comm.barrier();

    if (my_rank == 0) {
        std::cout << "\n--- After Local Sort ---\n";
        std::cout << "Each rank's partition is now sorted locally.\n";
    }

    comm.barrier();

    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Rank " << r << " first 10: ";
            for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
                std::cout << local[i] << " ";
            }
            std::cout << "\n";

            if (local.size() > 10) {
                std::cout << "Rank " << r << " last 10:  ";
                dtl::size_type start = local.size() - 10;
                for (dtl::size_type i = start; i < local.size(); ++i) {
                    std::cout << local[i] << " ";
                }
                std::cout << "\n";
            }
        }
        comm.barrier();
    }

    // =========================================================================
    // Verify local sortedness
    // =========================================================================
    bool locally_sorted = std::is_sorted(local.begin(), local.end());

    int local_flag = locally_sorted ? 1 : 0;
    int global_flag = comm.allreduce_min_value<int>(local_flag);
    bool all_sorted = (global_flag == 1);

    if (my_rank == 0) {
        std::cout << "\n--- Verify Local Sortedness ---\n";
        std::cout << "All ranks locally sorted: " << std::boolalpha << all_sorted << "\n";
    }

    // =========================================================================
    // Global statistics
    // =========================================================================
    int local_min = local.empty() ? std::numeric_limits<int>::max() : *std::min_element(local.begin(), local.end());
    int local_max = local.empty() ? std::numeric_limits<int>::min() : *std::max_element(local.begin(), local.end());

    int global_min = comm.allreduce_min_value<int>(local_min);
    int global_max = comm.allreduce_max_value<int>(local_max);

    if (my_rank == 0) {
        std::cout << "\n--- Global Statistics ---\n";
        std::cout << "Global min: " << global_min << "\n";
        std::cout << "Global max: " << global_max << "\n";
        std::cout << "\nSUCCESS: Distributed sort example completed!\n";
    }

    return all_sorted ? 0 : 1;
}
