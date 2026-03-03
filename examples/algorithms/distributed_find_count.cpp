// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_find_count.cpp
/// @brief Distributed find and count operations using DTL
///
/// Demonstrates:
/// - dtl::local_find to locate elements
/// - dtl::local_count with predicates
/// - comm.allreduce_sum_value for global count aggregation
///
/// Run:
///   mpirun -np 4 ./distributed_find_count

#include <dtl/dtl.hpp>

#include <iostream>
#include <algorithm>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Distributed Find & Count Example\n";
        std::cout << "======================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }

    comm.barrier();

    // Create distributed vector with global indices 0..99
    const dtl::size_type global_size = 100;
    dtl::distributed_vector<int> vec(global_size, ctx);
    auto local = vec.local_view();
    dtl::index_t offset = vec.global_offset();

    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(offset + static_cast<dtl::index_t>(i));
    }

    comm.barrier();

    // --- Find: locate element 42 ---
    const int target = 42;
    auto it = std::find(local.begin(), local.end(), target);
    int found_rank = -1;
    dtl::size_type found_local_idx = 0;

    if (it != local.end()) {
        found_rank = rank;
        found_local_idx = static_cast<dtl::size_type>(std::distance(local.begin(), it));
    }

    // Determine which rank found it
    int global_found_rank = comm.allreduce_max_value<int>(found_rank);

    if (rank == 0) {
        std::cout << "Find element " << target << ":\n";
        if (global_found_rank >= 0) {
            std::cout << "  Found on rank " << global_found_rank << "\n";
        } else {
            std::cout << "  Not found\n";
        }
    }

    if (rank == global_found_rank) {
        std::cout << "  Rank " << rank << ": local index " << found_local_idx
                  << ", global index " << (offset + static_cast<dtl::index_t>(found_local_idx)) << "\n";
    }

    comm.barrier();

    // --- Count: elements divisible by 7 ---
    long local_count = std::count_if(local.begin(), local.end(),
                                     [](int x) { return x % 7 == 0; });

    long global_count = comm.allreduce_sum_value<long>(local_count);

    // Print per-rank counts
    if (rank == 0) std::cout << "\nCount elements divisible by 7:\n";
    comm.barrier();

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << r << ": " << local_count << " elements\n";
        }
        comm.barrier();
    }

    // Verify: 0..99 has elements 0,7,14,21,28,35,42,49,56,63,70,77,84,91,98 = 15
    long expected_count = 15;

    if (rank == 0) {
        std::cout << "\nGlobal count: " << global_count
                  << " (expected: " << expected_count << ")\n";
        std::cout << (global_count == expected_count ? "SUCCESS" : "FAILURE") << "\n";
    }

    return 0;
}
