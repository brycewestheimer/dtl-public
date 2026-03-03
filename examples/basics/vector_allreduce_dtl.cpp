// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file vector_allreduce_dtl.cpp
/// @brief Distributed vector sum using only DTL abstractions (no raw MPI calls)
///
/// Demonstrates:
/// - dtl::environment for RAII lifecycle management
/// - env.make_world_context() for context-based construction
/// - dtl::distributed_vector with local_view()
/// - dtl::local_reduce for local summation
/// - comm.allreduce_sum_value<T>() for global reduction
///
/// Run:
///   mpirun -np 4 ./vector_allreduce_dtl

#include <dtl/dtl.hpp>

#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    auto rank = ctx.rank();
    auto size = ctx.size();

    const dtl::size_type global_size = 10000;

    // Create distributed vector
    dtl::distributed_vector<long> vec(global_size, ctx);

    if (rank == 0) {
        std::cout << "DTL Vector Allreduce Example (DTL-Pure)\n";
        std::cout << "========================================\n";
        std::cout << "Global size: " << global_size << "\n";
        std::cout << "Ranks: " << size << "\n\n";
    }

    comm.barrier();

    // Fill local partition with global indices
    auto local = vec.local_view();
    dtl::index_t offset = vec.global_offset();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<long>(offset + static_cast<dtl::index_t>(i));
    }

    comm.barrier();

    // Local reduction (no communication)
    long local_sum = dtl::local_reduce(vec, 0L, std::plus<>{});

    // Print local sums in rank order
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "  Rank " << r << " local sum: " << local_sum << "\n";
        }
        comm.barrier();
    }

    // Global reduction via DTL communicator
    long global_sum = comm.allreduce_sum_value<long>(local_sum);

    // Verify
    long expected = static_cast<long>(global_size) * static_cast<long>(global_size - 1) / 2;

    if (rank == 0) {
        std::cout << "\nGlobal sum: " << global_sum << "\n";
        std::cout << "Expected:   " << expected << "\n";
        std::cout << (global_sum == expected ? "SUCCESS" : "FAILURE") << "\n";
    }

    return (global_sum == expected) ? 0 : 1;
}
