// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_vector_sum.cpp
/// @brief Complete example of creating and summing a distributed vector
/// @details Demonstrates the full workflow of distributed vector operations
///          using dtl::environment and make_world_context():
///          creation, initialization, local processing, and collective reduction.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./distributed_vector_sum
///
/// Run (multiple ranks):
///   mpirun -np 4 ./distributed_vector_sum
///
/// Expected output (4 ranks):
///   DTL Distributed Vector Sum Example
///   ==================================
///
///   Configuration:
///     Global size: 10000
///     Number of ranks: 4
///
///   Partition distribution:
///     Rank 0: elements [0, 2500) - 2500 elements
///     Rank 1: elements [2500, 5000) - 2500 elements
///     Rank 2: elements [5000, 7500) - 2500 elements
///     Rank 3: elements [7500, 10000) - 2500 elements
///
///   Local sums:
///     Rank 0 local sum: 3123750
///     Rank 1 local sum: 9373750
///     Rank 2 local sum: 15623750
///     Rank 3 local sum: 21873750
///
///   Global sum (via dtl::reduce): 49995000
///   Expected sum (0+1+...+9999):  49995000
///
///   SUCCESS: Distributed sum is correct!

#include <dtl/dtl.hpp>

#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    // Configuration
    const dtl::size_type global_size = 10000;

    // Create distributed vector
    dtl::distributed_vector<long> vec(global_size, ctx);

    // Print header
    if (my_rank == 0) {
        std::cout << "DTL Distributed Vector Sum Example\n";
        std::cout << "==================================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Global size: " << global_size << "\n";
        std::cout << "  Number of ranks: " << num_ranks << "\n\n";
        std::cout << "Partition distribution:\n";
    }

    comm.barrier();

    // Print partition info for each rank
    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "  Rank " << r << ": elements ["
                      << vec.global_offset() << ", "
                      << (vec.global_offset() + vec.local_size()) << ") - "
                      << vec.local_size() << " elements\n";
        }
        comm.barrier();
    }

    // Initialize local partition with global indices
    auto local = vec.local_view();
    dtl::index_t offset = vec.global_offset();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<long>(offset + static_cast<dtl::index_t>(i));
    }

    comm.barrier();

    // Compute local sum
    long local_sum = dtl::local_reduce(vec, 0L, std::plus<>{});

    if (my_rank == 0) {
        std::cout << "\nLocal sums:\n";
    }

    comm.barrier();

    // Print local sums
    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "  Rank " << r << " local sum: " << local_sum << "\n";
        }
        comm.barrier();
    }

    // Compute global sum via distributed reduce
    long global_sum = comm.allreduce_sum_value<long>(local_sum);

    // Expected: sum of 0 to N-1 = N*(N-1)/2
    long expected = static_cast<long>(global_size) * static_cast<long>(global_size - 1) / 2;

    if (my_rank == 0) {
        std::cout << "\nGlobal sum (via dtl::reduce): " << global_sum << "\n";
        std::cout << "Expected sum (0+1+...+" << (global_size - 1) << "):  " << expected << "\n\n";

        if (global_sum == expected) {
            std::cout << "SUCCESS: Distributed sum is correct!\n";
        } else {
            std::cout << "FAILURE: Sum mismatch!\n";
        }
    }

    return (global_sum == expected) ? 0 : 1;
}
