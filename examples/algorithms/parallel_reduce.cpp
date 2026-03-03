// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file parallel_reduce.cpp
/// @brief Complete working example of distributed reduce with MPI
/// @details Demonstrates DTL's distributed reduce algorithm across multiple MPI ranks.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   mpirun -np 4 ./parallel_reduce
///
/// Expected output (4 ranks):
///   [Rank 0] DTL Distributed Reduce Example
///   [Rank 0] Number of ranks: 4
///   [Rank 0] Global vector size: 1000
///   [Rank 0] Local partition size: 250
///   [Rank 1] Local partition size: 250
///   [Rank 2] Local partition size: 250
///   [Rank 3] Local partition size: 250
///   [Rank 0] Local sum: 31125
///   [Rank 1] Local sum: 93625
///   [Rank 2] Local sum: 156125
///   [Rank 3] Local sum: 218625
///   [Rank 0] Global sum (via MPI allreduce): 499500
///   [Rank 1] Global sum (via MPI allreduce): 499500
///   [Rank 2] Global sum (via MPI allreduce): 499500
///   [Rank 3] Global sum (via MPI allreduce): 499500
///   [Rank 0] Expected (0+1+2+...+999): 499500
///   [Rank 0] SUCCESS: Global sum matches expected value!

#include <dtl/dtl.hpp>

#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    // Print header (rank 0 only)
    if (rank == 0) {
        std::cout << "[Rank 0] DTL Distributed Reduce Example\n";
        std::cout << "[Rank 0] Number of ranks: " << size << "\n";
    }
    comm.barrier();

    // Create distributed vector with 1000 elements
    const dtl::size_type global_size = 1000;
    dtl::distributed_vector<int> vec(global_size, ctx);

    // Print global and local sizes
    if (rank == 0) {
        std::cout << "[Rank 0] Global vector size: " << vec.global_size() << "\n";
    }
    comm.barrier();

    // Print local size for each rank
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "[Rank " << rank << "] Local partition size: "
                      << vec.local_size() << "\n";
        }
        comm.barrier();
    }

    // Fill local partition with global indices (0, 1, 2, ..., 999)
    auto local = vec.local_view();
    dtl::index_t offset = vec.global_offset();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(offset + i);
    }
    comm.barrier();

    // Compute local sum first (no communication)
    int local_sum = dtl::local_reduce(vec, 0, std::plus<>{});

    // Print local sums
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "[Rank " << rank << "] Local sum: " << local_sum << "\n";
        }
        comm.barrier();
    }

    // Compute global sum using MPI allreduce
    // This is the key Phase 5 functionality!
    int global_sum = dtl::global_reduce(dtl::par{}, vec, 0, std::plus<>{}, comm);

    // Print global sum (all ranks should have the same value)
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "[Rank " << rank << "] Global sum (via MPI allreduce): "
                      << global_sum << "\n";
        }
        comm.barrier();
    }

    // Verify result (sum of 0..999 = 999*1000/2 = 499500)
    int expected = static_cast<int>(global_size * (global_size - 1) / 2);
    if (rank == 0) {
        std::cout << "[Rank 0] Expected (0+1+2+...+" << (global_size - 1) << "): "
                  << expected << "\n";
        if (global_sum == expected) {
            std::cout << "[Rank 0] SUCCESS: Global sum matches expected value!\n";
        } else {
            std::cout << "[Rank 0] FAILURE: Global sum does not match!\n";
        }
    }

    // Demonstrate reduce_result which gives both local and global
    auto result = dtl::distributed_reduce(dtl::par{}, vec, 0, std::plus<>{}, comm);
    if (rank == 0) {
        std::cout << "\n[Rank 0] Using distributed_reduce for detailed result:\n";
        std::cout << "[Rank 0]   Local value:  " << result.local_value << "\n";
        std::cout << "[Rank 0]   Global value: " << result.global_value << "\n";
        std::cout << "[Rank 0]   has_global:   " << std::boolalpha << result.has_global << "\n";
    }

    // Demonstrate reduce_to (only root gets result)
    auto root_result = dtl::reduce_to(dtl::par{}, vec, 0, comm, 0);
    if (rank == 0) {
        std::cout << "\n[Rank 0] Using reduce_to (only root has valid global):\n";
        std::cout << "[Rank 0]   Global value: " << root_result.global_value << "\n";
    }

    return 0;
}
