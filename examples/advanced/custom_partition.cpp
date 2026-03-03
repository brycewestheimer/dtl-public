// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file custom_partition.cpp
/// @brief User-defined partition functions for custom data distribution
/// @details Demonstrates how to use DTL's partition policies to control
///          how data is distributed across ranks, including block, cyclic,
///          and custom partitioning strategies.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   mpirun -np 4 ./custom_partition
///
/// Expected output (4 ranks):
///   DTL Custom Partition Example
///   ============================
///
///   Configuration:
///     Global size: 100
///     Number of ranks: 4
///
///   --- Block Partition (Default) ---
///   Contiguous chunks assigned to each rank.
///   Rank 0: owns indices [0, 25)
///   Rank 1: owns indices [25, 50)
///   Rank 2: owns indices [50, 75)
///   Rank 3: owns indices [75, 100)
///
///   --- Simulated Cyclic Access Pattern ---
///   Round-robin assignment (index i -> rank i % num_ranks).
///   Index 0 -> Rank 0, Index 1 -> Rank 1, ...
///   Index 4 -> Rank 0, Index 5 -> Rank 1, ...
///
///   --- Load-Balanced Partition ---
///   Custom assignment based on computational weight.
///   Higher indices have more work, so give fewer to later ranks.
///
///   SUCCESS: Custom partition example completed!

#include <dtl/dtl.hpp>

#include <iostream>
#include <vector>

// Helper to determine which rank owns a global index under block partition
dtl::rank_t block_owner(dtl::index_t global_idx, dtl::size_type global_size, dtl::rank_t num_ranks) {
    dtl::size_type base_size = global_size / static_cast<dtl::size_type>(num_ranks);
    dtl::size_type remainder = global_size % static_cast<dtl::size_type>(num_ranks);

    // Ranks 0..remainder-1 get base_size+1 elements
    // Ranks remainder..num_ranks-1 get base_size elements
    dtl::size_type threshold = remainder * (base_size + 1);

    if (static_cast<dtl::size_type>(global_idx) < threshold) {
        return static_cast<dtl::rank_t>(static_cast<dtl::size_type>(global_idx) / (base_size + 1));
    } else {
        return static_cast<dtl::rank_t>(remainder +
            (static_cast<dtl::size_type>(global_idx) - threshold) / base_size);
    }
}

// Helper to determine which rank owns under cyclic partition
dtl::rank_t cyclic_owner(dtl::index_t global_idx, dtl::rank_t num_ranks) {
    return static_cast<dtl::rank_t>(global_idx % num_ranks);
}

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();

    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    // Extract communicator for barrier operations
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    const dtl::size_type global_size = 100;

    if (my_rank == 0) {
        std::cout << "DTL Custom Partition Example\n";
        std::cout << "============================\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Global size: " << global_size << "\n";
        std::cout << "  Number of ranks: " << num_ranks << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // Block Partition (DTL default)
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "--- Block Partition (Default) ---\n";
        std::cout << "Contiguous chunks assigned to each rank.\n";
    }

    dtl::distributed_vector<int> block_vec(global_size, ctx);

    comm.barrier();

    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Rank " << r << ": owns indices ["
                      << block_vec.global_offset() << ", "
                      << (block_vec.global_offset() + block_vec.local_size()) << ")\n";
        }
        comm.barrier();
    }

    // =========================================================================
    // Simulated Cyclic Access Pattern
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Simulated Cyclic Access Pattern ---\n";
        std::cout << "Round-robin assignment (index i -> rank i % num_ranks).\n";
    }

    comm.barrier();

    // Show cyclic ownership mapping
    if (my_rank == 0) {
        std::cout << "Index 0 -> Rank " << cyclic_owner(0, num_ranks)
                  << ", Index 1 -> Rank " << cyclic_owner(1, num_ranks)
                  << ", ...\n";
        std::cout << "Index 4 -> Rank " << cyclic_owner(4, num_ranks)
                  << ", Index 5 -> Rank " << cyclic_owner(5, num_ranks)
                  << ", ...\n";
    }

    // Count how many elements each rank would own under cyclic
    dtl::size_type cyclic_count = 0;
    for (dtl::size_type i = 0; i < global_size; ++i) {
        if (cyclic_owner(static_cast<dtl::index_t>(i), num_ranks) == my_rank) {
            cyclic_count++;
        }
    }

    comm.barrier();

    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            std::cout << "Rank " << r << " would own " << cyclic_count
                      << " elements under cyclic partition\n";
        }
        comm.barrier();
    }

    // =========================================================================
    // Load-Balanced Partition
    // =========================================================================
    if (my_rank == 0) {
        std::cout << "\n--- Load-Balanced Partition ---\n";
        std::cout << "Custom assignment based on computational weight.\n";
        std::cout << "Higher indices have more work, so give fewer to later ranks.\n";
    }

    // Simulate a workload where work(i) = i (linear increase)
    // Total work = 0 + 1 + ... + (N-1) = N*(N-1)/2
    // Target: each rank gets equal total work

    std::vector<dtl::size_type> rank_boundaries(static_cast<size_t>(num_ranks) + 1);
    rank_boundaries[0] = 0;

    double total_work = static_cast<double>(global_size) * static_cast<double>(global_size - 1) / 2.0;
    double work_per_rank = total_work / static_cast<double>(num_ranks);

    double accumulated_work = 0.0;
    dtl::rank_t current_rank = 0;

    for (dtl::size_type i = 0; i < global_size && current_rank < num_ranks - 1; ++i) {
        accumulated_work += static_cast<double>(i);
        if (accumulated_work >= work_per_rank * static_cast<double>(current_rank + 1)) {
            rank_boundaries[static_cast<size_t>(current_rank) + 1] = i;
            current_rank++;
        }
    }
    rank_boundaries[static_cast<size_t>(num_ranks)] = global_size;

    comm.barrier();

    // Show load-balanced boundaries
    for (dtl::rank_t r = 0; r < num_ranks; ++r) {
        if (my_rank == r) {
            dtl::size_type start = rank_boundaries[static_cast<size_t>(r)];
            dtl::size_type end = rank_boundaries[static_cast<size_t>(r) + 1];
            dtl::size_type count = end - start;

            // Compute total work for this rank
            double work = 0.0;
            for (dtl::size_type i = start; i < end; ++i) {
                work += static_cast<double>(i);
            }

            std::cout << "Rank " << r << ": indices [" << start << ", " << end
                      << "), " << count << " elements, work = " << work << "\n";
        }
        comm.barrier();
    }

    if (my_rank == 0) {
        std::cout << "\nSUCCESS: Custom partition example completed!\n";
    }

    return 0;
}
