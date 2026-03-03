// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rma_passive_target.cpp
/// @brief Passive-target RMA with lock/unlock using DTL
///
/// Demonstrates:
/// - window.lock() / window.unlock() for passive-target epochs
/// - One-sided reads without target participation
/// - RAII-style resource management
///
/// Note: Passive-target RMA may not work on all MPI implementations.
///
/// Run:
///   mpirun -np 4 ./rma_passive_target

#include <dtl/dtl.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/communication/rma_operations.hpp>

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (size < 2) {
        if (rank == 0) std::cout << "This example requires at least 2 ranks.\n";
        return 1;
    }

    if (rank == 0) {
        std::cout << "DTL RMA Passive-Target Example\n";
        std::cout << "================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }
    comm.barrier();

    // Each rank exposes a buffer with rank-specific data
    int local_val = (rank + 1) * 111;  // 111, 222, 333, 444, ...
    auto win_result = dtl::memory_window::create(&local_val, sizeof(int));
    if (!win_result) {
        std::cerr << "Rank " << rank << ": Window creation failed\n";
        return 1;
    }

    auto& window = win_result.value();
    comm.barrier();

    // Rank 0 reads from each other rank using passive-target locks
    if (rank == 0) {
        std::cout << "Rank 0 reading from each target rank:\n";

        for (dtl::rank_t target = 1; target < size; ++target) {
            // Lock the target for shared (read) access
            auto lock_result = window.lock(target, dtl::rma_lock_mode::shared);
            if (!lock_result) {
                std::cerr << "  Failed to lock rank " << target << "\n";
                continue;
            }

            // Read from the target
            int remote_val = 0;
            auto get_result = dtl::rma::get(
                target, 0, &remote_val, sizeof(int), window
            );

            // Flush to ensure completion before unlock
            (void)window.flush(target);

            // Unlock
            (void)window.unlock(target);

            if (get_result) {
                int expected = (target + 1) * 111;
                std::cout << "  Rank " << target << ": read " << remote_val
                          << " (expected: " << expected << ")"
                          << (remote_val == expected ? " OK" : " FAIL") << "\n";
            } else {
                std::cerr << "  Rank " << target << ": get failed\n";
            }
        }
    }

    comm.barrier();
    if (rank == 0) std::cout << "\nDone!\n";

    return 0;
}
