// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rma_put_get.cpp
/// @brief RMA Put/Get with fence synchronization using DTL
///
/// Demonstrates:
/// - memory_window::create for exposing local buffers
/// - dtl::rma::put / dtl::rma::get for one-sided data transfer
/// - window.fence() for active-target synchronization
///
/// Note: RMA may fail on WSL2 OpenMPI 4.1.6 due to MPI_Win_create issues.
///       This example includes error handling for graceful degradation.
///
/// Run:
///   mpirun -np 2 ./rma_put_get

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
        std::cout << "DTL RMA Put/Get Example (Fence Sync)\n";
        std::cout << "======================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }
    comm.barrier();

    // Each rank exposes a buffer via RMA window
    const dtl::size_type buf_size = 4;
    std::vector<int> local_buf(buf_size, rank * 100);

    auto win_result = dtl::memory_window::create(local_buf.data(),
                                                  buf_size * sizeof(int));
    if (!win_result) {
        std::cerr << "Rank " << rank
                  << ": Failed to create window (RMA may not be supported)\n";
        return 1;
    }

    auto& window = win_result.value();

    // Initial fence to open epoch
    auto fence_result = window.fence();
    if (!fence_result) {
        std::cerr << "Rank " << rank << ": Fence failed\n";
        return 1;
    }

    // Rank 0 puts value 999 into rank 1's buffer at offset 0
    if (rank == 0) {
        int value = 999;
        auto put_result = dtl::rma::put(
            static_cast<dtl::rank_t>(1),    // target rank
            0,                               // target offset (bytes)
            &value,                          // source data
            sizeof(int),                     // size
            window                           // window
        );
        if (put_result) {
            std::cout << "Rank 0: Put value " << value << " to rank 1\n";
        } else {
            std::cerr << "Rank 0: Put failed\n";
        }
    }

    // Fence to complete the put
    auto fence_after_put = window.fence();
    const bool fence_after_put_ok = static_cast<bool>(fence_after_put);
    const bool all_fence_after_put_ok = comm.allreduce_land_value(fence_after_put_ok);
    if (!all_fence_after_put_ok) {
        if (rank == 0) {
            std::cerr << "Fence after put failed\n";
        }
        return 1;
    }

    // Rank 1 prints the received value
    if (rank == 1) {
        std::cout << "Rank 1: buffer[0] = " << local_buf[0]
                  << " (expected: 999)\n";
    }
    comm.barrier();

    // Now rank 1 reads back from rank 0
    int read_val = -1;
    if (rank == 1) {
        auto get_result = dtl::rma::get(
            static_cast<dtl::rank_t>(0),    // target rank
            0,                               // target offset
            &read_val,                       // local buffer
            sizeof(int),                     // size
            window                           // window
        );
        if (get_result) {
            std::cout << "Rank 1: Got value " << read_val << " from rank 0\n";
        }
    }

    // Final fence
    auto final_fence = window.fence();
    const bool final_fence_ok = static_cast<bool>(final_fence);
    const bool all_final_fence_ok = comm.allreduce_land_value(final_fence_ok);
    if (!all_final_fence_ok) {
        if (rank == 0) {
            std::cerr << "Final fence failed\n";
        }
        return 1;
    }
    comm.barrier();

    // Verify
    bool ok = true;
    if (rank == 1) {
        ok = (local_buf[0] == 999) && (read_val == 0);
    }
    bool all_ok = comm.allreduce_land_value(ok);

    if (rank == 0) {
        std::cout << "\n" << (all_ok ? "SUCCESS" : "FAILURE") << "\n";
    }

    return 0;
}
