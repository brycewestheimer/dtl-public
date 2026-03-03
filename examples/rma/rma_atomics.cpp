// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rma_atomics.cpp
/// @brief Advanced RMA operations: async put/get, flush, and lock_all
///
/// Demonstrates:
/// - Async RMA with put_async/get_async and request handles
/// - Flush-based completion (flush, flush_local)
/// - lock_all/unlock_all for global passive-target epoch
///
/// Note: RMA may fail on WSL2 OpenMPI 4.1.6 (known MPI_Win_create issue).
///
/// Run:
///   mpirun -np 4 ./rma_atomics

#include <dtl/dtl.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/communication/rma_operations.hpp>

#include <iostream>
#include <vector>
#include <numeric>

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
        std::cout << "DTL Advanced RMA Example\n";
        std::cout << "=========================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }
    comm.barrier();

    // Each rank exposes a buffer of 4 ints
    constexpr int BUF_SIZE = 4;
    std::vector<int> local_buf(BUF_SIZE);
    for (int i = 0; i < BUF_SIZE; ++i) {
        local_buf[static_cast<std::size_t>(i)] = rank * 100 + i;
    }

    auto win_result = dtl::memory_window::create(
        local_buf.data(),
        static_cast<dtl::size_type>(BUF_SIZE * sizeof(int))
    );
    if (!win_result) {
        std::cerr << "Rank " << rank << ": Window creation failed\n";
        return 1;
    }
    auto& window = win_result.value();

    // --- 1. Async put with flush ---
    if (rank == 0) std::cout << "1. Async put with flush:\n";
    comm.barrier();

    // Use lock_all / unlock_all for global passive-target epoch
    auto lock_res = window.lock_all();
    if (!lock_res) {
        std::cerr << "Rank " << rank << ": lock_all failed\n";
        return 1;
    }

    // Rank 0 writes to rank 1's buffer[0] asynchronously
    if (rank == 0) {
        int put_val = 9999;
        auto put_res = dtl::rma::put(
            static_cast<dtl::rank_t>(1),
            static_cast<dtl::size_type>(0),
            put_val,
            window
        );

        if (put_res) {
            // Flush to ensure the put is visible at target
            auto flush_res = dtl::rma::flush(static_cast<dtl::rank_t>(1), window);
            if (flush_res) {
                std::cout << "  Rank 0: put " << put_val << " to rank 1, flushed OK\n";
            } else {
                std::cout << "  Rank 0: put succeeded but flush failed\n";
            }
        } else {
            std::cout << "  Rank 0: put failed\n";
        }
    }

    auto unlock_res = window.unlock_all();
    if (!unlock_res) {
        std::cerr << "Rank " << rank << ": unlock_all failed\n";
    }
    comm.barrier();

    if (rank == 1) {
        std::cout << "  Rank 1: buffer[0] = " << local_buf[0]
                  << " (expected 9999)\n";
    }
    comm.barrier();

    // --- 2. Async get with flush_local ---
    if (rank == 0) std::cout << "\n2. Get with flush_local:\n";
    comm.barrier();

    // Reset rank 0's buffer
    if (rank == 0) {
        for (int i = 0; i < BUF_SIZE; ++i) {
            local_buf[static_cast<std::size_t>(i)] = (i + 1) * 111;
        }
    }

    // Fence to make rank 0's new data visible
    auto fence_res = window.fence();
    if (!fence_res) {
        std::cerr << "Rank " << rank << ": fence failed\n";
    }

    // Rank 1 reads rank 0's buffer[2] using passive-target lock
    if (rank == 1) {
        auto lock_r = window.lock(static_cast<dtl::rank_t>(0));
        if (lock_r) {
            int read_val = -1;
            auto get_res = dtl::rma::get(
                static_cast<dtl::rank_t>(0),
                static_cast<dtl::size_type>(2 * sizeof(int)),
                read_val,
                window
            );

            if (get_res) {
                // flush_local to ensure local completion
                auto fl_res = dtl::rma::flush_local(
                    static_cast<dtl::rank_t>(0), window);
                (void)fl_res;

                std::cout << "  Rank 1: got value " << read_val
                          << " from rank 0 buffer[2] (expected 333)\n";
            } else {
                std::cout << "  Rank 1: get failed\n";
            }
            auto unlock_r = window.unlock(static_cast<dtl::rank_t>(0));
            (void)unlock_r;
        } else {
            std::cout << "  Rank 1: lock(0) failed\n";
        }
    }
    comm.barrier();

    // --- 3. Round-robin put via fence ---
    if (rank == 0) std::cout << "\n3. Round-robin put via fence:\n";

    // Each rank puts its rank value to the next rank's buffer[3]
    auto fence_r1 = window.fence();
    (void)fence_r1;

    dtl::rank_t next_rank = static_cast<dtl::rank_t>((rank + 1) % size);
    int my_tag = rank * 10 + 7;
    auto rr_put = dtl::rma::put(
        next_rank,
        static_cast<dtl::size_type>(3 * sizeof(int)),
        my_tag,
        window
    );
    if (!rr_put) {
        std::cerr << "Rank " << rank << ": round-robin put failed\n";
    }

    auto fence_r2 = window.fence();
    (void)fence_r2;

    // Verify: each rank should have received from (rank-1+size)%size
    dtl::rank_t prev_rank = static_cast<dtl::rank_t>((rank + size - 1) % size);
    int expected_tag = prev_rank * 10 + 7;
    bool ok = (local_buf[3] == expected_tag);
    std::cout << "  Rank " << rank << ": buffer[3] = " << local_buf[3]
              << " (from rank " << prev_rank
              << ", expected " << expected_tag << ")"
              << (ok ? " OK" : " FAIL") << "\n";

    comm.barrier();
    if (rank == 0) std::cout << "\nDone!\n";

    return 0;
}
