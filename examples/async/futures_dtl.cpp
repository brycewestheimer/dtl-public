// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file futures_dtl.cpp
/// @brief Distributed futures and promises using DTL
///
/// Demonstrates:
/// - dtl::distributed_future / distributed_promise
/// - Waiting with timeout (wait_for)
/// - Multiple futures with manual synchronization
///
/// Note: DTL continuation chaining (.then() / fmap) has known stability
///       issues. This example uses basic promise/future patterns.
///
/// Run:
///   mpirun -np 4 ./futures_dtl

#include <dtl/dtl.hpp>
#include <dtl/futures/distributed_future.hpp>

#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Distributed Futures Example\n";
        std::cout << "================================\n";
        std::cout << "Ranks: " << size << "\n\n";
    }

    comm.barrier();

    // --- 1. Basic promise/future ---
    {
        if (rank == 0) std::cout << "1. Basic promise/future:\n";

        auto promise = dtl::distributed_promise<int>();
        auto future = promise.get_future();

        // Fulfill the promise with rank-specific data
        int value = (rank + 1) * 100;
        promise.set_value(value);

        // Wait and get the result
        future.wait();
        int result = future.get();

        for (dtl::rank_t r = 0; r < size; ++r) {
            if (rank == r) {
                std::cout << "  Rank " << rank << ": future resolved to " << result << "\n";
            }
            comm.barrier();
        }
    }

    // --- 2. Wait with timeout ---
    {
        if (rank == 0) std::cout << "\n2. Wait with timeout:\n";
        comm.barrier();

        auto promise = dtl::distributed_promise<double>();
        auto future = promise.get_future();

        // Check before setting - should timeout
        auto status = future.wait_for(std::chrono::milliseconds(10));
        if (rank == 0) {
            std::cout << "  Before set: "
                      << (status == dtl::future_status::timeout ? "timeout" : "ready")
                      << " (expected: timeout)\n";
        }

        // Set the value
        promise.set_value(3.14 * (rank + 1));

        // Now wait should succeed immediately
        status = future.wait_for(std::chrono::milliseconds(1000));
        double result = future.get();

        for (dtl::rank_t r = 0; r < size; ++r) {
            if (rank == r) {
                std::cout << "  Rank " << rank << ": got " << result
                          << " (status: " << (status == dtl::future_status::ready ? "ready" : "other")
                          << ")\n";
            }
            comm.barrier();
        }
    }

    // --- 3. Multiple independent futures ---
    {
        if (rank == 0) std::cout << "\n3. Multiple independent futures:\n";
        comm.barrier();

        constexpr int N = 3;
        std::vector<dtl::distributed_promise<int>> promises;
        std::vector<dtl::distributed_future<int>> futures;

        for (int i = 0; i < N; ++i) {
            promises.emplace_back();
            futures.push_back(promises.back().get_future());
        }

        // Fulfill all promises
        for (int i = 0; i < N; ++i) {
            promises[i].set_value(rank * 100 + i);
        }

        // Wait for all and collect results
        int sum = 0;
        for (auto& f : futures) {
            f.wait();
            sum += f.get();
        }

        for (dtl::rank_t r = 0; r < size; ++r) {
            if (rank == r) {
                int expected = rank * 100 * N + (N * (N - 1)) / 2;
                std::cout << "  Rank " << rank << ": sum of " << N
                          << " futures = " << sum
                          << " (expected: " << expected << ")"
                          << (sum == expected ? " OK" : " FAIL") << "\n";
            }
            comm.barrier();
        }
    }

    // --- 4. Ready future factory ---
    {
        if (rank == 0) std::cout << "\n4. Ready future factory:\n";
        comm.barrier();

        auto ready = dtl::make_ready_distributed_future(rank * 42);

        if (ready.is_ready()) {
            int val = ready.get();
            for (dtl::rank_t r = 0; r < size; ++r) {
                if (rank == r) {
                    std::cout << "  Rank " << rank << ": ready future = " << val
                              << " (expected: " << (rank * 42) << ")"
                              << (val == rank * 42 ? " OK" : " FAIL") << "\n";
                }
                comm.barrier();
            }
        }
    }

    if (rank == 0) std::cout << "\nDone!\n";

    return 0;
}
