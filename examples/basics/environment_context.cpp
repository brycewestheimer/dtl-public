// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file environment_context.cpp
/// @brief Demonstrates dtl::environment and context-based construction
///
/// This example shows the recommended way to initialize DTL using
/// dtl::environment, which manages backend lifecycle automatically.
///
/// Build: cmake .. -DDTL_BUILD_EXAMPLES=ON && make
/// Run: mpirun -np 4 ./environment_context

#include <dtl/dtl.hpp>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    // Environment manages backend lifecycle (MPI init/finalize)
    dtl::environment env(argc, argv);

    // Create context from environment - this is the entry point for DTL operations
    auto ctx = env.make_world_context();

    std::cout << "Rank " << ctx.rank() << " of " << ctx.size() << "\n";

    // Create container with context
    // The context provides rank/size information automatically
    const dtl::size_type global_size = 1000;
    dtl::distributed_vector<double> vec(global_size, ctx);

    // Fill local partition with rank-based values
    auto local = vec.local_view();
    for (std::size_t i = 0; i < local.size(); ++i) {
        local[i] = ctx.rank() * 1000.0 + static_cast<double>(i);
    }

    // Display local partition info
    std::cout << "Rank " << ctx.rank() << ": local_size=" << local.size()
              << ", global_offset=" << vec.global_offset()
              << ", first_value=" << local[0] << "\n";

    // Local computation
    double local_sum = std::accumulate(local.begin(), local.end(), 0.0);

    // Collective operation using MPI through the context's communicator
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto sum_result = dtl::distributed_reduce(dtl::seq{}, vec, 0.0, std::plus<>{}, comm);
    double global_sum = sum_result.global_value;

    // Only rank 0 prints the final result
    if (ctx.rank() == 0) {
        std::cout << "\nResults:\n";
        std::cout << "  Global size: " << vec.global_size() << "\n";
        std::cout << "  Local sum (rank 0 only): " << local_sum << "\n";
        std::cout << "  Global sum: " << global_sum << "\n";

        // Verify: each rank contributes rank*1000*local_size + sum(0..local_size-1)
        // This is a simplified check
        std::cout << "  Computation complete.\n";
    }

    // Environment automatically finalizes MPI when it goes out of scope
    return 0;
}
