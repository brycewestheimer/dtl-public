// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_array_example.cpp
/// @brief Demonstrates distributed_array usage with compile-time fixed size
/// @details Shows how DTL's distributed_array provides a fixed-size distributed
///          container (analogous to std::array) where the total element count is
///          a compile-time constant. Unlike distributed_vector, the size cannot
///          change after construction.
///
///          Key concepts demonstrated:
///          - Creating a distributed_array with compile-time size
///          - Using local_view() for STL-compatible local access
///          - Using segmented_view() for distributed iteration
///          - Fill operations
///          - Index mapping: global-to-local and local-to-global
///          - Distribution queries: is_local(), owner()
///          - Compile-time extent access
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./distributed_array_example
///
/// Run (multiple ranks):
///   mpirun -np 4 ./distributed_array_example

#include <dtl/dtl.hpp>

#include <iostream>
#include <numeric>
#include <algorithm>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Distributed Array Example\n";
        std::cout << "=============================\n\n";
        std::cout << "Number of ranks: " << size << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // 1. Create a distributed array with compile-time size
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 1. Compile-Time Fixed Size ---\n";
    }
    comm.barrier();

    // N = 100 is known at compile time
    constexpr dtl::size_type N = 100;
    dtl::distributed_array<int, N> arr(ctx);

    if (rank == 0) {
        // Compile-time extent access
        std::cout << "Array extent (compile-time): "
                  << dtl::distributed_array<int, N>::extent << "\n";
        std::cout << "Array size(): " << arr.size() << "\n";
        std::cout << "Array global_size(): " << arr.global_size() << "\n";
        std::cout << "Array empty(): " << std::boolalpha << arr.empty() << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 2. Local partition information
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 2. Local Partition Info ---\n";
    }
    comm.barrier();

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": local_size=" << arr.local_size()
                      << ", global_offset=" << arr.global_offset() << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 3. Fill and local_view() access
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 3. Fill and Local View ---\n";
    }
    comm.barrier();

    // Fill all elements with a value
    arr.fill(42);

    auto local = arr.local_view();

    if (rank == 0) {
        std::cout << "After fill(42), first 10: ";
        for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // Use std::iota via local_view iterators
    dtl::index_t start_val = arr.global_offset();
    std::iota(local.begin(), local.end(), static_cast<int>(start_val));

    if (rank == 0) {
        std::cout << "After std::iota from global offset, first 10: ";
        for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 4. STL algorithms on local_view
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 4. STL Algorithms ---\n";
    }
    comm.barrier();

    // Transform: square each element
    std::transform(local.begin(), local.end(), local.begin(),
                   [](int x) { return x * x; });

    int local_sum = std::accumulate(local.begin(), local.end(), 0);
    int global_sum = comm.allreduce_sum_value<int>(local_sum);

    if (rank == 0) {
        std::cout << "After squaring, first 10: ";
        for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "\n";
        std::cout << "Global sum of squares: " << global_sum << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 5. Index mapping: local <-> global
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 5. Index Mapping ---\n";
    }
    comm.barrier();

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": local[0] -> global["
                      << arr.to_global(0) << "]";
            if (arr.local_size() > 0) {
                dtl::index_t last_local = static_cast<dtl::index_t>(arr.local_size() - 1);
                std::cout << ", local[" << last_local << "] -> global["
                          << arr.to_global(last_local) << "]";
            }
            std::cout << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 6. Distribution queries: is_local(), owner()
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 6. Distribution Queries ---\n";
        for (dtl::index_t g = 0; g < static_cast<dtl::index_t>(N); g += static_cast<dtl::index_t>(N) / 5) {
            std::cout << "Global[" << g << "]: owner=" << arr.owner(g)
                      << ", is_local=" << std::boolalpha << arr.is_local(g) << "\n";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 7. Segmented view iteration
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 7. Segmented View ---\n";
    }
    comm.barrier();

    // Restore sequential values for clearer output
    std::iota(local.begin(), local.end(), static_cast<int>(arr.global_offset()));

    auto seg_view = arr.segmented_view();

    if (rank == 0) {
        std::cout << "Total segments: " << seg_view.num_segments() << "\n";
        std::cout << "Total size: " << seg_view.total_size() << "\n";

        for (auto segment : seg_view) {
            std::cout << "  Segment rank=" << segment.rank
                      << ": offset=" << segment.global_offset
                      << ", size=" << segment.size()
                      << ", local=" << std::boolalpha << segment.is_local() << "\n";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 8. Construct with initial value
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 8. Construction with Initial Value ---\n";
    }
    comm.barrier();

    dtl::distributed_array<double, 50> arr2(3.14, ctx);

    if (rank == 0) {
        auto lv2 = arr2.local_view();
        std::cout << "distributed_array<double, 50> initialized with 3.14\n";
        std::cout << "First 5 elements: ";
        for (dtl::size_type i = 0; i < 5 && i < lv2.size(); ++i) {
            std::cout << lv2[i] << " ";
        }
        std::cout << "\n";
    }

    comm.barrier();
    if (rank == 0) {
        std::cout << "\nSUCCESS: Distributed array example completed!\n";
    }

    return 0;
}
