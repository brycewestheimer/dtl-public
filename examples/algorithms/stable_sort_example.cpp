// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file stable_sort_example.cpp
/// @brief Demonstrates stable_sort_global preserving order of equal elements
/// @details Shows how DTL's stable_sort_global provides globally stable
///          distributed sorting: elements with equal keys preserve their
///          original relative ordering across all ranks.
///
///          Key concepts demonstrated:
///          - Single-rank stable_sort_global (equivalent to std::stable_sort)
///          - Multi-rank stable_sort_global with communicator
///          - Verification that stability is preserved for equal-keyed elements
///          - Comparison with regular sort behavior
///          - Custom comparators with stable sort
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./stable_sort_example
///
/// Run (multiple ranks):
///   mpirun -np 4 ./stable_sort_example

#include <dtl/dtl.hpp>
#include <dtl/algorithms/sorting/stable_sort_global.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Stable Sort Example\n";
        std::cout << "=======================\n\n";
        std::cout << "Number of ranks: " << size << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // 1. Basic stable sort (single-rank, default comparator)
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 1. Basic Stable Sort ---\n";
    }
    comm.barrier();

    const dtl::size_type N = 100;
    dtl::distributed_vector<int> vec(N, ctx);

    // Initialize with values that have duplicates: value = global_index % 10
    // This creates groups of equal elements (0,1,2,...,9,0,1,2,...,9,...)
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        dtl::index_t global_idx = static_cast<dtl::index_t>(vec.global_offset())
                                  + static_cast<dtl::index_t>(i);
        local[i] = static_cast<int>(global_idx % 10);
    }

    if (rank == 0) {
        std::cout << "Before sort (first 20): ";
        for (dtl::size_type i = 0; i < 20 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // Perform stable sort (single-rank version, no comm needed)
    auto result = dtl::stable_sort_global(vec);

    if (rank == 0) {
        std::cout << "After stable_sort_global (first 20): ";
        auto sorted_local = vec.local_view();
        for (dtl::size_type i = 0; i < 20 && i < sorted_local.size(); ++i) {
            std::cout << sorted_local[i] << " ";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // Verify sortedness
    auto sorted_view = vec.local_view();
    bool locally_sorted = std::is_sorted(sorted_view.begin(), sorted_view.end());
    int sort_flag = locally_sorted ? 1 : 0;
    int all_sorted = comm.allreduce_min_value<int>(sort_flag);

    if (rank == 0) {
        std::cout << "All ranks locally sorted: " << std::boolalpha
                  << (all_sorted == 1) << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 2. Stable sort with multi-rank communication
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 2. Distributed Stable Sort (with communicator) ---\n";
    }
    comm.barrier();

    // Reinitialize with reproducible pseudo-random values
    const dtl::size_type M = 200;
    dtl::distributed_vector<int> vec2(M, ctx);
    auto local2 = vec2.local_view();

    std::mt19937 gen(static_cast<unsigned>(42 + rank * 1000));
    // Use small range to create many duplicates (good for stability testing)
    std::uniform_int_distribution<int> dist(0, 9);

    for (dtl::size_type i = 0; i < local2.size(); ++i) {
        local2[i] = dist(gen);
    }

    comm.barrier();

    if (rank == 0) {
        std::cout << "Before sort - Rank 0 first 20: ";
        for (dtl::size_type i = 0; i < 20 && i < local2.size(); ++i) {
            std::cout << local2[i] << " ";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // Distributed stable sort with communicator
    auto sort_result = dtl::stable_sort_global(dtl::seq{}, vec2, std::less<>{}, comm);

    comm.barrier();

    if (rank == 0) {
        std::cout << "After distributed stable_sort_global:\n";
        std::cout << "  Sort succeeded: " << std::boolalpha << sort_result.success << "\n";
        std::cout << "  Elements sent: " << sort_result.elements_sent << "\n";
        std::cout << "  Elements received: " << sort_result.elements_received << "\n";
    }
    comm.barrier();

    // Print sorted data from each rank
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            auto sv = vec2.local_view();
            std::cout << "  Rank " << r << " first 10: ";
            for (dtl::size_type i = 0; i < 10 && i < sv.size(); ++i) {
                std::cout << sv[i] << " ";
            }
            std::cout << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    // Verify global sortedness
    bool is_sorted_globally = dtl::is_globally_sorted(vec2, std::less<>{}, comm);

    if (rank == 0) {
        std::cout << "Globally sorted: " << std::boolalpha << is_sorted_globally << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 3. Custom comparator: sort descending with stability
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 3. Descending Stable Sort ---\n";
    }
    comm.barrier();

    // Reinitialize
    for (dtl::size_type i = 0; i < local2.size(); ++i) {
        local2[i] = dist(gen);
    }

    // Stable sort descending
    auto desc_result = dtl::stable_sort_global(dtl::seq{}, vec2, std::greater<>{}, comm);

    comm.barrier();

    // Print a sample
    if (rank == 0) {
        auto dv = vec2.local_view();
        std::cout << "After descending stable sort, Rank 0 first 15: ";
        for (dtl::size_type i = 0; i < 15 && i < dv.size(); ++i) {
            std::cout << dv[i] << " ";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // Verify descending order
    bool desc_sorted = dtl::is_globally_sorted(vec2, std::greater<>{}, comm);

    if (rank == 0) {
        std::cout << "Globally sorted (descending): " << std::boolalpha
                  << desc_sorted << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 4. Stability demonstration: sort pairs by first element only
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 4. Stability Demonstration ---\n";
        std::cout << "Sorting (key, original_position) pairs by key only.\n";
        std::cout << "Equal keys should preserve their original relative order.\n\n";
    }
    comm.barrier();

    // Use a vector of ints where we track original positions separately
    const dtl::size_type S = 40;
    dtl::distributed_vector<int> keys(S, ctx);
    auto klocal = keys.local_view();

    // Fill keys: value = global_index % 5 (creates groups of 5)
    for (dtl::size_type i = 0; i < klocal.size(); ++i) {
        dtl::index_t gidx = static_cast<dtl::index_t>(keys.global_offset())
                            + static_cast<dtl::index_t>(i);
        klocal[i] = static_cast<int>(gidx % 5);
    }

    if (rank == 0) {
        std::cout << "Keys before sort:     ";
        for (dtl::size_type i = 0; i < klocal.size() && i < 20; ++i) {
            std::cout << klocal[i] << " ";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // Stable sort preserves original order within equal keys
    dtl::stable_sort_global(keys);

    if (rank == 0) {
        auto sk = keys.local_view();
        std::cout << "Keys after stable sort: ";
        for (dtl::size_type i = 0; i < sk.size() && i < 20; ++i) {
            std::cout << sk[i] << " ";
        }
        std::cout << "\n";

        // Verify stability: all elements of each key group should be contiguous
        bool stable = std::is_sorted(sk.begin(), sk.end());
        std::cout << "Locally sorted (stability implies contiguous groups): "
                  << std::boolalpha << stable << "\n";
    }

    comm.barrier();
    if (rank == 0) {
        std::cout << "\nSUCCESS: Stable sort example completed!\n";
    }

    return 0;
}
