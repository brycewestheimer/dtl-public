// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file local_view_stl.cpp
/// @brief Demonstrates using local_view with standard STL algorithms
/// @details Shows how DTL's local_view provides STL-compatible iterators,
///          enabling seamless use of standard algorithms on local data.
///          Uses dtl::environment and make_world_context() for initialization.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./local_view_stl
///
/// Run (multiple ranks):
///   mpirun -np 4 ./local_view_stl
///
/// Expected output (single rank):
///   DTL Local View + STL Algorithms Example
///
///   Created distributed_vector with 100 elements on 1 rank(s)
///   Rank 0: local partition has 100 elements
///
///   --- Using STL algorithms on local_view ---
///
///   1. std::iota - filling with sequence:
///      First 10 elements: 0 1 2 3 4 5 6 7 8 9
///
///   2. std::transform - squaring elements:
///      First 10 elements: 0 1 4 9 16 25 36 49 64 81
///
///   3. std::accumulate - computing sum:
///      Sum of all elements: 328350
///
///   4. std::find_if - finding first element > 50:
///      First element > 50: 64 at index 8
///
///   5. std::count_if - counting elements > 100:
///      Elements > 100: 89
///
///   6. std::sort (descending):
///      First 10 elements: 9801 9604 9409 9216 9025 8836 8649 8464 8281 8100
///
///   7. std::binary_search - checking for 4900 (70^2):
///      4900 found: true
///
///   SUCCESS: All STL algorithms worked correctly!

#include <dtl/dtl.hpp>

#include <iostream>
#include <numeric>
#include <algorithm>

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();

    auto my_rank = ctx.rank();
    auto num_ranks = ctx.size();

    if (my_rank == 0) {
        std::cout << "DTL Local View + STL Algorithms Example\n\n";
    }

    // Create distributed vector
    const dtl::size_type global_size = 100;
    dtl::distributed_vector<int> vec(global_size, ctx);

    if (my_rank == 0) {
        std::cout << "Created distributed_vector with " << global_size
                  << " elements on " << num_ranks << " rank(s)\n";
    }

    comm.barrier();

    std::cout << "Rank " << my_rank << ": local partition has "
              << vec.local_size() << " elements\n";

    comm.barrier();

    if (my_rank == 0) {
        std::cout << "\n--- Using STL algorithms on local_view ---\n\n";
    }

    // Get local view - this is STL-compatible!
    auto local = vec.local_view();

    // 1. std::iota - fill with sequential values
    dtl::index_t start_val = static_cast<dtl::index_t>(vec.global_offset());
    std::iota(local.begin(), local.end(), static_cast<int>(start_val));

    if (my_rank == 0) {
        std::cout << "1. std::iota - filling with sequence:\n";
        std::cout << "   First 10 elements: ";
        for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "\n\n";
    }

    // 2. std::transform - square each element
    std::transform(local.begin(), local.end(), local.begin(),
                   [](int x) { return x * x; });

    if (my_rank == 0) {
        std::cout << "2. std::transform - squaring elements:\n";
        std::cout << "   First 10 elements: ";
        for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "\n\n";
    }

    // 3. std::accumulate - compute sum
    int local_sum = std::accumulate(local.begin(), local.end(), 0);

    int global_sum = comm.allreduce_sum_value<int>(local_sum);

    if (my_rank == 0) {
        std::cout << "3. std::accumulate - computing sum:\n";
        std::cout << "   Sum of all elements: " << global_sum << "\n\n";
    }

    // 4. std::find_if - find first element > 50
    auto it = std::find_if(local.begin(), local.end(),
                           [](int x) { return x > 50; });

    if (my_rank == 0 && it != local.end()) {
        auto idx = std::distance(local.begin(), it);
        std::cout << "4. std::find_if - finding first element > 50:\n";
        std::cout << "   First element > 50: " << *it
                  << " at index " << idx << "\n\n";
    }

    // 5. std::count_if - count elements > 100
    auto local_count = std::count_if(local.begin(), local.end(),
                                     [](int x) { return x > 100; });

    long global_count = comm.allreduce_sum_value<long>(static_cast<long>(local_count));

    if (my_rank == 0) {
        std::cout << "5. std::count_if - counting elements > 100:\n";
        std::cout << "   Elements > 100: " << global_count << "\n\n";
    }

    // 6. std::sort - sort in descending order
    std::sort(local.begin(), local.end(), std::greater<int>{});

    if (my_rank == 0) {
        std::cout << "6. std::sort (descending):\n";
        std::cout << "   First 10 elements: ";
        for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "\n\n";
    }

    // 7. std::binary_search - search for specific value
    // Note: Must sort ascending first for binary_search
    std::sort(local.begin(), local.end());
    bool found = std::binary_search(local.begin(), local.end(), 4900); // 70^2

    if (my_rank == 0) {
        std::cout << "7. std::binary_search - checking for 4900 (70^2):\n";
        std::cout << "   4900 found: " << std::boolalpha << found << "\n\n";
    }

    if (my_rank == 0) {
        std::cout << "SUCCESS: All STL algorithms worked correctly!\n";
    }

    return 0;
}
