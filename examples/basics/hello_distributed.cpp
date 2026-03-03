// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hello_distributed.cpp
/// @brief Basic distributed vector example (standalone mode, no MPI required)
/// @details Demonstrates DTL's distributed_vector in single-rank standalone mode.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   ./hello_distributed
///
/// Expected output:
///   DTL Hello Distributed Example
///   Created distributed_vector with 100 elements
///   Rank: 0, Total ranks: 1
///   Global size: 100, Local size: 100
///
///   Filling with values 0, 1, 2, ...
///   First 10 elements: 0 1 2 3 4 5 6 7 8 9
///
///   Using for_each to square each element...
///   First 10 elements after squaring: 0 1 4 9 16 25 36 49 64 81
///
///   Local sum (no MPI required): 328350
///   Expected sum of squares: 328350
///   SUCCESS!

#include <dtl/dtl.hpp>
#include <iostream>

int main() {
    std::cout << "DTL Hello Distributed Example\n\n";

    // Create a distributed vector in standalone mode
    // (single rank, no MPI initialization required)
    const dtl::size_type global_size = 100;
    const auto ctx = dtl::make_cpu_context();

    dtl::distributed_vector<int> vec(global_size, ctx);

    std::cout << "Created distributed_vector with " << global_size << " elements\n";
    std::cout << "Rank: " << vec.rank() << ", Total ranks: " << vec.num_ranks() << "\n";
    std::cout << "Global size: " << vec.global_size()
              << ", Local size: " << vec.local_size() << "\n\n";

    // Fill with values using local view (STL-compatible, no communication)
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i);
    }

    std::cout << "Filling with values 0, 1, 2, ...\n";
    std::cout << "First 10 elements: ";
    for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
        std::cout << local[i] << " ";
    }
    std::cout << "\n\n";

    // Use DTL's for_each algorithm to transform elements
    std::cout << "Using for_each to square each element...\n";
    dtl::for_each(vec, [](int& x) { x = x * x; });

    std::cout << "First 10 elements after squaring: ";
    for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
        std::cout << local[i] << " ";
    }
    std::cout << "\n\n";

    // Use local_reduce (no MPI communication needed)
    int sum = dtl::local_reduce(vec, 0, std::plus<>{});
    std::cout << "Local sum (no MPI required): " << sum << "\n";

    // Verify: sum of i^2 for i=0..99 = 99*100*199/6 = 328350
    int expected = 99 * 100 * 199 / 6;
    std::cout << "Expected sum of squares: " << expected << "\n";

    if (sum == expected) {
        std::cout << "SUCCESS!\n";
        return 0;
    } else {
        std::cout << "FAILURE: sums don't match\n";
        return 1;
    }
}
