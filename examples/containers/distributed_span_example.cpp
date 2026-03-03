// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_span_example.cpp
/// @brief Demonstrates distributed_span as a non-owning view over distributed data
/// @details Shows how DTL's distributed_span provides a lightweight, non-owning
///          view over distributed data (analogous to std::span). The span does
///          not own the data — it views existing data from a distributed
///          container or raw pointers.
///
///          Key concepts demonstrated:
///          - Creating a distributed_span from a distributed_vector
///          - Using make_distributed_span() factory function
///          - Reading through the span's local data
///          - Subspans: first(), last(), subspan()
///          - Passing distributed data to functions without copying
///          - Distribution queries: size(), local_size(), rank()
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./distributed_span_example
///
/// Run (multiple ranks):
///   mpirun -np 4 ./distributed_span_example

#include <dtl/dtl.hpp>

#include <iostream>
#include <numeric>
#include <algorithm>

/// @brief Process data through a span (demonstrates zero-copy function argument)
/// @details This function takes a distributed_span, reads local data through
///          the span's iterators, and returns a local sum. No copy is made.
template <typename T>
T compute_local_sum(dtl::distributed_span<T, dtl::dynamic_extent> span) {
    T sum = T{};
    for (auto it = span.begin(); it != span.end(); ++it) {
        sum += *it;
    }
    return sum;
}

/// @brief Compute mean of span's local data
template <typename T>
double compute_local_mean(dtl::distributed_span<T, dtl::dynamic_extent> span) {
    if (span.local_size() == 0) return 0.0;
    T sum = compute_local_sum(span);
    return static_cast<double>(sum) / static_cast<double>(span.local_size());
}

int main(int argc, char** argv) {
    // Initialize DTL environment (handles all backend init/finalize via RAII)
    dtl::environment env(argc, argv);
    auto ctx = env.make_world_context();
    auto& comm = ctx.get<dtl::mpi_domain>().communicator();
    auto rank = ctx.rank();
    auto size = ctx.size();

    if (rank == 0) {
        std::cout << "DTL Distributed Span Example\n";
        std::cout << "============================\n\n";
        std::cout << "Number of ranks: " << size << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // 1. Create source data: a distributed_vector
    // =========================================================================
    const dtl::size_type N = 100;
    dtl::distributed_vector<int> vec(N, ctx);

    // Fill with sequential values
    auto local = vec.local_view();
    dtl::index_t start_val = static_cast<dtl::index_t>(vec.global_offset());
    std::iota(local.begin(), local.end(), static_cast<int>(start_val));

    if (rank == 0) {
        std::cout << "Source distributed_vector: " << N << " elements\n";
        std::cout << "Rank 0 local data: ";
        for (dtl::size_type i = 0; i < 10 && i < local.size(); ++i) {
            std::cout << local[i] << " ";
        }
        std::cout << "...\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 2. Create a distributed_span from the vector
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 1. Creating distributed_span from distributed_vector ---\n";
    }
    comm.barrier();

    auto span = dtl::make_distributed_span(vec);

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": span.size()=" << span.size()
                      << ", span.local_size()=" << span.local_size()
                      << ", span.empty()=" << std::boolalpha << span.empty()
                      << ", span.num_ranks()=" << span.num_ranks() << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 3. Accessing data through the span
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 2. Accessing Data Through the Span ---\n";
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "span[0] = " << span[0] << "\n";
        std::cout << "span.front() = " << span.front() << "\n";
        std::cout << "span.back() = " << span.back() << "\n";
        std::cout << "span.data() = " << span.data() << " (pointer to local data)\n";
        std::cout << "span.size_bytes() = " << span.size_bytes() << " bytes\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 4. Iterating via span's local iterators
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 3. Iterating Through Span ---\n";
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "Elements via span iterators: ";
        int count = 0;
        for (auto it = span.begin(); it != span.end() && count < 15; ++it, ++count) {
            std::cout << *it << " ";
        }
        std::cout << "...\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 5. Zero-copy function passing via span
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 4. Zero-Copy Function Arguments ---\n";
    }
    comm.barrier();

    // Pass data to functions via distributed_span (no copy!)
    int local_sum = compute_local_sum(span);
    double local_mean = compute_local_mean(span);

    int global_sum = comm.allreduce_sum_value<int>(local_sum);

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": local_sum=" << local_sum
                      << ", local_mean=" << local_mean << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "Global sum: " << global_sum << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 6. Subspan operations: first(), last(), subspan()
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 5. Subspan Operations ---\n";
    }
    comm.barrier();

    if (span.local_size() >= 5) {
        // first(n): view of first n local elements
        auto first3 = span.first(3);
        std::cout << "Rank " << rank << " first(3): ";
        for (auto it = first3.begin(); it != first3.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "(size=" << first3.local_size() << ")\n";

        // last(n): view of last n local elements
        auto last3 = span.last(3);
        std::cout << "Rank " << rank << " last(3): ";
        for (auto it = last3.begin(); it != last3.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "(size=" << last3.local_size() << ")\n";

        // subspan(offset, count): view of count elements starting at offset
        auto mid = span.subspan(2, 4);
        std::cout << "Rank " << rank << " subspan(2, 4): ";
        for (auto it = mid.begin(); it != mid.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "(size=" << mid.local_size() << ")\n";
    }
    comm.barrier();

    if (rank == 0) {
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 7. Const span: read-only view
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- 6. Const Span (Read-Only View) ---\n";
    }
    comm.barrier();

    const auto& const_vec = vec;
    auto const_span = dtl::make_distributed_span(const_vec);

    if (rank == 0) {
        std::cout << "Const span first 5: ";
        for (dtl::size_type i = 0; i < 5 && i < const_span.local_size(); ++i) {
            std::cout << const_span[i] << " ";
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 8. Span from raw pointer (single-rank mode)
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- 7. Span from Raw Pointer ---\n";
    }
    comm.barrier();

    if (rank == 0) {
        int raw_data[] = {10, 20, 30, 40, 50};
        dtl::distributed_span<int, dtl::dynamic_extent> raw_span(
            raw_data, 5, 5);

        std::cout << "Raw data span: ";
        for (auto it = raw_span.begin(); it != raw_span.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << "(size=" << raw_span.size()
                  << ", local_size=" << raw_span.local_size() << ")\n";
    }

    comm.barrier();
    if (rank == 0) {
        std::cout << "\nSUCCESS: Distributed span example completed!\n";
    }

    return 0;
}
