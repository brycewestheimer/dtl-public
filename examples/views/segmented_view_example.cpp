// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file segmented_view_example.cpp
/// @brief Demonstrates segmented iteration over distributed data
/// @details Shows how DTL's segmented_view provides a segment-based iteration
///          model for distributed containers. Each segment corresponds to one
///          rank's partition. This is the primary iteration substrate for
///          distributed algorithms, enabling predictable bulk
///          operations without hidden communication.
///
///          Key concepts demonstrated:
///          - Creating a segmented_view from a distributed_vector
///          - Iterating over segments (each segment = one rank's partition)
///          - Distinguishing local vs remote segments
///          - Using local_segment() for direct access to this rank's data
///          - Using for_each_local() convenience method
///          - Using segment metadata (global_offset, size, rank)
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./segmented_view_example
///
/// Run (multiple ranks):
///   mpirun -np 4 ./segmented_view_example

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
        std::cout << "DTL Segmented View Example\n";
        std::cout << "==========================\n\n";
        std::cout << "Number of ranks: " << size << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // 1. Create a distributed vector and initialize local data
    // =========================================================================
    const dtl::size_type N = 40;
    dtl::distributed_vector<int> vec(N, ctx);

    // Fill local partition with rank-specific values: rank*1000 + local_index
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(rank) * 1000 + static_cast<int>(i);
    }

    comm.barrier();

    // =========================================================================
    // 2. Create segmented view and inspect segment metadata
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Segment Metadata ---\n";
    }
    comm.barrier();

    auto seg_view = vec.segmented_view();

    if (rank == 0) {
        std::cout << "Total segments: " << seg_view.num_segments() << "\n";
        std::cout << "Total elements: " << seg_view.total_size() << "\n";
        std::cout << "Local segment size: " << seg_view.local_size() << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 3. Iterate over all segments, showing local vs remote
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Iterating All Segments (from rank 0's perspective) ---\n";
    }
    comm.barrier();

    if (rank == 0) {
        for (auto segment : seg_view) {
            std::cout << "Segment for rank " << segment.rank
                      << ": global_offset=" << segment.global_offset
                      << ", size=" << segment.size()
                      << ", is_local=" << std::boolalpha << segment.is_local()
                      << ", is_remote=" << segment.is_remote() << "\n";

            if (segment.is_local()) {
                std::cout << "  Local data: ";
                for (dtl::size_type i = 0; i < segment.size() && i < 8; ++i) {
                    std::cout << segment[i] << " ";
                }
                if (segment.size() > 8) std::cout << "...";
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }
    comm.barrier();

    // =========================================================================
    // 4. Use local_segment() for direct access to this rank's data
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Using local_segment() ---\n";
    }
    comm.barrier();

    auto my_segment = seg_view.local_segment();

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": local_segment has "
                      << my_segment.size() << " elements"
                      << ", global_offset=" << my_segment.global_offset
                      << ", first=" << (my_segment.empty() ? -1 : my_segment[0])
                      << ", last=" << (my_segment.empty() ? -1 : my_segment[my_segment.size() - 1])
                      << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    // =========================================================================
    // 5. Use for_each_local() convenience method to modify local data
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- Using for_each_local() to double values ---\n";
    }
    comm.barrier();

    seg_view.for_each_local([](int& x) { x *= 2; });

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << " first 5 after doubling: ";
            auto lv = vec.local_view();
            for (dtl::size_type i = 0; i < 5 && i < lv.size(); ++i) {
                std::cout << lv[i] << " ";
            }
            std::cout << "\n";
        }
        comm.barrier();
    }
    comm.barrier();

    // =========================================================================
    // 6. Use segment global_offset to compute global indices
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- Global Index Mapping ---\n";
    }
    comm.barrier();

    auto seg = seg_view.local_segment();
    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": local[0] -> global["
                      << seg.to_global(0) << "]"
                      << ", local[" << (seg.size() - 1) << "] -> global["
                      << seg.to_global(seg.size() - 1) << "]\n";
        }
        comm.barrier();
    }
    comm.barrier();

    // =========================================================================
    // 7. Using offset_for_rank() for O(1) offset lookups
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- Precomputed Offsets (O(1) lookup) ---\n";
        for (dtl::rank_t r = 0; r < size; ++r) {
            std::cout << "Rank " << r << " starts at global offset "
                      << seg_view.offset_for_rank(r) << "\n";
        }
    }
    comm.barrier();

    // =========================================================================
    // 8. Compute per-segment statistics using STL algorithms
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- Per-Segment Local Statistics ---\n";
    }
    comm.barrier();

    // Each rank computes stats on its local segment
    int local_sum = std::accumulate(my_segment.begin(), my_segment.end(), 0);
    int local_min = *std::min_element(my_segment.begin(), my_segment.end());
    int local_max = *std::max_element(my_segment.begin(), my_segment.end());

    for (dtl::rank_t r = 0; r < size; ++r) {
        if (rank == r) {
            std::cout << "Rank " << r << ": sum=" << local_sum
                      << ", min=" << local_min
                      << ", max=" << local_max << "\n";
        }
        comm.barrier();
    }

    // Global reduction
    int global_sum = comm.allreduce_sum_value<int>(local_sum);
    int global_min = comm.allreduce_min_value<int>(local_min);
    int global_max = comm.allreduce_max_value<int>(local_max);

    comm.barrier();

    if (rank == 0) {
        std::cout << "\nGlobal: sum=" << global_sum
                  << ", min=" << global_min
                  << ", max=" << global_max << "\n";
    }

    comm.barrier();
    if (rank == 0) {
        std::cout << "\nSUCCESS: Segmented view example completed!\n";
    }

    return 0;
}
