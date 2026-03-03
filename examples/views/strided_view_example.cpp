// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file strided_view_example.cpp
/// @brief Demonstrates strided access patterns over distributed data
/// @details Shows how DTL's strided_view provides N-D strided access patterns
///          over local data. strided_view accesses every Nth element, useful for:
///          - Red-black Gauss-Seidel iterations
///          - Interleaved/multi-channel data processing
///          - Even/odd partitioning
///          - Decimation/downsampling
///
///          Key concepts demonstrated:
///          - Creating strided views with make_strided_view()
///          - Even/odd element selection via stride=2 with offset 0 or 1
///          - Using strided views with STL algorithms
///          - Combining strided views for interleaved processing
///          - Multi-stride patterns for downsampling
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run (single rank):
///   ./strided_view_example
///
/// Run (multiple ranks):
///   mpirun -np 4 ./strided_view_example

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
        std::cout << "DTL Strided View Example\n";
        std::cout << "========================\n\n";
        std::cout << "Number of ranks: " << size << "\n\n";
    }

    comm.barrier();

    // =========================================================================
    // 1. Create a distributed vector and fill with sequential values
    // =========================================================================
    const dtl::size_type N = 80;
    dtl::distributed_vector<int> vec(N, ctx);

    auto local = vec.local_view();
    dtl::index_t start_val = static_cast<dtl::index_t>(vec.global_offset());
    std::iota(local.begin(), local.end(), static_cast<int>(start_val));

    comm.barrier();

    if (rank == 0) {
        std::cout << "Created distributed_vector with " << N << " elements\n";
        std::cout << "Rank 0 local data: ";
        for (dtl::size_type i = 0; i < local.size() && i < 20; ++i) {
            std::cout << local[i] << " ";
        }
        if (local.size() > 20) std::cout << "...";
        std::cout << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 2. Even/Odd splitting with stride=2
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Even/Odd Splitting (stride=2) ---\n";
    }
    comm.barrier();

    // Even-indexed elements: offset=0, stride=2
    auto even = dtl::make_strided_view(local, 2, 0);
    // Odd-indexed elements: offset=1, stride=2
    auto odd = dtl::make_strided_view(local, 2, 1);

    if (rank == 0) {
        std::cout << "Even indices (" << even.size() << " elements): ";
        for (dtl::size_type i = 0; i < even.size() && i < 10; ++i) {
            std::cout << even[i] << " ";
        }
        if (even.size() > 10) std::cout << "...";
        std::cout << "\n";

        std::cout << "Odd indices  (" << odd.size() << " elements): ";
        for (dtl::size_type i = 0; i < odd.size() && i < 10; ++i) {
            std::cout << odd[i] << " ";
        }
        if (odd.size() > 10) std::cout << "...";
        std::cout << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 3. Using STL algorithms on strided views
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- STL Algorithms on Strided Views ---\n";
    }
    comm.barrier();

    // Sum even-indexed elements
    int even_sum = std::accumulate(even.begin(), even.end(), 0);
    int odd_sum = std::accumulate(odd.begin(), odd.end(), 0);

    int global_even_sum = comm.allreduce_sum_value<int>(even_sum);
    int global_odd_sum = comm.allreduce_sum_value<int>(odd_sum);

    if (rank == 0) {
        std::cout << "Global sum of even-indexed elements: " << global_even_sum << "\n";
        std::cout << "Global sum of odd-indexed elements:  " << global_odd_sum << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 4. Transform only even-indexed elements (e.g., red-black pattern)
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Red-Black Pattern: Negate even-indexed elements ---\n";
    }
    comm.barrier();

    // Negate even-indexed elements (simulating red-black Gauss-Seidel sweep)
    std::transform(even.begin(), even.end(), even.begin(),
                   [](int x) { return -x; });

    if (rank == 0) {
        std::cout << "After negating even indices, local data: ";
        for (dtl::size_type i = 0; i < local.size() && i < 20; ++i) {
            std::cout << local[i] << " ";
        }
        if (local.size() > 20) std::cout << "...";
        std::cout << "\n\n";
    }
    comm.barrier();

    // Restore original values for subsequent demonstrations
    std::iota(local.begin(), local.end(), static_cast<int>(start_val));
    comm.barrier();

    // =========================================================================
    // 5. Downsampling with stride=4 (every 4th element)
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Downsampling (stride=4) ---\n";
    }
    comm.barrier();

    auto downsampled = dtl::make_strided_view(local, 4, 0);

    if (rank == 0) {
        std::cout << "Original size:     " << local.size() << "\n";
        std::cout << "Downsampled size:  " << downsampled.size() << "\n";
        std::cout << "Stride:            " << downsampled.stride() << "\n";
        std::cout << "Offset:            " << downsampled.offset() << "\n";
        std::cout << "Downsampled values: ";
        for (dtl::size_type i = 0; i < downsampled.size() && i < 10; ++i) {
            std::cout << downsampled[i] << " ";
        }
        if (downsampled.size() > 10) std::cout << "...";
        std::cout << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 6. Multi-channel interleaved data
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Interleaved Multi-Channel Data (3 channels, stride=3) ---\n";
    }
    comm.barrier();

    // Create a vector where data is interleaved: [R0,G0,B0, R1,G1,B1, ...]
    const dtl::size_type pixel_count = 60;  // 20 pixels * 3 channels
    dtl::distributed_vector<int> rgb_data(pixel_count, ctx);
    auto rgb_local = rgb_data.local_view();

    // Fill: channel 0 (Red) = 100+pixel, channel 1 (Green) = 200+pixel,
    //        channel 2 (Blue) = 300+pixel
    for (dtl::size_type i = 0; i < rgb_local.size(); ++i) {
        dtl::size_type channel = i % 3;
        dtl::size_type pixel = i / 3;
        rgb_local[i] = static_cast<int>((channel + 1) * 100 + pixel);
    }

    // Extract each channel using stride=3
    auto red_channel   = dtl::make_strided_view(rgb_local, 3, 0);
    auto green_channel = dtl::make_strided_view(rgb_local, 3, 1);
    auto blue_channel  = dtl::make_strided_view(rgb_local, 3, 2);

    if (rank == 0) {
        std::cout << "Red channel   (" << red_channel.size() << " pixels): ";
        for (dtl::size_type i = 0; i < red_channel.size() && i < 8; ++i) {
            std::cout << red_channel[i] << " ";
        }
        if (red_channel.size() > 8) std::cout << "...";
        std::cout << "\n";

        std::cout << "Green channel (" << green_channel.size() << " pixels): ";
        for (dtl::size_type i = 0; i < green_channel.size() && i < 8; ++i) {
            std::cout << green_channel[i] << " ";
        }
        if (green_channel.size() > 8) std::cout << "...";
        std::cout << "\n";

        std::cout << "Blue channel  (" << blue_channel.size() << " pixels): ";
        for (dtl::size_type i = 0; i < blue_channel.size() && i < 8; ++i) {
            std::cout << blue_channel[i] << " ";
        }
        if (blue_channel.size() > 8) std::cout << "...";
        std::cout << "\n\n";
    }
    comm.barrier();

    // =========================================================================
    // 7. Per-channel statistics using strided views
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Per-Channel Statistics ---\n";
    }
    comm.barrier();

    auto channel_stats = [&](const char* name, auto& channel) {
        int local_sum_ch = std::accumulate(channel.begin(), channel.end(), 0);
        int global_sum_ch = comm.allreduce_sum_value<int>(local_sum_ch);

        int local_min_ch = *std::min_element(channel.begin(), channel.end());
        int local_max_ch = *std::max_element(channel.begin(), channel.end());
        int global_min_ch = comm.allreduce_min_value<int>(local_min_ch);
        int global_max_ch = comm.allreduce_max_value<int>(local_max_ch);

        if (rank == 0) {
            std::cout << name << ": sum=" << global_sum_ch
                      << ", min=" << global_min_ch
                      << ", max=" << global_max_ch << "\n";
        }
    };

    channel_stats("Red  ", red_channel);
    channel_stats("Green", green_channel);
    channel_stats("Blue ", blue_channel);

    comm.barrier();
    if (rank == 0) {
        std::cout << "\nSUCCESS: Strided view example completed!\n";
    }

    return 0;
}
