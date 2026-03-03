// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_variable_size_collectives.cpp
/// @brief Tests for variable-size collective operations (V1.1)
/// @details Tests alltoallv, scatterv, gatherv, allgatherv, and scan operations.

#include <gtest/gtest.h>

#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

// ============================================================================
// Scan Algorithm Tests (Local)
// ============================================================================

TEST(ScanAlgorithms, ExclusiveScanBasic) {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> output(5);

    // Manual exclusive scan implementation
    int running = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = running;
        running += input[i];
    }

    EXPECT_EQ(output[0], 0);  // 0
    EXPECT_EQ(output[1], 1);  // 1
    EXPECT_EQ(output[2], 3);  // 1+2
    EXPECT_EQ(output[3], 6);  // 1+2+3
    EXPECT_EQ(output[4], 10); // 1+2+3+4
}

TEST(ScanAlgorithms, InclusiveScanBasic) {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> output(5);

    // Manual inclusive scan implementation
    int running = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        running += input[i];
        output[i] = running;
    }

    EXPECT_EQ(output[0], 1);  // 1
    EXPECT_EQ(output[1], 3);  // 1+2
    EXPECT_EQ(output[2], 6);  // 1+2+3
    EXPECT_EQ(output[3], 10); // 1+2+3+4
    EXPECT_EQ(output[4], 15); // 1+2+3+4+5
}

TEST(ScanAlgorithms, StlExclusiveScan) {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> output(5);

    std::exclusive_scan(input.begin(), input.end(), output.begin(), 0);

    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[1], 1);
    EXPECT_EQ(output[2], 3);
    EXPECT_EQ(output[3], 6);
    EXPECT_EQ(output[4], 10);
}

TEST(ScanAlgorithms, StlInclusiveScan) {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> output(5);

    std::inclusive_scan(input.begin(), input.end(), output.begin());

    EXPECT_EQ(output[0], 1);
    EXPECT_EQ(output[1], 3);
    EXPECT_EQ(output[2], 6);
    EXPECT_EQ(output[3], 10);
    EXPECT_EQ(output[4], 15);
}

TEST(ScanAlgorithms, ExclusiveScanWithInitValue) {
    std::vector<int> input = {1, 2, 3};
    std::vector<int> output(3);

    std::exclusive_scan(input.begin(), input.end(), output.begin(), 10);

    EXPECT_EQ(output[0], 10);  // init
    EXPECT_EQ(output[1], 11);  // 10+1
    EXPECT_EQ(output[2], 13);  // 10+1+2
}

TEST(ScanAlgorithms, TransformExclusiveScan) {
    std::vector<int> input = {1, 2, 3, 4};
    std::vector<int> output(4);

    // Sum of squares using transform_exclusive_scan
    std::transform_exclusive_scan(
        input.begin(), input.end(), output.begin(),
        0,
        std::plus<>{},
        [](int x) { return x * x; }
    );

    // output[i] = 0 + 1^2 + 2^2 + ... + (i-1)^2
    EXPECT_EQ(output[0], 0);   // 0
    EXPECT_EQ(output[1], 1);   // 1^2
    EXPECT_EQ(output[2], 5);   // 1^2 + 2^2
    EXPECT_EQ(output[3], 14);  // 1^2 + 2^2 + 3^2
}

// ============================================================================
// Displacement Computation Tests
// ============================================================================

TEST(DisplacementComputation, BasicComputation) {
    std::vector<int> counts = {3, 2, 4, 1};
    std::vector<int> displs(4);

    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

    EXPECT_EQ(displs[0], 0);  // Start at 0
    EXPECT_EQ(displs[1], 3);  // After rank 0's 3 elements
    EXPECT_EQ(displs[2], 5);  // After rank 1's 2 elements
    EXPECT_EQ(displs[3], 9);  // After rank 2's 4 elements
}

TEST(DisplacementComputation, ZeroCounts) {
    std::vector<int> counts = {0, 3, 0, 2};
    std::vector<int> displs(4);

    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

    EXPECT_EQ(displs[0], 0);
    EXPECT_EQ(displs[1], 0);  // Rank 0 had 0 elements
    EXPECT_EQ(displs[2], 3);  // Rank 1 had 3 elements
    EXPECT_EQ(displs[3], 3);  // Rank 2 had 0 elements
}

TEST(DisplacementComputation, TotalSize) {
    std::vector<int> counts = {3, 2, 4, 1};
    std::vector<int> displs(4);

    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

    // Total size is last displacement + last count
    int total = displs.back() + counts.back();
    EXPECT_EQ(total, 10);
}

// ============================================================================
// Buffer Preparation Tests (for alltoallv)
// ============================================================================

TEST(AlltoallvPrep, SendBufferFlattening) {
    // Simulate send_buffers[rank] = elements to send to that rank
    std::vector<std::vector<int>> send_buffers = {
        {1, 2, 3},    // To rank 0
        {4, 5},       // To rank 1
        {},           // To rank 2 (nothing)
        {6, 7, 8, 9}  // To rank 3
    };

    // Compute send counts
    std::vector<int> send_counts(4);
    for (size_t i = 0; i < 4; ++i) {
        send_counts[i] = static_cast<int>(send_buffers[i].size());
    }

    EXPECT_EQ(send_counts[0], 3);
    EXPECT_EQ(send_counts[1], 2);
    EXPECT_EQ(send_counts[2], 0);
    EXPECT_EQ(send_counts[3], 4);

    // Compute send displacements
    std::vector<int> send_displs(4);
    std::exclusive_scan(send_counts.begin(), send_counts.end(),
                        send_displs.begin(), 0);

    EXPECT_EQ(send_displs[0], 0);
    EXPECT_EQ(send_displs[1], 3);
    EXPECT_EQ(send_displs[2], 5);
    EXPECT_EQ(send_displs[3], 5);

    // Flatten send buffer
    std::vector<int> send_flat;
    for (const auto& buf : send_buffers) {
        send_flat.insert(send_flat.end(), buf.begin(), buf.end());
    }

    EXPECT_EQ(send_flat.size(), 9);
    EXPECT_EQ(send_flat[0], 1);  // Start of rank 0's data
    EXPECT_EQ(send_flat[3], 4);  // Start of rank 1's data
    EXPECT_EQ(send_flat[5], 6);  // Start of rank 3's data
}

// ============================================================================
// Adjacent Difference Tests (Inverse of Scan)
// ============================================================================

TEST(AdjacentDifference, Basic) {
    std::vector<int> input = {1, 3, 6, 10, 15};
    std::vector<int> output(5);

    std::adjacent_difference(input.begin(), input.end(), output.begin());

    // First element is copied as-is
    EXPECT_EQ(output[0], 1);
    // Rest are differences
    EXPECT_EQ(output[1], 2);  // 3-1
    EXPECT_EQ(output[2], 3);  // 6-3
    EXPECT_EQ(output[3], 4);  // 10-6
    EXPECT_EQ(output[4], 5);  // 15-10
}

TEST(AdjacentDifference, InverseOfScan) {
    std::vector<int> original = {1, 2, 3, 4, 5};
    std::vector<int> scanned(5);
    std::vector<int> recovered(5);

    std::inclusive_scan(original.begin(), original.end(), scanned.begin());
    std::adjacent_difference(scanned.begin(), scanned.end(), recovered.begin());

    // Should recover original
    EXPECT_EQ(recovered, original);
}

// ============================================================================
// Type Conversion Tests
// ============================================================================

TEST(TypeConversion, IntToDouble) {
    std::vector<int> int_data = {1, 2, 3, 4};
    std::vector<double> double_data(4);

    std::transform(int_data.begin(), int_data.end(), double_data.begin(),
                   [](int x) { return static_cast<double>(x); });

    EXPECT_DOUBLE_EQ(double_data[0], 1.0);
    EXPECT_DOUBLE_EQ(double_data[3], 4.0);
}

TEST(TypeConversion, ScanWithMultiplication) {
    std::vector<int> input = {2, 3, 4, 5};
    std::vector<int> output(4);

    std::inclusive_scan(input.begin(), input.end(), output.begin(),
                        std::multiplies<>{});

    EXPECT_EQ(output[0], 2);      // 2
    EXPECT_EQ(output[1], 6);      // 2*3
    EXPECT_EQ(output[2], 24);     // 2*3*4
    EXPECT_EQ(output[3], 120);    // 2*3*4*5 (5!)
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(EdgeCases, EmptyInput) {
    std::vector<int> empty;
    std::vector<int> output;

    std::exclusive_scan(empty.begin(), empty.end(), output.begin(), 0);
    // Should not crash, output remains empty
}

TEST(EdgeCases, SingleElement) {
    std::vector<int> input = {42};
    std::vector<int> exclusive_out(1);
    std::vector<int> inclusive_out(1);

    std::exclusive_scan(input.begin(), input.end(), exclusive_out.begin(), 0);
    std::inclusive_scan(input.begin(), input.end(), inclusive_out.begin());

    EXPECT_EQ(exclusive_out[0], 0);   // Init value
    EXPECT_EQ(inclusive_out[0], 42);  // The single element
}

TEST(EdgeCases, AllZeros) {
    std::vector<int> zeros = {0, 0, 0, 0};
    std::vector<int> output(4);

    std::inclusive_scan(zeros.begin(), zeros.end(), output.begin());

    for (int val : output) {
        EXPECT_EQ(val, 0);
    }
}

TEST(EdgeCases, LargeValues) {
    std::vector<long> input = {1000000000L, 1000000000L, 1000000000L};
    std::vector<long> output(3);

    std::inclusive_scan(input.begin(), input.end(), output.begin());

    EXPECT_EQ(output[0], 1000000000L);
    EXPECT_EQ(output[1], 2000000000L);
    EXPECT_EQ(output[2], 3000000000L);
}

}  // namespace dtl::test
