// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_scan.cpp
/// @brief Tests for distributed scan algorithms (V1.1)

#include <gtest/gtest.h>

#include <dtl/algorithms/reductions/scan.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <numeric>
#include <vector>

namespace dtl::test {

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

// ============================================================================
// Local Inclusive Scan Tests
// ============================================================================

TEST(LocalInclusiveScan, SingleRank) {
    distributed_vector<int> input(10, test_context{0, 1});
    distributed_vector<int> output(10, test_context{0, 1});

    // Initialize input: 1, 2, 3, ..., 10
    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(i + 1);
    }

    local_inclusive_scan(input, output, 0, std::plus<>{});

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 1);   // 1
    EXPECT_EQ(local_out[1], 3);   // 1+2
    EXPECT_EQ(local_out[2], 6);   // 1+2+3
    EXPECT_EQ(local_out[9], 55);  // Sum of 1..10
}

TEST(LocalInclusiveScan, WithInitValue) {
    distributed_vector<int> input(5, test_context{0, 1});
    distributed_vector<int> output(5, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 1;  // All ones
    }

    local_inclusive_scan(input, output, 10, std::plus<>{});

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 11);  // 10+1
    EXPECT_EQ(local_out[1], 12);  // 10+1+1
    EXPECT_EQ(local_out[4], 15);  // 10+5
}

TEST(LocalInclusiveScan, EmptyContainer) {
    distributed_vector<int> input(0, test_context{0, 1});
    distributed_vector<int> output(0, test_context{0, 1});

    local_inclusive_scan(input, output, 0, std::plus<>{});

    EXPECT_EQ(output.local_size(), 0);
}

TEST(LocalInclusiveScan, Multiplication) {
    distributed_vector<int> input(5, test_context{0, 1});
    distributed_vector<int> output(5, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(i + 1);
    }

    local_inclusive_scan(input, output, 1, std::multiplies<>{});

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 1);    // 1
    EXPECT_EQ(local_out[1], 2);    // 1*2
    EXPECT_EQ(local_out[2], 6);    // 1*2*3
    EXPECT_EQ(local_out[4], 120);  // 5!
}

// ============================================================================
// Local Exclusive Scan Tests
// ============================================================================

TEST(LocalExclusiveScan, SingleRank) {
    distributed_vector<int> input(5, test_context{0, 1});
    distributed_vector<int> output(5, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(i + 1);
    }

    local_exclusive_scan(input, output, 0, std::plus<>{});

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 0);   // init
    EXPECT_EQ(local_out[1], 1);   // 1
    EXPECT_EQ(local_out[2], 3);   // 1+2
    EXPECT_EQ(local_out[3], 6);   // 1+2+3
    EXPECT_EQ(local_out[4], 10);  // 1+2+3+4
}

TEST(LocalExclusiveScan, WithInitValue) {
    distributed_vector<int> input(4, test_context{0, 1});
    distributed_vector<int> output(4, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 2;
    }

    local_exclusive_scan(input, output, 100, std::plus<>{});

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 100);
    EXPECT_EQ(local_out[1], 102);
    EXPECT_EQ(local_out[2], 104);
    EXPECT_EQ(local_out[3], 106);
}

// ============================================================================
// Distributed Scan Tests (Single Rank)
// ============================================================================

TEST(DistributedInclusiveScan, SingleRank) {
    distributed_vector<int> input(10, test_context{0, 1});
    distributed_vector<int> output(10, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 1;
    }

    auto result = inclusive_scan(seq{}, input, output, 0, std::plus<>{});

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    for (size_type i = 0; i < local_out.size(); ++i) {
        EXPECT_EQ(local_out[i], static_cast<int>(i + 1));
    }
}

TEST(DistributedExclusiveScan, SingleRank) {
    distributed_vector<int> input(5, test_context{0, 1});
    distributed_vector<int> output(5, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 2;
    }

    auto result = exclusive_scan(seq{}, input, output, 0, std::plus<>{});

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 0);
    EXPECT_EQ(local_out[1], 2);
    EXPECT_EQ(local_out[2], 4);
    EXPECT_EQ(local_out[3], 6);
    EXPECT_EQ(local_out[4], 8);
}

// ============================================================================
// Scan with Default Plus Operation
// ============================================================================

TEST(DistributedScan, InclusiveScanDefaultPlus) {
    distributed_vector<double> input(8, test_context{0, 1});
    distributed_vector<double> output(8, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 0.5;
    }

    auto result = inclusive_scan(par{}, input, output);

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_DOUBLE_EQ(local_out[0], 0.5);
    EXPECT_DOUBLE_EQ(local_out[7], 4.0);
}

TEST(DistributedScan, ExclusiveScanDefaultPlus) {
    distributed_vector<double> input(4, test_context{0, 1});
    distributed_vector<double> output(4, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 1.5;
    }

    auto result = exclusive_scan(par{}, input, output);

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_DOUBLE_EQ(local_out[0], 0.0);
    EXPECT_DOUBLE_EQ(local_out[1], 1.5);
    EXPECT_DOUBLE_EQ(local_out[2], 3.0);
    EXPECT_DOUBLE_EQ(local_out[3], 4.5);
}

// ============================================================================
// Transform Scan Tests
// ============================================================================

TEST(TransformScan, TransformInclusiveScan) {
    distributed_vector<int> input(5, test_context{0, 1});
    distributed_vector<int> output(5, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(i + 1);
    }

    // Sum of squares
    auto result = transform_inclusive_scan(
        seq{}, input, output, 0,
        std::plus<>{},
        [](int x) { return x * x; }
    );

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 1);   // 1^2
    EXPECT_EQ(local_out[1], 5);   // 1+4
    EXPECT_EQ(local_out[2], 14);  // 1+4+9
    EXPECT_EQ(local_out[3], 30);  // 1+4+9+16
    EXPECT_EQ(local_out[4], 55);  // 1+4+9+16+25
}

TEST(TransformScan, TransformExclusiveScan) {
    distributed_vector<int> input(4, test_context{0, 1});
    distributed_vector<int> output(4, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(i + 1);
    }

    // Exclusive sum of squares
    auto result = transform_exclusive_scan(
        seq{}, input, output, 0,
        std::plus<>{},
        [](int x) { return x * x; }
    );

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 0);   // init
    EXPECT_EQ(local_out[1], 1);   // 1^2
    EXPECT_EQ(local_out[2], 5);   // 1+4
    EXPECT_EQ(local_out[3], 14);  // 1+4+9
}

// ============================================================================
// Adjacent Difference Tests
// ============================================================================

TEST(AdjacentDifference, Basic) {
    distributed_vector<int> input(5, test_context{0, 1});
    distributed_vector<int> output(5, test_context{0, 1});

    auto local_in = input.local_view();
    // Cumulative sums: 1, 3, 6, 10, 15
    local_in[0] = 1;
    local_in[1] = 3;
    local_in[2] = 6;
    local_in[3] = 10;
    local_in[4] = 15;

    auto result = adjacent_difference(seq{}, input, output);

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 1);  // First element copied
    EXPECT_EQ(local_out[1], 2);  // 3-1
    EXPECT_EQ(local_out[2], 3);  // 6-3
    EXPECT_EQ(local_out[3], 4);  // 10-6
    EXPECT_EQ(local_out[4], 5);  // 15-10
}

TEST(AdjacentDifference, InverseOfInclusiveScan) {
    distributed_vector<int> original(5, test_context{0, 1});
    distributed_vector<int> scanned(5, test_context{0, 1});
    distributed_vector<int> recovered(5, test_context{0, 1});

    auto local = original.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i + 1);
    }

    inclusive_scan(seq{}, original, scanned, 0, std::plus<>{});
    adjacent_difference(seq{}, scanned, recovered);

    auto orig_view = original.local_view();
    auto rec_view = recovered.local_view();

    for (size_type i = 0; i < orig_view.size(); ++i) {
        EXPECT_EQ(rec_view[i], orig_view[i]);
    }
}

// ============================================================================
// Partial Sum Tests
// ============================================================================

TEST(PartialSum, Basic) {
    distributed_vector<int> input(6, test_context{0, 1});
    distributed_vector<int> output(6, test_context{0, 1});

    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 1;
    }

    auto result = partial_sum(seq{}, input, output);

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    for (size_type i = 0; i < local_out.size(); ++i) {
        EXPECT_EQ(local_out[i], static_cast<int>(i + 1));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(ScanErrors, SizeMismatch) {
    distributed_vector<int> input(10, test_context{0, 1});
    distributed_vector<int> output(5, test_context{0, 1});  // Different size

    auto result = inclusive_scan(seq{}, input, output, 0, std::plus<>{});

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_argument);
}

// ============================================================================
// Type Tests
// ============================================================================

TEST(ScanTypes, FloatingPoint) {
    distributed_vector<double> input(4, test_context{0, 1});
    distributed_vector<double> output(4, test_context{0, 1});

    auto local_in = input.local_view();
    local_in[0] = 1.1;
    local_in[1] = 2.2;
    local_in[2] = 3.3;
    local_in[3] = 4.4;

    auto result = inclusive_scan(seq{}, input, output, 0.0, std::plus<>{});

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_NEAR(local_out[0], 1.1, 1e-10);
    EXPECT_NEAR(local_out[1], 3.3, 1e-10);
    EXPECT_NEAR(local_out[2], 6.6, 1e-10);
    EXPECT_NEAR(local_out[3], 11.0, 1e-10);
}

TEST(ScanTypes, LongIntegers) {
    distributed_vector<long> input(4, test_context{0, 1});
    distributed_vector<long> output(4, test_context{0, 1});

    auto local_in = input.local_view();
    local_in[0] = 1000000000L;
    local_in[1] = 1000000000L;
    local_in[2] = 1000000000L;
    local_in[3] = 1000000000L;

    auto result = inclusive_scan(seq{}, input, output, 0L, std::plus<>{});

    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();
    EXPECT_EQ(local_out[0], 1000000000L);
    EXPECT_EQ(local_out[3], 4000000000L);
}

}  // namespace dtl::test
