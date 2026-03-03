// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_collective_error.cpp
/// @brief Unit tests for collective_error aggregation
/// @details Tests for Phase 11.5: distributed error handling

#include <dtl/error/collective_error.hpp>
#include <dtl/error/status.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

// =============================================================================
// Construction Tests
// =============================================================================

TEST(CollectiveErrorTest, DefaultConstruction) {
    collective_error err;

    EXPECT_FALSE(err.has_errors());
    EXPECT_TRUE(err.all_succeeded());
    EXPECT_EQ(err.error_count(), 0);
    EXPECT_EQ(err.num_ranks(), 0);
}

TEST(CollectiveErrorTest, ConstructWithNumRanks) {
    collective_error err(4);

    EXPECT_FALSE(err.has_errors());
    EXPECT_TRUE(err.all_succeeded());
    EXPECT_EQ(err.error_count(), 0);
    EXPECT_EQ(err.num_ranks(), 4);
}

// =============================================================================
// Error Recording Tests
// =============================================================================

TEST(CollectiveErrorTest, AddSingleError) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument});

    EXPECT_TRUE(err.has_errors());
    EXPECT_FALSE(err.all_succeeded());
    EXPECT_EQ(err.error_count(), 1);
}

TEST(CollectiveErrorTest, AddMultipleErrors) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument});
    err.add_error(1, status{status_code::out_of_range});
    err.add_error(2, status{status_code::not_implemented});

    EXPECT_EQ(err.error_count(), 3);
}

TEST(CollectiveErrorTest, AddSameRankTwice) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument});
    err.add_error(0, status{status_code::out_of_range});

    // Both errors should be recorded (implementation stores all)
    EXPECT_EQ(err.error_count(), 2);
}

TEST(CollectiveErrorTest, AddAllRanks) {
    collective_error err(4);

    for (rank_t r = 0; r < 4; ++r) {
        err.add_error(r, status{status_code::internal_error});
    }

    EXPECT_EQ(err.error_count(), 4);
}

TEST(CollectiveErrorTest, AddOkStatusIgnored) {
    collective_error err(4);

    // Adding an ok status should not add an error
    err.add_error(0, status{status_code::ok});

    EXPECT_FALSE(err.has_errors());
    EXPECT_EQ(err.error_count(), 0);
}

// =============================================================================
// Query Tests
// =============================================================================

TEST(CollectiveErrorTest, ErrorsVector) {
    collective_error err(4);

    err.add_error(1, status{status_code::invalid_argument});
    err.add_error(3, status{status_code::out_of_range});

    const auto& errors = err.errors();
    EXPECT_EQ(errors.size(), 2);
    EXPECT_EQ(errors[0].rank, 1);
    EXPECT_EQ(errors[1].rank, 3);
}

TEST(CollectiveErrorTest, RankErrorHasError) {
    collective_error err(4);

    err.add_error(2, status{status_code::invalid_argument});

    const auto& errors = err.errors();
    EXPECT_TRUE(errors[0].has_error());
}

// =============================================================================
// First Error Tests
// =============================================================================

TEST(CollectiveErrorTest, FirstErrorExists) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument, 0, "first"});
    err.add_error(1, status{status_code::out_of_range, 1, "second"});

    status first = err.first_error();
    EXPECT_EQ(first.code(), status_code::invalid_argument);
}

TEST(CollectiveErrorTest, FirstErrorNoErrors) {
    collective_error err(4);

    status first = err.first_error();
    EXPECT_TRUE(first.ok());
}

// =============================================================================
// Most Common Error Tests
// =============================================================================

TEST(CollectiveErrorTest, MostCommonErrorSingle) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument});

    status_code common = err.most_common_error();
    EXPECT_EQ(common, status_code::invalid_argument);
}

TEST(CollectiveErrorTest, MostCommonErrorMultiple) {
    collective_error err(8);

    // 3 invalid_argument, 2 out_of_range, 1 internal_error
    err.add_error(0, status{status_code::invalid_argument});
    err.add_error(1, status{status_code::invalid_argument});
    err.add_error(2, status{status_code::invalid_argument});
    err.add_error(3, status{status_code::out_of_range});
    err.add_error(4, status{status_code::out_of_range});
    err.add_error(5, status{status_code::internal_error});

    status_code common = err.most_common_error();
    EXPECT_EQ(common, status_code::invalid_argument);
}

TEST(CollectiveErrorTest, MostCommonErrorEmpty) {
    collective_error err(4);

    status_code common = err.most_common_error();
    EXPECT_EQ(common, status_code::ok);
}

// =============================================================================
// Summary Tests
// =============================================================================

TEST(CollectiveErrorTest, SummaryAllSucceeded) {
    collective_error err(4);

    status summary = err.summary();
    EXPECT_TRUE(summary.ok());
}

TEST(CollectiveErrorTest, SummaryWithErrors) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument});
    err.add_error(2, status{status_code::out_of_range});

    status summary = err.summary();
    EXPECT_FALSE(summary.ok());
    EXPECT_FALSE(summary.message().empty());
}

TEST(CollectiveErrorTest, SummaryMessage) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument});

    status summary = err.summary();
    std::string msg = summary.message();

    // Should mention rank counts
    EXPECT_TRUE(msg.find("1") != std::string::npos);
    EXPECT_TRUE(msg.find("4") != std::string::npos);
}

// =============================================================================
// to_string Tests
// =============================================================================

TEST(CollectiveErrorTest, ToStringAllSucceeded) {
    collective_error err(4);

    std::string str = err.to_string();
    EXPECT_TRUE(str.find("succeeded") != std::string::npos);
    EXPECT_TRUE(str.find("4") != std::string::npos);
}

TEST(CollectiveErrorTest, ToStringWithErrors) {
    collective_error err(4);

    err.add_error(0, status{status_code::invalid_argument});

    std::string str = err.to_string();
    EXPECT_TRUE(str.find("failed") != std::string::npos);
    EXPECT_TRUE(str.find("rank") != std::string::npos);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(CollectiveErrorTest, ZeroRanks) {
    collective_error err(0);

    EXPECT_EQ(err.num_ranks(), 0);
    EXPECT_TRUE(err.all_succeeded());
}

TEST(CollectiveErrorTest, LargeRankCount) {
    collective_error err(1000);

    EXPECT_EQ(err.num_ranks(), 1000);
}

TEST(CollectiveErrorTest, ManyErrors) {
    collective_error err(100);

    for (rank_t r = 0; r < 50; ++r) {
        err.add_error(r, status{status_code::internal_error});
    }

    EXPECT_EQ(err.error_count(), 50);
}

}  // namespace dtl::test
