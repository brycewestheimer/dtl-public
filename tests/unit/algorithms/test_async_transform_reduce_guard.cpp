// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_async_transform_reduce_guard.cpp
/// @brief Verify async_transform_reduce multi-rank guard

#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

struct mock_container {
    int num_ranks() const { return ranks_; }
    int ranks_ = 1;
};

}  // namespace

TEST(AsyncTransformReduceGuard, SingleRankNoThrow) {
    mock_container c;
    c.ranks_ = 1;
    EXPECT_NO_THROW(dtl::detail::require_collective_comm_or_single_rank(c, "dtl::async_transform_reduce"));
}

TEST(AsyncTransformReduceGuard, MultiRankThrows) {
    mock_container c;
    c.ranks_ = 2;
    EXPECT_THROW(dtl::detail::require_collective_comm_or_single_rank(c, "dtl::async_transform_reduce"),
                 std::runtime_error);
}
