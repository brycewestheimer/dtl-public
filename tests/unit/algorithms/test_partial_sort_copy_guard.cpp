// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_partial_sort_copy_guard.cpp
/// @brief Verify partial_sort_copy multi-rank guard

#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

struct mock_container {
    int num_ranks() const { return ranks_; }
    int ranks_ = 1;
};

}  // namespace

TEST(PartialSortCopyGuard, SingleRankNoThrow) {
    mock_container c;
    c.ranks_ = 1;
    EXPECT_NO_THROW(dtl::detail::require_collective_comm_or_single_rank(c, "dtl::partial_sort_copy"));
}

TEST(PartialSortCopyGuard, MultiRankThrows) {
    mock_container c;
    c.ranks_ = 2;
    EXPECT_THROW(dtl::detail::require_collective_comm_or_single_rank(c, "dtl::partial_sort_copy"),
                 std::runtime_error);
}
