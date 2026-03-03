// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_structural_mutation_conformance.cpp
/// @brief Structural mutation conformance tests for migrated algorithms (Phase 05)

#include <dtl/algorithms/modifying/rotate.hpp>
#include <dtl/algorithms/sorting/sort.hpp>
#include <dtl/algorithms/sorting/unique.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/execution/seq.hpp>

#include "mock_single_rank_comm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
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

template <typename T>
std::vector<T> sorted_copy(std::vector<T> values) {
    std::sort(values.begin(), values.end());
    return values;
}

}  // namespace

TEST(StructuralMutationConformanceTest, DistributedSortPreservesElementMultiset) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::vector<int> before(10);

    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>((17 * i + 5) % 13);
        before[i] = local[i];
    }

    mock_single_rank_comm comm;
    auto result = dtl::sort(seq{}, vec, std::less<>{}, comm);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(vec.local_size(), before.size());
    EXPECT_TRUE(vec.structural_metadata_consistent());

    std::vector<int> after(local.begin(), local.end());
    EXPECT_EQ(sorted_copy(before), sorted_copy(after));
    EXPECT_TRUE(std::is_sorted(after.begin(), after.end()));
}

TEST(StructuralMutationConformanceTest, DistributedRotatePreservesElementMultiset) {
    distributed_vector<int> vec(9, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);
    std::vector<int> before(local.begin(), local.end());

    mock_single_rank_comm comm;
    auto result = dtl::rotate(seq{}, vec, 4, comm);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(vec.structural_metadata_consistent());

    std::vector<int> after(local.begin(), local.end());
    EXPECT_EQ(sorted_copy(before), sorted_copy(after));
}

TEST(StructuralMutationConformanceTest, DistributedUniqueUpdatesGlobalMetadata) {
    distributed_vector<int> vec(9, test_context{0, 1});
    auto local = vec.local_view();
    local[0] = 1;
    local[1] = 1;
    local[2] = 1;
    local[3] = 2;
    local[4] = 3;
    local[5] = 3;
    local[6] = 4;
    local[7] = 4;
    local[8] = 5;

    mock_single_rank_comm comm;
    auto result = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result.value().new_size, 5u);
    EXPECT_EQ(result.value().removed_count, 4u);
    EXPECT_EQ(vec.size(), 5u);
    EXPECT_EQ(vec.local_size(), 5u);
    EXPECT_TRUE(vec.structural_metadata_consistent());

    auto compacted = vec.local_view();
    EXPECT_EQ(compacted[0], 1);
    EXPECT_EQ(compacted[1], 2);
    EXPECT_EQ(compacted[2], 3);
    EXPECT_EQ(compacted[3], 4);
    EXPECT_EQ(compacted[4], 5);
}

}  // namespace dtl::test
