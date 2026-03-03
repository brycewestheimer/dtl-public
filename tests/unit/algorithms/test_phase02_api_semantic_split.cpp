// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_phase02_api_semantic_split.cpp
/// @brief Phase 02 API semantic split tests

#include <dtl/algorithms/algorithms.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/execution/seq.hpp>
#include "mock_single_rank_comm.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>

namespace dtl::test {

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;
    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};

}  // namespace

TEST(Phase02ApiSemanticSplit, ReduceNoCommThrowsForMultiRank) {
    distributed_vector<int> vec(8, test_context{0, 2});
    EXPECT_THROW((void)dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}), std::runtime_error);
}

TEST(Phase02ApiSemanticSplit, CountNoCommThrowsForMultiRank) {
    distributed_vector<int> vec(8, test_context{0, 2});
    EXPECT_THROW((void)dtl::count(dtl::seq{}, vec, 1), std::runtime_error);
}

TEST(Phase02ApiSemanticSplit, FindNoCommThrowsForMultiRank) {
    distributed_vector<int> vec(8, test_context{0, 2});
    EXPECT_THROW((void)dtl::find(dtl::seq{}, vec, 1), std::runtime_error);
}

TEST(Phase02ApiSemanticSplit, InclusiveScanNoCommFailsForMultiRank) {
    distributed_vector<int> input(8, test_context{0, 2});
    distributed_vector<int> output(8, test_context{0, 2});

    auto res = dtl::inclusive_scan(dtl::seq{}, input, output, 0, std::plus<>{});
    EXPECT_FALSE(res.has_value());
    EXPECT_EQ(res.error().code(), status_code::precondition_failed);
}

TEST(Phase02ApiSemanticSplit, DegenerateSingleRankStillWorks) {
    distributed_vector<int> vec(4, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1;
    lv[1] = 2;
    lv[2] = 3;
    lv[3] = 4;

    EXPECT_EQ(dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}), 10);
    EXPECT_EQ(dtl::count(dtl::seq{}, vec, 3), 1);

    auto found = dtl::find(dtl::seq{}, vec, 2);
    EXPECT_TRUE(found.found);
    EXPECT_EQ(found.global_index, 1);
}

TEST(Phase02ApiSemanticSplit, DistributedReduceMetadataTruthfulSingleRank) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 2;
    lv[1] = 3;
    lv[2] = 5;

    auto result = dtl::distributed_reduce(dtl::seq{}, vec, 0, std::plus<>{});
    EXPECT_TRUE(result.has_global);
    EXPECT_EQ(result.local_value, 10);
    EXPECT_EQ(result.global_value, 10);
}

TEST(Phase02ApiSemanticSplit, CanonicalDomainNamespacesAreUsable) {
    distributed_vector<int> vec(4, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1;
    lv[1] = 2;
    lv[2] = 3;
    lv[3] = 4;

    mock_single_rank_comm comm{};

    EXPECT_EQ(dtl::algorithms::local::reduce(vec, 0, std::plus<>{}), 10);
    EXPECT_EQ(dtl::algorithms::collective::reduce(dtl::seq{}, vec, 0, std::plus<>{}, comm), 10);

    auto distributed = dtl::algorithms::distributed::reduce(dtl::seq{}, vec, 0, std::plus<>{});
    EXPECT_TRUE(distributed.has_global);
    EXPECT_EQ(distributed.global_value, 10);
}

}  // namespace dtl::test
