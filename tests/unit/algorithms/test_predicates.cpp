// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_predicates.cpp
/// @brief Unit tests for predicate algorithms: all_of, any_of, none_of (Phase 06 T08)

#include <dtl/algorithms/non_modifying/predicates.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

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

// =============================================================================
// all_of Tests
// =============================================================================

TEST(AllOfTest, AllPositive) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    EXPECT_TRUE(dtl::local_all_of(vec, [](int x) { return x > 0; }));
}

TEST(AllOfTest, NotAllPositive) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = -2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    EXPECT_FALSE(dtl::local_all_of(vec, [](int x) { return x > 0; }));
}

TEST(AllOfTest, EmptyContainerReturnsTrue) {
    distributed_vector<int> vec(0, test_context{0, 1});
    // all_of on empty range is vacuously true
    EXPECT_TRUE(dtl::local_all_of(vec, [](int x) { return x > 0; }));
}

TEST(AllOfTest, SingleElementTrue) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 42;
    EXPECT_TRUE(dtl::local_all_of(vec, [](int x) { return x == 42; }));
}

TEST(AllOfTest, SingleElementFalse) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 42;
    EXPECT_FALSE(dtl::local_all_of(vec, [](int x) { return x == 0; }));
}

// =============================================================================
// any_of Tests
// =============================================================================

TEST(AnyOfTest, SomeMatch) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    EXPECT_TRUE(dtl::local_any_of(vec, [](int x) { return x == 3; }));
}

TEST(AnyOfTest, NoneMatch) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    EXPECT_FALSE(dtl::local_any_of(vec, [](int x) { return x > 10; }));
}

TEST(AnyOfTest, AllMatch) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3;

    EXPECT_TRUE(dtl::local_any_of(vec, [](int x) { return x > 0; }));
}

TEST(AnyOfTest, EmptyContainerReturnsFalse) {
    distributed_vector<int> vec(0, test_context{0, 1});
    EXPECT_FALSE(dtl::local_any_of(vec, [](int x) { return x > 0; }));
}

// =============================================================================
// none_of Tests
// =============================================================================

TEST(NoneOfTest, NoneMatch) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    EXPECT_TRUE(dtl::local_none_of(vec, [](int x) { return x < 0; }));
}

TEST(NoneOfTest, SomeMatch) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = -2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    EXPECT_FALSE(dtl::local_none_of(vec, [](int x) { return x < 0; }));
}

TEST(NoneOfTest, EmptyContainerReturnsTrue) {
    distributed_vector<int> vec(0, test_context{0, 1});
    EXPECT_TRUE(dtl::local_none_of(vec, [](int x) { return x > 0; }));
}

TEST(NoneOfTest, SingleElementNoMatch) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 5;
    EXPECT_TRUE(dtl::local_none_of(vec, [](int x) { return x < 0; }));
}

}  // namespace dtl::test
