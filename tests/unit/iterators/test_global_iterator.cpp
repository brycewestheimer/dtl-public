// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_global_iterator.cpp
/// @brief Unit tests for global_iterator
/// @details Phase 08, Task 06: Verify global_iterator::operator* returns actual values

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/iterators/global_iterator.hpp>

#include <gtest/gtest.h>

#include <numeric>

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
// Dereference Tests (was previously a stub returning T{})
// =============================================================================

TEST(GlobalIteratorTest, DereferenceReturnsCorrectValue) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);  // {0,1,2,...,9}

    global_iterator<distributed_vector<int>> it(&vec, 0);
    auto ref = *it;
    EXPECT_TRUE(ref.is_local());
    EXPECT_EQ(ref.get().value(), 0);
}

TEST(GlobalIteratorTest, DereferenceMultiplePositions) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 100);

    global_iterator<distributed_vector<int>> it(&vec, 0);
    for (int i = 0; i < 10; ++i) {
        auto ref = *it;
        EXPECT_TRUE(ref.is_local());
        EXPECT_EQ(ref.get().value(), 100 + i);
        ++it;
    }
}

TEST(GlobalIteratorTest, DereferenceNonZeroStart) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    global_iterator<distributed_vector<int>> it(&vec, 5);
    auto ref = *it;
    EXPECT_EQ(ref.get().value(), 5);
}

// =============================================================================
// Remote Element Detection
// =============================================================================

TEST(GlobalIteratorTest, RemoteElementForMultiRank) {
    distributed_vector<int> vec(100, test_context{0, 4});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    // Index 0 is local to rank 0
    global_iterator<distributed_vector<int>> local_it(&vec, 0);
    EXPECT_TRUE(local_it.is_local());

    // Index 50 is on rank 2 (remote)
    global_iterator<distributed_vector<int>> remote_it(&vec, 50);
    EXPECT_FALSE(remote_it.is_local());

    auto ref = *remote_it;
    EXPECT_TRUE(ref.is_remote());
}

// =============================================================================
// Iterator Traversal
// =============================================================================

TEST(GlobalIteratorTest, PreIncrement) {
    distributed_vector<int> vec(10, test_context{0, 1});
    global_iterator<distributed_vector<int>> it(&vec, 0);

    ++it;
    EXPECT_EQ(it.global_index(), 1);
    ++it;
    EXPECT_EQ(it.global_index(), 2);
}

TEST(GlobalIteratorTest, PostIncrement) {
    distributed_vector<int> vec(10, test_context{0, 1});
    global_iterator<distributed_vector<int>> it(&vec, 0);

    auto old = it++;
    EXPECT_EQ(old.global_index(), 0);
    EXPECT_EQ(it.global_index(), 1);
}

// =============================================================================
// Comparison
// =============================================================================

TEST(GlobalIteratorTest, Equality) {
    distributed_vector<int> vec(10, test_context{0, 1});
    global_iterator<distributed_vector<int>> a(&vec, 5);
    global_iterator<distributed_vector<int>> b(&vec, 5);
    global_iterator<distributed_vector<int>> c(&vec, 3);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
}

// =============================================================================
// Query Methods
// =============================================================================

TEST(GlobalIteratorTest, OwnerQuery) {
    distributed_vector<int> vec(100, test_context{0, 4});

    global_iterator<distributed_vector<int>> it0(&vec, 0);
    EXPECT_EQ(it0.owner(), 0);

    global_iterator<distributed_vector<int>> it75(&vec, 75);
    EXPECT_EQ(it75.owner(), 3);
}

TEST(GlobalIteratorTest, DefaultConstructorSafe) {
    global_iterator<distributed_vector<int>> it;
    EXPECT_FALSE(it.is_local());
    EXPECT_EQ(it.global_index(), 0);
}

// =============================================================================
// Const Iterator
// =============================================================================

TEST(GlobalIteratorTest, ConstIteratorDeref) {
    distributed_vector<int> vec(10, 42, test_context{0, 1});

    const_global_iterator<distributed_vector<int>> cit(&vec, 0);
    auto ref = *cit;
    EXPECT_TRUE(ref.is_local());
    EXPECT_EQ(ref.get().value(), 42);
}

}  // namespace dtl::test
