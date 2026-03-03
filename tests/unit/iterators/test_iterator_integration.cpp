// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_iterator_integration.cpp
/// @brief Unit tests for iterator module integration
/// @details Phase 08, Task 07: Verify iterators are integrated into views/containers
///
/// Decision summary:
/// - local_iterator: Kept as public API but local_view uses T* iterators directly
///   (local_iterator is a tagged wrapper around T* for type safety in generic code)
/// - global_iterator: Integrated into global_view::begin()/end() (this phase)
/// - device_iterator: Kept as experimental for future GPU container iteration

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/iterators/global_iterator.hpp>
#include <dtl/iterators/local_iterator.hpp>
#include <dtl/iterators/device_iterator.hpp>
#include <dtl/views/global_view.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <type_traits>

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
// global_iterator Integration Tests (now wired into global_view)
// =============================================================================

TEST(IteratorIntegrationTest, GlobalViewReturnsGlobalIterator) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto gv = vec.global_view();

    static_assert(std::is_same_v<decltype(gv.begin()),
                                  global_iterator<distributed_vector<int>>>);
    static_assert(std::is_same_v<decltype(gv.end()),
                                  global_iterator<distributed_vector<int>>>);
}

TEST(IteratorIntegrationTest, GlobalViewBeginEnd) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto gv = vec.global_view();
    EXPECT_NE(gv.begin(), gv.end());

    auto it = gv.begin();
    EXPECT_EQ(it.global_index(), 0);

    auto ref = *it;
    EXPECT_EQ(ref.get().value(), 0);
}

TEST(IteratorIntegrationTest, GlobalViewIteration) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 10);  // {10, 11, 12, 13, 14}

    auto gv = vec.global_view();
    int count = 0;
    for (auto it = gv.begin(); it != gv.end(); ++it) {
        auto ref = *it;
        EXPECT_TRUE(ref.is_local());
        EXPECT_EQ(ref.get().value(), 10 + count);
        ++count;
    }
    EXPECT_EQ(count, 5);
}

TEST(IteratorIntegrationTest, ConstGlobalViewIteration) {
    distributed_vector<int> vec(5, 42, test_context{0, 1});
    const auto& cvec = vec;
    auto gv = cvec.global_view();

    auto it = gv.begin();
    auto ref = *it;
    EXPECT_EQ(ref.get().value(), 42);
}

TEST(IteratorIntegrationTest, EmptyGlobalView) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto gv = vec.global_view();
    EXPECT_EQ(gv.begin(), gv.end());
}

// =============================================================================
// local_iterator Standalone Tests (kept as public API)
// =============================================================================

TEST(IteratorIntegrationTest, LocalIteratorStandalone) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_iterator<distributed_vector<int>> it(data.data());
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);
}

TEST(IteratorIntegrationTest, LocalIteratorRandomAccess) {
    std::vector<int> data = {10, 20, 30, 40, 50};
    local_iterator<distributed_vector<int>> begin(data.data());
    local_iterator<distributed_vector<int>> end(data.data() + 5);

    EXPECT_EQ(end - begin, 5);
    EXPECT_EQ(begin[3], 40);
}

// =============================================================================
// device_iterator Tests (kept as experimental)
// =============================================================================

TEST(IteratorIntegrationTest, DeviceIteratorHostConstruction) {
    std::vector<int> data = {1, 2, 3};
    // device_iterator wraps a pointer — works on host for setup
    device_iterator<int> it(data.data());
    // Can read on host (though intended for device code)
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);
}

TEST(IteratorIntegrationTest, DeviceIteratorArithmetic) {
    std::vector<int> data = {10, 20, 30, 40, 50};
    device_iterator<int> begin(data.data());
    device_iterator<int> end(data.data() + 5);

    EXPECT_EQ(end - begin, 5);
    EXPECT_EQ((begin + 2) - begin, 2);
    EXPECT_EQ(begin[4], 50);
}

}  // namespace dtl::test
