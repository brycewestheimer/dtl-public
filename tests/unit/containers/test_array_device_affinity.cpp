// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_array_device_affinity.cpp
/// @brief Unit tests for distributed_array device affinity
/// @details Phase 08, Task 02: Verify device affinity parity with distributed_vector

#include <dtl/containers/distributed_array.hpp>
#include <dtl/containers/detail/device_affinity.hpp>

#include <gtest/gtest.h>

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
// Device Affinity Tests
// =============================================================================

TEST(ArrayDeviceAffinityTest, DefaultHostOnly) {
    distributed_array<int, 100> arr;
    // Default placement is host_only — no device affinity
    EXPECT_FALSE(arr.has_device_affinity());
    EXPECT_EQ(arr.device_id(), detail::no_device_affinity);
}

TEST(ArrayDeviceAffinityTest, HostOnlyWithContext) {
    distributed_array<int, 100> arr(test_context{0, 1});
    EXPECT_FALSE(arr.has_device_affinity());
    EXPECT_TRUE(arr.is_host_accessible());
}

TEST(ArrayDeviceAffinityTest, AllocatorTypeMatchesPolicy) {
    // The allocator_type should be select_allocator_t<T, placement_policy>
    using array_type = distributed_array<int, 100>;
    using expected_alloc = select_allocator_t<int, typename array_type::placement_policy>;
    static_assert(std::is_same_v<array_type::allocator_type, expected_alloc>);
}

TEST(ArrayDeviceAffinityTest, GetAllocator) {
    distributed_array<int, 100> arr;
    auto alloc = arr.get_allocator();
    // Should not throw — allocator is valid
    (void)alloc;
}

TEST(ArrayDeviceAffinityTest, MultiRankHostAccessible) {
    distributed_array<int, 100> arr(test_context{1, 4});
    EXPECT_TRUE(arr.is_host_accessible());
    EXPECT_FALSE(arr.has_device_affinity());
}

}  // namespace dtl::test
