// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_tensor_device_affinity.cpp
/// @brief Unit tests for distributed_tensor device affinity
/// @details Phase 08, Task 02: Verify device affinity parity with distributed_vector

#include <dtl/containers/distributed_tensor.hpp>
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

TEST(TensorDeviceAffinityTest, DefaultHostOnly) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    EXPECT_FALSE(tensor.has_device_affinity());
    EXPECT_EQ(tensor.device_id(), detail::no_device_affinity);
}

TEST(TensorDeviceAffinityTest, HostAccessible) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    EXPECT_TRUE(tensor.is_host_accessible());
}

TEST(TensorDeviceAffinityTest, AllocatorTypeMatchesPolicy) {
    using tensor_type = distributed_tensor<int, 2>;
    using expected_alloc = select_allocator_t<int, typename tensor_type::placement_policy>;
    static_assert(std::is_same_v<tensor_type::allocator_type, expected_alloc>);
}

TEST(TensorDeviceAffinityTest, GetAllocator) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    auto alloc = tensor.get_allocator();
    (void)alloc;
}

TEST(TensorDeviceAffinityTest, MultiRankHostOnly) {
    distributed_tensor<double, 3> tensor({10, 10, 10}, test_context{2, 4});
    EXPECT_FALSE(tensor.has_device_affinity());
    EXPECT_TRUE(tensor.is_host_accessible());
}

}  // namespace dtl::test
