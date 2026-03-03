// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_unified_memory_prefetch.cpp
/// @brief Unit tests for prefetch policy helpers
/// @details Tests prefetch policy enum, string conversion, and hint structures.

#include <dtl/memory/prefetch_policy.hpp>

#include <gtest/gtest.h>

#include <string_view>

namespace dtl::test {

// =============================================================================
// Prefetch Policy Enum Tests
// =============================================================================

TEST(PrefetchPolicyTest, PrefetchPolicyEnum) {
    // All enum values must exist
    auto none = dtl::prefetch_policy::none;
    auto to_device = dtl::prefetch_policy::to_device;
    auto to_host = dtl::prefetch_policy::to_host;
    auto bidir = dtl::prefetch_policy::bidirectional;

    (void)none; (void)to_device; (void)to_host; (void)bidir;
    SUCCEED();
}

TEST(PrefetchPolicyTest, NonePolicy) {
    EXPECT_EQ(static_cast<int>(dtl::prefetch_policy::none), 0);
}

TEST(PrefetchPolicyTest, ToDevicePolicy) {
    auto policy = dtl::prefetch_policy::to_device;
    EXPECT_NE(static_cast<int>(policy), static_cast<int>(dtl::prefetch_policy::none));
}

TEST(PrefetchPolicyTest, ToHostPolicy) {
    auto policy = dtl::prefetch_policy::to_host;
    EXPECT_NE(static_cast<int>(policy), static_cast<int>(dtl::prefetch_policy::none));
    EXPECT_NE(static_cast<int>(policy), static_cast<int>(dtl::prefetch_policy::to_device));
}

TEST(PrefetchPolicyTest, BidirectionalPolicy) {
    auto policy = dtl::prefetch_policy::bidirectional;
    EXPECT_NE(static_cast<int>(policy), static_cast<int>(dtl::prefetch_policy::none));
    EXPECT_NE(static_cast<int>(policy), static_cast<int>(dtl::prefetch_policy::to_device));
    EXPECT_NE(static_cast<int>(policy), static_cast<int>(dtl::prefetch_policy::to_host));
}

TEST(PrefetchPolicyTest, PrefetchPolicyDistinct) {
    // All four values must be distinct
    auto none = static_cast<int>(dtl::prefetch_policy::none);
    auto to_device = static_cast<int>(dtl::prefetch_policy::to_device);
    auto to_host = static_cast<int>(dtl::prefetch_policy::to_host);
    auto bidir = static_cast<int>(dtl::prefetch_policy::bidirectional);

    EXPECT_NE(none, to_device);
    EXPECT_NE(none, to_host);
    EXPECT_NE(none, bidir);
    EXPECT_NE(to_device, to_host);
    EXPECT_NE(to_device, bidir);
    EXPECT_NE(to_host, bidir);
}

// =============================================================================
// String Conversion Tests
// =============================================================================

TEST(PrefetchPolicyTest, ToStringNone) {
    auto str = dtl::to_string(dtl::prefetch_policy::none);
    EXPECT_EQ(str, "none");
}

TEST(PrefetchPolicyTest, ToStringToDevice) {
    auto str = dtl::to_string(dtl::prefetch_policy::to_device);
    EXPECT_EQ(str, "to_device");
}

TEST(PrefetchPolicyTest, ToStringToHost) {
    auto str = dtl::to_string(dtl::prefetch_policy::to_host);
    EXPECT_EQ(str, "to_host");
}

TEST(PrefetchPolicyTest, ToStringBidirectional) {
    auto str = dtl::to_string(dtl::prefetch_policy::bidirectional);
    EXPECT_EQ(str, "bidirectional");
}

// =============================================================================
// Prefetch Hint Tests
// =============================================================================

TEST(PrefetchPolicyTest, PrefetchHintDefault) {
    dtl::prefetch_hint hint{};
    EXPECT_EQ(hint.policy, dtl::prefetch_policy::none);
    EXPECT_EQ(hint.device_id, 0);
    EXPECT_EQ(hint.offset, 0u);
    EXPECT_EQ(hint.size, 0u);
}

TEST(PrefetchPolicyTest, MakeDevicePrefetch) {
    auto hint = dtl::make_device_prefetch(2);
    EXPECT_EQ(hint.policy, dtl::prefetch_policy::to_device);
    EXPECT_EQ(hint.device_id, 2);
    EXPECT_EQ(hint.offset, 0u);
    EXPECT_EQ(hint.size, 0u);
}

TEST(PrefetchPolicyTest, MakeHostPrefetch) {
    auto hint = dtl::make_host_prefetch();
    EXPECT_EQ(hint.policy, dtl::prefetch_policy::to_host);
    EXPECT_EQ(hint.device_id, 0);
    EXPECT_EQ(hint.offset, 0u);
    EXPECT_EQ(hint.size, 0u);
}

TEST(PrefetchPolicyTest, PrefetchHintDeviceId) {
    dtl::prefetch_hint hint{};
    hint.device_id = 3;
    EXPECT_EQ(hint.device_id, 3);
}

TEST(PrefetchPolicyTest, PrefetchHintSizeOffset) {
    dtl::prefetch_hint hint{};
    hint.offset = 1024;
    hint.size = 4096;
    EXPECT_EQ(hint.offset, 1024u);
    EXPECT_EQ(hint.size, 4096u);
}

}  // namespace dtl::test
