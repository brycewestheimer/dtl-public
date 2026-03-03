// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_send_modes.cpp
/// @brief Unit tests for send_mode enum and to_string (Phase 12B)
/// @details Tests the four MPI send mode variants and their string
///          representations defined in send_mode.hpp.

#include <dtl/communication/send_mode.hpp>

#include <gtest/gtest.h>

#include <set>
#include <string_view>

namespace dtl::test {

// =============================================================================
// Enum Value Tests
// =============================================================================

TEST(SendModeTest, StandardMode) {
    auto mode = send_mode::standard;
    EXPECT_EQ(mode, send_mode::standard);
}

TEST(SendModeTest, SynchronousMode) {
    auto mode = send_mode::synchronous;
    EXPECT_EQ(mode, send_mode::synchronous);
}

TEST(SendModeTest, ReadyMode) {
    auto mode = send_mode::ready;
    EXPECT_EQ(mode, send_mode::ready);
}

TEST(SendModeTest, BufferedMode) {
    auto mode = send_mode::buffered;
    EXPECT_EQ(mode, send_mode::buffered);
}

// =============================================================================
// to_string Tests
// =============================================================================

TEST(SendModeTest, ToStringStandard) {
    EXPECT_EQ(to_string(send_mode::standard), "standard");
}

TEST(SendModeTest, ToStringSynchronous) {
    EXPECT_EQ(to_string(send_mode::synchronous), "synchronous");
}

TEST(SendModeTest, ToStringReady) {
    EXPECT_EQ(to_string(send_mode::ready), "ready");
}

TEST(SendModeTest, ToStringBuffered) {
    EXPECT_EQ(to_string(send_mode::buffered), "buffered");
}

// =============================================================================
// Enum Property Tests
// =============================================================================

TEST(SendModeTest, ModeDistinct) {
    // All 4 modes must be distinct values
    std::set<int> values;
    values.insert(static_cast<int>(send_mode::standard));
    values.insert(static_cast<int>(send_mode::synchronous));
    values.insert(static_cast<int>(send_mode::ready));
    values.insert(static_cast<int>(send_mode::buffered));
    EXPECT_EQ(values.size(), 4u);
}

TEST(SendModeTest, ModeEnumSize) {
    // Verify all 4 modes can be cast to int without issues
    int s = static_cast<int>(send_mode::standard);
    int y = static_cast<int>(send_mode::synchronous);
    int r = static_cast<int>(send_mode::ready);
    int b = static_cast<int>(send_mode::buffered);

    // Each cast should produce a valid integer
    (void)s;
    (void)y;
    (void)r;
    (void)b;

    // Verify they can be compared
    EXPECT_NE(s, y);
    EXPECT_NE(y, r);
    EXPECT_NE(r, b);
}

}  // namespace dtl::test
