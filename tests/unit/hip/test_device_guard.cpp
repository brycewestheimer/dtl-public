// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_guard.cpp
/// @brief Unit tests for HIP device guard
/// @details Tests device guard construction, switching, and restoration.

#include <dtl/core/config.hpp>
#include <dtl/hip/device_guard.hpp>

#include <gtest/gtest.h>

namespace dtl::hip::test {

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST(HipDeviceGuardTest, ConstructWithInvalidDevice) {
    // Should be a no-op when constructing with invalid device
    device_guard guard(invalid_device_id);
    EXPECT_EQ(guard.target_device(), invalid_device_id);
    EXPECT_FALSE(guard.switched());
}

#if DTL_ENABLE_HIP

TEST(HipDeviceGuardTest, AvailableReturnsCorrectValue) {
    bool available = device_guard::available();

    int count = device_count();
    EXPECT_EQ(available, count > 0);
}

TEST(HipDeviceGuardTest, CurrentDeviceReturnsValidDevice) {
    if (!device_guard::available()) {
        GTEST_SKIP() << "No HIP devices available";
    }

    int current = current_device_id();
    EXPECT_GE(current, 0);
    EXPECT_LT(current, device_count());
}

TEST(HipDeviceGuardTest, GuardSetsDevice) {
    if (!device_guard::available()) {
        GTEST_SKIP() << "No HIP devices available";
    }

    int original = current_device_id();

    {
        device_guard guard(0);
        EXPECT_EQ(guard.target_device(), 0);
        EXPECT_EQ(current_device_id(), 0);
    }

    EXPECT_EQ(current_device_id(), original);
}

TEST(HipDeviceGuardTest, GuardRestoresPreviousDevice) {
    if (device_count() < 2) {
        GTEST_SKIP() << "Need at least 2 HIP devices for this test";
    }

    hipSetDevice(0);
    EXPECT_EQ(current_device_id(), 0);

    {
        device_guard guard(1);
        EXPECT_EQ(current_device_id(), 1);
        EXPECT_EQ(guard.previous_device(), 0);
        EXPECT_TRUE(guard.switched());
    }

    EXPECT_EQ(current_device_id(), 0);
}

TEST(HipDeviceGuardTest, GuardNoSwitchIfAlreadyOnDevice) {
    if (!device_guard::available()) {
        GTEST_SKIP() << "No HIP devices available";
    }

    int current = current_device_id();

    {
        device_guard guard(current);
        EXPECT_EQ(guard.target_device(), current);
        EXPECT_FALSE(guard.switched());
    }

    EXPECT_EQ(current_device_id(), current);
}

#else  // !DTL_ENABLE_HIP

TEST(HipDeviceGuardTest, AvailableReturnsFalseWhenHipDisabled) {
    EXPECT_FALSE(device_guard::available());
}

TEST(HipDeviceGuardTest, CurrentDeviceReturnsInvalidWhenHipDisabled) {
    EXPECT_EQ(current_device_id(), invalid_device_id);
}

TEST(HipDeviceGuardTest, DeviceCountReturnsZeroWhenHipDisabled) {
    EXPECT_EQ(device_count(), 0);
}

TEST(HipDeviceGuardTest, GuardIsNoOpWhenHipDisabled) {
    device_guard guard(0);
    EXPECT_EQ(guard.target_device(), 0);
    EXPECT_FALSE(guard.switched());
}

#endif  // DTL_ENABLE_HIP

}  // namespace dtl::hip::test
