// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_guard.cpp
/// @brief Unit tests for CUDA device guard
/// @details Tests device guard construction, switching, and restoration.

#include <dtl/core/config.hpp>
#include <dtl/cuda/device_guard.hpp>

#include <gtest/gtest.h>

namespace dtl::cuda::test {

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST(CudaDeviceGuardTest, ConstructWithInvalidDevice) {
    // Should be a no-op when constructing with invalid device
    device_guard guard(invalid_device_id);
    EXPECT_EQ(guard.target_device(), invalid_device_id);
    EXPECT_FALSE(guard.switched());
}

#if DTL_ENABLE_CUDA

TEST(CudaDeviceGuardTest, AvailableReturnsCorrectValue) {
    // Check if any CUDA devices are available
    bool available = device_guard::available();

    int count = device_count();
    EXPECT_EQ(available, count > 0);
}

TEST(CudaDeviceGuardTest, CurrentDeviceReturnsValidDevice) {
    if (!device_guard::available()) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    int current = current_device_id();
    EXPECT_GE(current, 0);
    EXPECT_LT(current, device_count());
}

TEST(CudaDeviceGuardTest, GuardSetsDevice) {
    if (!device_guard::available()) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    int original = current_device_id();

    {
        device_guard guard(0);
        EXPECT_EQ(guard.target_device(), 0);
        EXPECT_EQ(current_device_id(), 0);
    }

    // Device should be restored
    EXPECT_EQ(current_device_id(), original);
}

TEST(CudaDeviceGuardTest, GuardRestoresPreviousDevice) {
    if (device_count() < 2) {
        GTEST_SKIP() << "Need at least 2 CUDA devices for this test";
    }

    // Start on device 0
    cudaSetDevice(0);
    EXPECT_EQ(current_device_id(), 0);

    {
        device_guard guard(1);
        EXPECT_EQ(current_device_id(), 1);
        EXPECT_EQ(guard.previous_device(), 0);
        EXPECT_TRUE(guard.switched());
    }

    // Should be back on device 0
    EXPECT_EQ(current_device_id(), 0);
}

TEST(CudaDeviceGuardTest, NestedGuards) {
    if (device_count() < 3) {
        GTEST_SKIP() << "Need at least 3 CUDA devices for this test";
    }

    cudaSetDevice(0);
    int original = current_device_id();
    EXPECT_EQ(original, 0);

    {
        device_guard guard1(1);
        EXPECT_EQ(current_device_id(), 1);

        {
            device_guard guard2(2);
            EXPECT_EQ(current_device_id(), 2);
        }
        // guard2 destroyed, back to 1
        EXPECT_EQ(current_device_id(), 1);
    }
    // guard1 destroyed, back to original
    EXPECT_EQ(current_device_id(), original);
}

TEST(CudaDeviceGuardTest, GuardNoSwitchIfAlreadyOnDevice) {
    if (!device_guard::available()) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    int current = current_device_id();

    {
        device_guard guard(current);
        EXPECT_EQ(guard.target_device(), current);
        // If already on target device, no switch needed
        EXPECT_FALSE(guard.switched());
    }

    EXPECT_EQ(current_device_id(), current);
}

#else  // !DTL_ENABLE_CUDA

TEST(CudaDeviceGuardTest, AvailableReturnsFalseWhenCudaDisabled) {
    EXPECT_FALSE(device_guard::available());
}

TEST(CudaDeviceGuardTest, CurrentDeviceReturnsInvalidWhenCudaDisabled) {
    EXPECT_EQ(current_device_id(), invalid_device_id);
}

TEST(CudaDeviceGuardTest, DeviceCountReturnsZeroWhenCudaDisabled) {
    EXPECT_EQ(device_count(), 0);
}

TEST(CudaDeviceGuardTest, GuardIsNoOpWhenCudaDisabled) {
    device_guard guard(0);
    EXPECT_EQ(guard.target_device(), 0);
    EXPECT_FALSE(guard.switched());
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl::cuda::test
