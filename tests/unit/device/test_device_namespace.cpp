// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_namespace.cpp
/// @brief Unit tests for dtl::device:: vendor-agnostic namespace

#include <dtl/core/config.hpp>
#include <dtl/device/device.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// ============================================================================
// Device Query Tests (work with or without GPU)
// ============================================================================

TEST(DeviceNamespace, DeviceCountNonNegative) {
    int count = dtl::device::device_count();
    EXPECT_GE(count, 0);
}

TEST(DeviceNamespace, InvalidDeviceIdConstant) {
    EXPECT_EQ(dtl::device::invalid_device_id, -1);
}

TEST(DeviceNamespace, CurrentDeviceIdReturnsValidOrInvalid) {
    int id = dtl::device::current_device_id();
    if (dtl::device::device_count() > 0) {
        EXPECT_GE(id, 0);
    } else {
        // No GPU — should return invalid
        EXPECT_EQ(id, dtl::device::invalid_device_id);
    }
}

// ============================================================================
// Type Alias Verification (CUDA backend)
// ============================================================================

#if DTL_ENABLE_CUDA

TEST(DeviceNamespace, DeviceGuardIsCudaType) {
    static_assert(std::is_same_v<dtl::device::device_guard, dtl::cuda::device_guard>,
                  "dtl::device::device_guard must be dtl::cuda::device_guard when CUDA enabled");
    SUCCEED();
}

TEST(DeviceNamespace, StreamHandleIsCudaType) {
    static_assert(std::is_same_v<dtl::device::stream_handle, dtl::cuda::stream_handle>,
                  "dtl::device::stream_handle must be dtl::cuda::stream_handle when CUDA enabled");
    SUCCEED();
}

TEST(DeviceNamespace, DeviceBufferIsCudaType) {
    static_assert(std::is_same_v<dtl::device::device_buffer<float>,
                                 dtl::cuda::device_buffer<float>>,
                  "dtl::device::device_buffer<float> must be dtl::cuda::device_buffer<float>");
    SUCCEED();
}

TEST(DeviceNamespace, DeviceCountMatchesCuda) {
    EXPECT_EQ(dtl::device::device_count(), dtl::cuda::device_count());
}

TEST(DeviceNamespace, CurrentDeviceMatchesCuda) {
    EXPECT_EQ(dtl::device::current_device_id(), dtl::cuda::current_device_id());
}

// ============================================================================
// Runtime Tests (require GPU)
// ============================================================================

TEST(DeviceNamespace, DeviceGuardConstructAndDestruct) {
    if (dtl::device::device_count() == 0) {
        GTEST_SKIP() << "No GPU available";
    }

    {
        dtl::device::device_guard guard(0);
        EXPECT_EQ(dtl::device::current_device_id(), 0);
    }
    // Guard destroyed — previous device restored
}

TEST(DeviceNamespace, DeviceBufferAllocateAndFree) {
    if (dtl::device::device_count() == 0) {
        GTEST_SKIP() << "No GPU available";
    }

    dtl::device::device_buffer<float> buf(1024, 0);
    EXPECT_EQ(buf.size(), 1024u);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_FALSE(buf.empty());
}

TEST(DeviceNamespace, StreamHandleCreateAndSync) {
    if (dtl::device::device_count() == 0) {
        GTEST_SKIP() << "No GPU available";
    }

    dtl::device::stream_handle stream(true);  // Create new stream
    EXPECT_TRUE(stream.synchronize());
}

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// HIP Type Alias Verification
// ============================================================================

#if DTL_ENABLE_HIP && !DTL_ENABLE_CUDA

TEST(DeviceNamespace, DeviceGuardIsHipType) {
    static_assert(std::is_same_v<dtl::device::device_guard, dtl::hip::device_guard>,
                  "dtl::device::device_guard must be dtl::hip::device_guard when HIP enabled");
    SUCCEED();
}

TEST(DeviceNamespace, StreamHandleIsHipType) {
    static_assert(std::is_same_v<dtl::device::stream_handle, dtl::hip::stream_handle>,
                  "dtl::device::stream_handle must be dtl::hip::stream_handle when HIP enabled");
    SUCCEED();
}

#endif  // DTL_ENABLE_HIP && !DTL_ENABLE_CUDA

// ============================================================================
// No-GPU Stub Tests
// ============================================================================

#if !DTL_ENABLE_CUDA && !DTL_ENABLE_HIP

TEST(DeviceNamespace, StubDeviceCountZero) {
    EXPECT_EQ(dtl::device::device_count(), 0);
}

TEST(DeviceNamespace, StubCurrentDeviceInvalid) {
    EXPECT_EQ(dtl::device::current_device_id(), -1);
}

TEST(DeviceNamespace, StubStreamHandleCompiles) {
    dtl::device::stream_handle stream;
    EXPECT_TRUE(stream.is_default());
    EXPECT_TRUE(stream.synchronize());
}

TEST(DeviceNamespace, StubDeviceBufferCompiles) {
    dtl::device::device_buffer<float> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
}

#endif  // No GPU backend

}  // namespace dtl::test
