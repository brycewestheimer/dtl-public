// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_hip_memory_transfer.cpp
/// @brief Unit tests for HIP memory transfer operations
/// @details Tests API structure without requiring HIP hardware.

#include <backends/hip/hip_memory_transfer.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Transfer Direction Enum Tests
// =============================================================================

TEST(HipMemoryTransferTest, TransferDirectionEnum) {
    // All enum values must exist and be distinct
    auto h2d = dtl::hip::transfer_direction::host_to_device;
    auto d2h = dtl::hip::transfer_direction::device_to_host;
    auto d2d = dtl::hip::transfer_direction::device_to_device;
    auto h2h = dtl::hip::transfer_direction::host_to_host;
    auto automatic = dtl::hip::transfer_direction::automatic;

    EXPECT_NE(static_cast<int>(h2d), static_cast<int>(d2h));
    EXPECT_NE(static_cast<int>(d2h), static_cast<int>(d2d));
    EXPECT_NE(static_cast<int>(d2d), static_cast<int>(h2h));
    EXPECT_NE(static_cast<int>(h2h), static_cast<int>(automatic));
}

// =============================================================================
// Synchronous Transfer Stub Tests
// =============================================================================

TEST(HipMemoryTransferTest, HostToDeviceStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::copy_host_to_device(&dst, &src, sizeof(int));
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, DeviceToHostStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::copy_device_to_host(&dst, &src, sizeof(int));
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, DeviceToDeviceStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::copy_device_to_device(&dst, &src, sizeof(int));
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, AutoDirectionCopy) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::copy(&dst, &src, sizeof(int));
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

// =============================================================================
// Asynchronous Transfer Stub Tests
// =============================================================================

TEST(HipMemoryTransferTest, AsyncHostToDeviceStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::async_copy_host_to_device(
        &dst, &src, sizeof(int), nullptr);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, AsyncDeviceToHostStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::async_copy_device_to_host(
        &dst, &src, sizeof(int), nullptr);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, AsyncDeviceToDeviceStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::async_copy_device_to_device(
        &dst, &src, sizeof(int), nullptr);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, AsyncCopyStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::async_copy(
        &dst, &src, sizeof(int), nullptr);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

// =============================================================================
// 2D Transfer Stub Tests
// =============================================================================

TEST(HipMemoryTransferTest, Copy2DStub) {
    int src[4] = {1, 2, 3, 4};
    int dst[4] = {0, 0, 0, 0};
    auto result = dtl::hip::hip_memory_transfer::copy_2d(
        dst, sizeof(int) * 2,
        src, sizeof(int) * 2,
        sizeof(int) * 2, 2,
        dtl::hip::transfer_direction::host_to_device);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

// =============================================================================
// Pinned Memory Stub Tests
// =============================================================================

TEST(HipMemoryTransferTest, AllocatePinnedStub) {
    auto result = dtl::hip::hip_memory_transfer::allocate_pinned(1024);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, FreePinnedStub) {
    auto result = dtl::hip::hip_memory_transfer::free_pinned(nullptr);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

// =============================================================================
// Peer Access Stub Tests
// =============================================================================

TEST(HipMemoryTransferTest, PeerAccessStub) {
    auto result = dtl::hip::hip_memory_transfer::enable_peer_access(0, 1);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

TEST(HipMemoryTransferTest, CopyPeerStub) {
    int src = 42;
    int dst = 0;
    auto result = dtl::hip::hip_memory_transfer::copy_peer(&dst, 0, &src, 1, sizeof(int));
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
#endif
}

// =============================================================================
// Pinned Buffer Tests
// =============================================================================

TEST(HipMemoryTransferTest, PinnedBufferDefault) {
    dtl::hip::pinned_buffer<float> buf;
    EXPECT_FALSE(buf.valid());
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(HipMemoryTransferTest, PinnedBufferMove) {
    dtl::hip::pinned_buffer<float> buf1;
    dtl::hip::pinned_buffer<float> buf2(std::move(buf1));
    EXPECT_FALSE(buf2.valid());
    EXPECT_EQ(buf2.size(), 0u);
}

// =============================================================================
// Convenience Function Tests
// =============================================================================

TEST(HipMemoryTransferTest, CopyToDeviceHelper) {
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[4] = {};
    auto result = dtl::hip::copy_to_device(dst, src, 4);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
#endif
}

TEST(HipMemoryTransferTest, CopyToHostHelper) {
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[4] = {};
    auto result = dtl::hip::copy_to_host(dst, src, 4);
#if !DTL_ENABLE_HIP
    EXPECT_TRUE(result.has_error());
#endif
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(HipMemoryTransferTest, ClassNotCopyable) {
    static_assert(!std::is_copy_constructible_v<dtl::hip::hip_memory_transfer>,
                  "hip_memory_transfer should not be copyable");
    static_assert(!std::is_copy_assignable_v<dtl::hip::hip_memory_transfer>,
                  "hip_memory_transfer should not be copy-assignable");
}

TEST(HipMemoryTransferTest, ClassNotMovable) {
    static_assert(!std::is_move_constructible_v<dtl::hip::hip_memory_transfer>,
                  "hip_memory_transfer should not be movable");
    static_assert(!std::is_move_assignable_v<dtl::hip::hip_memory_transfer>,
                  "hip_memory_transfer should not be move-assignable");
}

}  // namespace dtl::test
