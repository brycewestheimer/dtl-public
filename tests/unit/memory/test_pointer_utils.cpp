// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_pointer_utils.cpp
/// @brief Tests for pointer kind detection and GPU-aware MPI utilities
/// @details Tests the host-only code paths of pointer_utils.hpp. When CUDA
///          is not enabled, pointer queries return unknown and staging buffers
///          pass through the original pointer without copying.

#include <dtl/memory/pointer_utils.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

// ============================================================================
// pointer_kind enum tests
// ============================================================================

TEST(PointerUtilsTest, NullPointerReturnsUnknown) {
    EXPECT_EQ(query_pointer_kind(nullptr), pointer_kind::unknown);
}

TEST(PointerUtilsTest, HostHeapPointer) {
    // Regular heap allocation - should be unregistered or unknown depending on CUDA availability
    auto* ptr = new int(42);
    auto kind = query_pointer_kind(ptr);
    // Without CUDA runtime: unknown. With CUDA: host or unregistered
    EXPECT_TRUE(kind == pointer_kind::unknown ||
                kind == pointer_kind::unregistered ||
                kind == pointer_kind::host);
    delete ptr;
}

TEST(PointerUtilsTest, StackPointerClassification) {
    int stack_var = 42;
    auto kind = query_pointer_kind(&stack_var);
    EXPECT_TRUE(kind == pointer_kind::unknown ||
                kind == pointer_kind::unregistered ||
                kind == pointer_kind::host);
}

TEST(PointerUtilsTest, VectorDataPointerClassification) {
    std::vector<double> vec(100, 3.14);
    auto kind = query_pointer_kind(vec.data());
    EXPECT_TRUE(kind == pointer_kind::unknown ||
                kind == pointer_kind::unregistered ||
                kind == pointer_kind::host);
}

TEST(PointerUtilsTest, HostAccessibleForHostPointer) {
    int value = 42;
    EXPECT_TRUE(is_host_accessible(&value));
}

TEST(PointerUtilsTest, DeviceAccessibleForHostPointer) {
    int value = 42;
    // Host memory is not device accessible (unless managed)
    EXPECT_FALSE(is_device_accessible(&value));
}

TEST(PointerUtilsTest, HostAccessibleForHeapPointer) {
    auto* ptr = new double(1.0);
    EXPECT_TRUE(is_host_accessible(ptr));
    delete ptr;
}

TEST(PointerUtilsTest, DeviceAccessibleForHeapPointer) {
    auto* ptr = new double(1.0);
    EXPECT_FALSE(is_device_accessible(ptr));
    delete ptr;
}

TEST(PointerUtilsTest, NullPointerHostAccessible) {
    // Null pointer returns unknown, which is treated as host-accessible
    EXPECT_TRUE(is_host_accessible(nullptr));
}

TEST(PointerUtilsTest, NullPointerNotDeviceAccessible) {
    EXPECT_FALSE(is_device_accessible(nullptr));
}

// ============================================================================
// GPU-aware MPI detection
// ============================================================================

TEST(PointerUtilsTest, GpuAwareMpiConsistent) {
    // Calling twice should return the same value (cached)
    bool first = is_gpu_aware_mpi();
    bool second = is_gpu_aware_mpi();
    EXPECT_EQ(first, second);
}

TEST(PointerUtilsTest, GpuAwareMpiReturnsBool) {
    // Just verify it returns without crashing
    [[maybe_unused]] bool result = is_gpu_aware_mpi();
}

#if !DTL_ENABLE_CUDA
TEST(PointerUtilsTest, GpuAwareMpiFalseWithoutCuda) {
    // Without CUDA, GPU-aware MPI is always false
    EXPECT_FALSE(is_gpu_aware_mpi());
}
#endif

// ============================================================================
// device_staging_buffer tests (host-only paths)
// ============================================================================

TEST(DeviceStagingBufferTest, HostPointerNoStaging) {
    int value = 42;
    device_staging_buffer buf(&value, sizeof(int), true);
    // For host memory, no staging needed
    EXPECT_FALSE(buf.is_staged());
    // data() should return the original pointer
    EXPECT_EQ(buf.data(), &value);
}

TEST(DeviceStagingBufferTest, HostPointerNoStagingRecv) {
    int value = 0;
    device_staging_buffer buf(&value, sizeof(int), false);
    EXPECT_FALSE(buf.is_staged());
    EXPECT_EQ(buf.data(), &value);
}

TEST(DeviceStagingBufferTest, NullPointerHandled) {
    // Null pointer should not crash
    device_staging_buffer buf(nullptr, 0, true);
    EXPECT_FALSE(buf.is_staged());
}

TEST(DeviceStagingBufferTest, NullPointerRecvHandled) {
    device_staging_buffer buf(nullptr, 0, false);
    EXPECT_FALSE(buf.is_staged());
}

TEST(DeviceStagingBufferTest, ConstDataAccessMatchesMutableData) {
    int value = 42;
    device_staging_buffer buf(&value, sizeof(int), true);
    const auto& cbuf = buf;
    EXPECT_EQ(buf.data(), cbuf.data());
}

TEST(DeviceStagingBufferTest, LargeHostBufferNoStaging) {
    std::vector<double> data(1000, 1.5);
    device_staging_buffer buf(data.data(), data.size() * sizeof(double), true);
    EXPECT_FALSE(buf.is_staged());
    EXPECT_EQ(buf.data(), data.data());
}

TEST(DeviceStagingBufferTest, MoveConstructor) {
    int value = 42;
    device_staging_buffer buf1(&value, sizeof(int), true);
    device_staging_buffer buf2(std::move(buf1));
    EXPECT_FALSE(buf2.is_staged());
    EXPECT_EQ(buf2.data(), &value);
}

// ============================================================================
// Compile-time checks
// ============================================================================

TEST(PointerUtilsTest, EnumValues) {
    // Verify all enum values are distinct
    EXPECT_NE(static_cast<int>(pointer_kind::host), static_cast<int>(pointer_kind::device));
    EXPECT_NE(static_cast<int>(pointer_kind::device), static_cast<int>(pointer_kind::managed));
    EXPECT_NE(static_cast<int>(pointer_kind::managed), static_cast<int>(pointer_kind::unregistered));
    EXPECT_NE(static_cast<int>(pointer_kind::unregistered), static_cast<int>(pointer_kind::unknown));
}

TEST(PointerUtilsTest, EnumAllValuesDistinct) {
    // Full pairwise check of all enum values
    constexpr int host_val = static_cast<int>(pointer_kind::host);
    constexpr int device_val = static_cast<int>(pointer_kind::device);
    constexpr int managed_val = static_cast<int>(pointer_kind::managed);
    constexpr int unreg_val = static_cast<int>(pointer_kind::unregistered);
    constexpr int unknown_val = static_cast<int>(pointer_kind::unknown);

    EXPECT_NE(host_val, device_val);
    EXPECT_NE(host_val, managed_val);
    EXPECT_NE(host_val, unreg_val);
    EXPECT_NE(host_val, unknown_val);
    EXPECT_NE(device_val, managed_val);
    EXPECT_NE(device_val, unreg_val);
    EXPECT_NE(device_val, unknown_val);
    EXPECT_NE(managed_val, unreg_val);
    EXPECT_NE(managed_val, unknown_val);
    EXPECT_NE(unreg_val, unknown_val);
}

}  // namespace dtl::test
