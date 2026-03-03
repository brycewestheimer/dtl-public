// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_hip_device_buffer.cpp
/// @brief Unit tests for dtl::hip::device_buffer
/// @details Tests RAII device memory management for HIP.
///          When HIP is not available, placeholder tests document expected API.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_HIP
#include <dtl/hip/device_buffer.hpp>
#include <hip/hip_runtime.h>
#endif

#include <gtest/gtest.h>

namespace dtl::test {

#if DTL_ENABLE_HIP

TEST(HipDeviceBufferTest, DefaultConstruct) {
    dtl::hip::device_buffer<float> buf;
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.capacity(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_TRUE(buf.empty());
}

TEST(HipDeviceBufferTest, SizedConstruct) {
    dtl::hip::device_buffer<float> buf(1024);
    EXPECT_EQ(buf.size(), 1024u);
    EXPECT_GE(buf.capacity(), 1024u);
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.size_bytes(), 1024 * sizeof(float));
}

TEST(HipDeviceBufferTest, MoveConstruct) {
    dtl::hip::device_buffer<int> a(256);
    auto* ptr = a.data();
    dtl::hip::device_buffer<int> b(std::move(a));

    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(b.size(), 256u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
}

TEST(HipDeviceBufferTest, MoveAssign) {
    dtl::hip::device_buffer<int> a(256);
    dtl::hip::device_buffer<int> b(128);
    auto* ptr = a.data();

    b = std::move(a);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(b.size(), 256u);
    EXPECT_EQ(a.data(), nullptr);
}

TEST(HipDeviceBufferTest, Resize) {
    dtl::hip::device_buffer<float> buf(64);
    EXPECT_EQ(buf.size(), 64u);

    buf.resize(128);
    EXPECT_EQ(buf.size(), 128u);
    EXPECT_GE(buf.capacity(), 128u);

    // Shrink within capacity
    buf.resize(32);
    EXPECT_EQ(buf.size(), 32u);
    EXPECT_GE(buf.capacity(), 128u);  // Capacity unchanged
}

TEST(HipDeviceBufferTest, Reserve) {
    dtl::hip::device_buffer<double> buf(64);
    buf.reserve(512);
    EXPECT_GE(buf.capacity(), 512u);
    EXPECT_EQ(buf.size(), 64u);  // Size unchanged
}

TEST(HipDeviceBufferTest, ClearAndRelease) {
    dtl::hip::device_buffer<int> buf(256);
    EXPECT_FALSE(buf.empty());

    buf.clear();
    EXPECT_TRUE(buf.empty());
    EXPECT_GE(buf.capacity(), 256u);  // Memory retained

    buf.resize(128);
    auto* ptr = buf.release();
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);

    // Caller owns the memory, must free
    hipFree(ptr);
}

TEST(HipDeviceBufferTest, Memset) {
    dtl::hip::device_buffer<int> buf(256);
    buf.memset(0);

    std::vector<int> host(256, -1);
    hipMemcpy(host.data(), buf.data(), 256 * sizeof(int), hipMemcpyDeviceToHost);

    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(host[i], 0) << "index=" << i;
    }
}

#else  // !DTL_ENABLE_HIP

TEST(HipDeviceBufferTest, DefaultConstructPlaceholder) {
    SUCCEED();
}

TEST(HipDeviceBufferTest, SizedConstructPlaceholder) {
    SUCCEED();
}

TEST(HipDeviceBufferTest, MoveConstructPlaceholder) {
    SUCCEED();
}

TEST(HipDeviceBufferTest, MoveAssignPlaceholder) {
    SUCCEED();
}

TEST(HipDeviceBufferTest, ResizePlaceholder) {
    SUCCEED();
}

TEST(HipDeviceBufferTest, ReservePlaceholder) {
    SUCCEED();
}

TEST(HipDeviceBufferTest, ClearAndReleasePlaceholder) {
    SUCCEED();
}

TEST(HipDeviceBufferTest, MemsetPlaceholder) {
    SUCCEED();
}

#endif  // DTL_ENABLE_HIP

}  // namespace dtl::test
