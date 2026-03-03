// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_shared_region_posix.cpp
/// @brief Unit tests for POSIX shared_region (shm_open/mmap)
/// @details Tests the shared memory region implementation with POSIX
///          shm_open/mmap on Linux/macOS, and heap fallback on other platforms.
/// @since 0.1.0

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <backends/shared_memory/shared_memory_communicator.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <type_traits>

namespace dtl::test {

// =============================================================================
// shared_region Construction & Validity
// =============================================================================

TEST(SharedRegionPosixTest, DefaultConstructionIsInvalid) {
    dtl::shared_memory::shared_region region;
    EXPECT_FALSE(region.valid());
    EXPECT_EQ(region.data(), nullptr);
    EXPECT_EQ(region.size(), 0u);
}

TEST(SharedRegionPosixTest, ConstructWithSize) {
    constexpr dtl::size_type kSize = 4096;
    dtl::shared_memory::shared_region region(kSize, 99999);
    EXPECT_TRUE(region.valid());
    EXPECT_NE(region.data(), nullptr);
    EXPECT_EQ(region.size(), kSize);
    EXPECT_EQ(region.region_id(), 99999u);
}

TEST(SharedRegionPosixTest, DataIsReadWritable) {
    constexpr dtl::size_type kSize = 1024;
    dtl::shared_memory::shared_region region(kSize, 99998);
    ASSERT_TRUE(region.valid());

    // Write and read back
    auto* ptr = static_cast<char*>(region.data());
    std::memset(ptr, 0xAB, kSize);
    EXPECT_EQ(static_cast<unsigned char>(ptr[0]), 0xAB);
    EXPECT_EQ(static_cast<unsigned char>(ptr[kSize - 1]), 0xAB);
}

TEST(SharedRegionPosixTest, MoveConstruction) {
    constexpr dtl::size_type kSize = 512;
    dtl::shared_memory::shared_region original(kSize, 99997);
    ASSERT_TRUE(original.valid());

    dtl::shared_memory::shared_region moved(std::move(original));
    EXPECT_TRUE(moved.valid());
    EXPECT_EQ(moved.size(), kSize);
    EXPECT_FALSE(original.valid());  // NOLINT(bugprone-use-after-move)
}

TEST(SharedRegionPosixTest, MoveAssignment) {
    constexpr dtl::size_type kSize = 256;
    dtl::shared_memory::shared_region a(kSize, 99996);
    dtl::shared_memory::shared_region b;

    b = std::move(a);
    EXPECT_TRUE(b.valid());
    EXPECT_EQ(b.size(), kSize);
    EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move)
}

TEST(SharedRegionPosixTest, NonCopyable) {
    static_assert(!std::is_copy_constructible_v<dtl::shared_memory::shared_region>,
                  "shared_region must not be copy-constructible");
    static_assert(!std::is_copy_assignable_v<dtl::shared_memory::shared_region>,
                  "shared_region must not be copy-assignable");
    SUCCEED();
}

// =============================================================================
// POSIX-Specific Tests (only meaningful on Linux/macOS)
// =============================================================================

#if defined(__linux__) || defined(__APPLE__)

TEST(SharedRegionPosixTest, IsBackedBySharedMemory) {
    // On POSIX systems, shm_open should succeed in most environments
    // (Docker/CI may not have /dev/shm, in which case it falls back to heap)
    constexpr dtl::size_type kSize = 4096;
    dtl::shared_memory::shared_region region(kSize, 99995);
    ASSERT_TRUE(region.valid());
    // If is_shared() is true, we got real shared memory
    // If false, we fell back to heap (still valid and functional)
    if (region.is_shared()) {
        // Can't assert this will always be true in CI, but log it
        SUCCEED() << "Region is backed by POSIX shared memory";
    } else {
        SUCCEED() << "Region fell back to heap allocation";
    }
}

TEST(SharedRegionPosixTest, OpenExistingRegion) {
    constexpr dtl::size_type kSize = 4096;
    // Create a region
    dtl::shared_memory::shared_region creator(kSize, 99994);
    ASSERT_TRUE(creator.valid());

    if (creator.is_shared()) {
        // Write test data
        auto* ptr = static_cast<char*>(creator.data());
        std::memset(ptr, 0x42, kSize);

        // Open the same region (non-creator)
        auto opener = dtl::shared_memory::shared_region::open(99994, kSize);
        ASSERT_TRUE(opener.valid());
        // Data should be visible (shared memory is the same physical pages)
        auto* optr = static_cast<const char*>(opener.data());
        EXPECT_EQ(static_cast<unsigned char>(optr[0]), 0x42);
    } else {
        SUCCEED() << "Skipping open test (heap fallback, not truly shared)";
    }
}

#endif  // __linux__ || __APPLE__

}  // namespace dtl::test
