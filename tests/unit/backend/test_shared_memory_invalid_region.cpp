// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_shared_memory_invalid_region.cpp
/// @brief Verify shared_memory handles null/invalid regions gracefully
/// @since 0.1.0-alpha

#include <backends/shared_memory/shared_memory_communicator.hpp>
#include <gtest/gtest.h>

namespace dtl::test {

TEST(SharedMemoryInvalidRegion, DefaultConstructedCommSafe) {
    // Default-constructed communicator should be safe to query
    dtl::shared_memory::shared_memory_communicator comm;
    EXPECT_EQ(comm.rank(), dtl::no_rank);
    EXPECT_EQ(comm.size(), 0);
}

TEST(SharedMemoryInvalidRegion, DefaultConstructedSharedBufferNull) {
    // Default-constructed communicator has no shared buffer
    dtl::shared_memory::shared_memory_communicator comm;
    EXPECT_EQ(comm.shared_buffer(), nullptr);
    EXPECT_EQ(comm.shared_buffer_size(), 0u);
}

TEST(SharedMemoryInvalidRegion, InitializedCommHasValidBuffer) {
    // Properly initialized communicator should have a valid shared buffer
    dtl::shared_memory::shared_memory_communicator comm(0, 1);
    EXPECT_TRUE(comm.valid());
    EXPECT_NE(comm.shared_buffer(), nullptr);
    EXPECT_GT(comm.shared_buffer_size(), 0u);
}

TEST(SharedMemoryInvalidRegion, SharedRegionDefaultInvalid) {
    // Default-constructed shared region is invalid
    dtl::shared_memory::shared_region region;
    EXPECT_FALSE(region.valid());
    EXPECT_EQ(region.data(), nullptr);
    EXPECT_EQ(region.size(), 0u);
}

TEST(SharedMemoryInvalidRegion, SharedRegionValidAfterConstruction) {
    // Region constructed with a valid size should be valid
    dtl::shared_memory::shared_region region(1024, 99999);
    EXPECT_TRUE(region.valid());
    EXPECT_NE(region.data(), nullptr);
    EXPECT_EQ(region.size(), 1024u);
}

}  // namespace dtl::test
