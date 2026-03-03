// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_thread_level.cpp
/// @brief Tests for MPI thread level verification
/// @details Updated for V1.4.0 instance-based API.

#include <dtl/core/environment.hpp>
#include <dtl/runtime/runtime_registry.hpp>
#include <gtest/gtest.h>
#include <cstring>

namespace dtl::test {

TEST(ThreadLevelTest, ThreadLevelAccessor) {
    // Query registry directly to avoid needing an environment instance
    // just for this pre-init check
    [[maybe_unused]] int level = runtime::runtime_registry::instance().mpi_thread_level();
    // Level is -1 when MPI not initialized, or 0-3 when it is
    EXPECT_GE(level, -1);
    EXPECT_LE(level, 3);
}

TEST(ThreadLevelTest, ThreadLevelName) {
    const char* name = runtime::runtime_registry::instance().mpi_thread_level_name();
    EXPECT_NE(name, nullptr);
    // Should be a non-empty string
    EXPECT_GT(std::strlen(name), 0u);
}

TEST(ThreadLevelTest, ThreadLevelNameConsistentWithValue) {
    int level = runtime::runtime_registry::instance().mpi_thread_level();
    const char* name = runtime::runtime_registry::instance().mpi_thread_level_name();

    if (level == -1) {
        // MPI not available
        EXPECT_TRUE(
            std::strcmp(name, "unknown") == 0 ||
            std::strcmp(name, "MPI_NOT_ENABLED") == 0
        );
    }
}

}  // namespace dtl::test
