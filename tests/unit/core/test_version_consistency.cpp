// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_version_consistency.cpp
/// @brief Verify DTL_VERSION_STRING matches MAJOR.MINOR.PATCH

#include <dtl/core/version.hpp>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

TEST(VersionConsistency, VersionStringMatchesMajorMinorPatch) {
    std::ostringstream expected;
    expected << DTL_VERSION_MAJOR << "." << DTL_VERSION_MINOR << "." << DTL_VERSION_PATCH;
    EXPECT_EQ(std::string(DTL_VERSION_STRING), expected.str());
}

TEST(VersionConsistency, PrereleaseDefined) {
    // Verify the prerelease macro is set for alpha
    std::string prerelease = DTL_VERSION_PRERELEASE;
    EXPECT_EQ(prerelease, "alpha.1");
    std::string full = DTL_VERSION_FULL;
    EXPECT_EQ(full, "0.1.0-alpha.1");
}
