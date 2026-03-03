// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_header_smoke.cpp
/// @brief Verifies that the master DTL header compiles cleanly
/// @details This is a compile-time-only test. If this file compiles,
///          the DTL library headers are self-consistent.
///
/// This test exists to catch:
/// - Duplicate trait/concept definitions
/// - Forward declaration mismatches
/// - Missing macro definitions
/// - Include order issues
/// - ODR violations
///
/// @since 0.1.0

#include <dtl/dtl.hpp>

int main() {
    // Instantiate a few core types to verify templates are well-formed
    // These are compile-time checks - if they compile, the test passes

    // Core types
    [[maybe_unused]] dtl::index_t idx = 0;
    [[maybe_unused]] dtl::rank_t rank = 0;
    [[maybe_unused]] dtl::size_type sz = 0;

    // Status/result types
    [[maybe_unused]] dtl::status s{dtl::status_code::ok};

    // If we get here, the library headers compiled successfully
    return 0;
}
