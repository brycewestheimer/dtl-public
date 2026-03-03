// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_prefetch_callable.cpp
/// @brief Verify prefetch functions are callable (non-CUDA path)
/// @details On non-CUDA builds, prefetch_to_device and prefetch_to_host
///          should be no-ops that compile and run without error.

#include <dtl/policies/placement/unified_memory.hpp>
#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

// =============================================================================
// Prefetch to Device Tests
// =============================================================================

TEST(PrefetchCallable, PrefetchToDeviceNoOp) {
    std::vector<double> data(100, 1.0);
    // Should be a no-op on non-CUDA builds, valid CUDA prefetch on CUDA builds
    dtl::unified_memory::prefetch_to_device(
        data.data(), data.size() * sizeof(double));
    SUCCEED();
}

TEST(PrefetchCallable, PrefetchToDeviceWithDeviceId) {
    std::vector<float> data(64, 2.0f);
    dtl::unified_memory::prefetch_to_device(
        data.data(), data.size() * sizeof(float), 0);
    SUCCEED();
}

TEST(PrefetchCallable, PrefetchToDeviceZeroSize) {
    std::vector<int> data;
    // Zero-size prefetch should be harmless
    dtl::unified_memory::prefetch_to_device(data.data(), 0);
    SUCCEED();
}

// =============================================================================
// Prefetch to Host Tests
// =============================================================================

TEST(PrefetchCallable, PrefetchToHostNoOp) {
    std::vector<double> data(100, 1.0);
    // Should be a no-op on non-CUDA builds
    dtl::unified_memory::prefetch_to_host(
        data.data(), data.size() * sizeof(double));
    SUCCEED();
}

TEST(PrefetchCallable, PrefetchToHostZeroSize) {
    std::vector<int> data;
    dtl::unified_memory::prefetch_to_host(data.data(), 0);
    SUCCEED();
}

// =============================================================================
// Policy Property Tests
// =============================================================================

TEST(PrefetchCallable, SupportsPrefetchIsTrue) {
    EXPECT_TRUE(dtl::unified_memory::supports_prefetch());
}

TEST(PrefetchCallable, IsHostAndDeviceAccessible) {
    EXPECT_TRUE(dtl::unified_memory::is_host_accessible());
    EXPECT_TRUE(dtl::unified_memory::is_device_accessible());
}

TEST(PrefetchCallable, DoesNotRequireExplicitCopy) {
    EXPECT_FALSE(dtl::unified_memory::requires_explicit_copy());
}

}  // namespace dtl::test
