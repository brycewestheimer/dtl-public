// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_backend_discovery.cpp
/// @brief Tests for runtime backend discovery service
/// @since 0.1.0

#include <dtl/runtime/backend_discovery.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace dtl::runtime::testing {

// =============================================================================
// Capability Bitmask Tests
// =============================================================================

TEST(BackendCapability, NoneIsZero) {
    EXPECT_EQ(static_cast<uint32_t>(backend_capability::none), 0u);
}

TEST(BackendCapability, BitwiseOrCombines) {
    auto combined = backend_capability::point_to_point | backend_capability::collectives;
    EXPECT_TRUE(has_capability(combined, backend_capability::point_to_point));
    EXPECT_TRUE(has_capability(combined, backend_capability::collectives));
    EXPECT_FALSE(has_capability(combined, backend_capability::rma));
}

TEST(BackendCapability, BitwiseAndIntersects) {
    auto a = backend_capability::point_to_point | backend_capability::collectives;
    auto b = backend_capability::collectives | backend_capability::rma;
    auto intersection = a & b;
    EXPECT_TRUE(has_capability(intersection, backend_capability::collectives));
    EXPECT_FALSE(has_capability(intersection, backend_capability::point_to_point));
    EXPECT_FALSE(has_capability(intersection, backend_capability::rma));
}

TEST(BackendCapability, HasCapabilityReturnsFalseForNone) {
    EXPECT_FALSE(has_capability(backend_capability::none, backend_capability::point_to_point));
}

TEST(BackendCapability, AllFlagsAreDistinct) {
    // Each flag should be a unique power of two
    auto all = backend_capability::point_to_point
             | backend_capability::collectives
             | backend_capability::rma
             | backend_capability::gpu_aware
             | backend_capability::async_operations
             | backend_capability::thread_multiple
             | backend_capability::rdma
             | backend_capability::device_execution
             | backend_capability::memory_management;

    // All 9 flags set means 9 bits should be set
    uint32_t val = static_cast<uint32_t>(all);
    uint32_t bit_count = 0;
    while (val) {
        bit_count += val & 1u;
        val >>= 1;
    }
    EXPECT_EQ(bit_count, 9u);
}

// =============================================================================
// Backend Descriptor Tests
// =============================================================================

TEST(BackendDescriptor, DefaultConstruction) {
    backend_descriptor desc;
    EXPECT_TRUE(desc.name.empty());
    EXPECT_TRUE(desc.version.empty());
    EXPECT_EQ(desc.capabilities, backend_capability{});
    EXPECT_FALSE(desc.available);
    EXPECT_FALSE(desc.compiled);
    EXPECT_EQ(desc.maturity, backend_maturity::stub);
    EXPECT_EQ(desc.compiled_capabilities, backend_capability{});
    EXPECT_EQ(desc.runtime_capabilities, backend_capability{});
    EXPECT_EQ(desc.functional_capabilities, backend_capability{});
    EXPECT_TRUE(desc.capability_levels.empty());
}

// =============================================================================
// Available Backends Tests
// =============================================================================

TEST(BackendDiscovery, AvailableBackendsReturnsKnownBackends) {
    auto backends = available_backends();
    EXPECT_FALSE(backends.empty());

    // Should have descriptors for all 8 known backends
    EXPECT_EQ(backends.size(), 8u);

    // Check that MPI is listed
    auto mpi_it = std::find_if(backends.begin(), backends.end(),
        [](const backend_descriptor& d) { return d.name == "MPI"; });
    EXPECT_NE(mpi_it, backends.end());

    // Check that CUDA is listed
    auto cuda_it = std::find_if(backends.begin(), backends.end(),
        [](const backend_descriptor& d) { return d.name == "CUDA"; });
    EXPECT_NE(cuda_it, backends.end());
}

TEST(BackendDiscovery, MpiBackendHasExpectedCapabilities) {
    auto desc = query_backend("mpi");
    EXPECT_EQ(desc.name, "MPI");
    EXPECT_EQ(desc.maturity, backend_maturity::production);
#if DTL_ENABLE_MPI
    EXPECT_TRUE(has_capability(desc.compiled_capabilities, backend_capability::point_to_point));
    EXPECT_TRUE(has_capability(desc.compiled_capabilities, backend_capability::collectives));
    EXPECT_TRUE(has_capability(desc.compiled_capabilities, backend_capability::rma));
    EXPECT_TRUE(has_capability(desc.compiled_capabilities, backend_capability::async_operations));
#else
    EXPECT_FALSE(has_capability(desc.compiled_capabilities, backend_capability::point_to_point));
    EXPECT_FALSE(has_capability(desc.compiled_capabilities, backend_capability::collectives));
    EXPECT_FALSE(has_capability(desc.compiled_capabilities, backend_capability::rma));
    EXPECT_FALSE(has_capability(desc.compiled_capabilities, backend_capability::async_operations));
#endif
}

TEST(BackendDiscovery, CudaBackendHasGpuCapabilities) {
    auto desc = query_backend("cuda");
    EXPECT_EQ(desc.name, "CUDA");
#if DTL_ENABLE_CUDA
    EXPECT_EQ(desc.maturity, backend_maturity::partial);
    EXPECT_TRUE(has_capability(desc.compiled_capabilities, backend_capability::gpu_aware));
    EXPECT_TRUE(has_capability(desc.compiled_capabilities, backend_capability::device_execution));
    EXPECT_TRUE(has_capability(desc.compiled_capabilities, backend_capability::memory_management));
#else
    EXPECT_FALSE(has_capability(desc.compiled_capabilities, backend_capability::gpu_aware));
    EXPECT_FALSE(has_capability(desc.compiled_capabilities, backend_capability::device_execution));
    EXPECT_FALSE(has_capability(desc.compiled_capabilities, backend_capability::memory_management));
#endif
}

TEST(BackendDiscovery, StubBackendsDoNotReportFunctionalCapabilities) {
    auto ucx = query_backend("ucx");
    auto gasnet = query_backend("gasnet");
    auto sycl = query_backend("sycl");

    EXPECT_EQ(ucx.maturity, backend_maturity::stub);
    EXPECT_EQ(gasnet.maturity, backend_maturity::stub);
    EXPECT_EQ(sycl.maturity, backend_maturity::stub);

    EXPECT_EQ(ucx.functional_capabilities, backend_capability::none);
    EXPECT_EQ(gasnet.functional_capabilities, backend_capability::none);
    EXPECT_EQ(sycl.functional_capabilities, backend_capability::none);
}

TEST(BackendDiscovery, CapabilityLevelsAreReportedPerFeature) {
    auto desc = query_backend("mpi");
    EXPECT_FALSE(desc.capability_levels.empty());

    auto p2p = std::find_if(
        desc.capability_levels.begin(), desc.capability_levels.end(),
        [](const capability_descriptor& c) {
            return c.capability == backend_capability::point_to_point;
        });

    ASSERT_NE(p2p, desc.capability_levels.end());
#if DTL_ENABLE_MPI
    EXPECT_TRUE(p2p->level == capability_level::compiled
             || p2p->level == capability_level::runtime_available
             || p2p->level == capability_level::functional);
#else
    EXPECT_EQ(p2p->level, capability_level::unavailable);
#endif
}

// =============================================================================
// Query Backend Tests
// =============================================================================

TEST(BackendDiscovery, QueryUnknownBackend) {
    auto desc = query_backend("unknown");
    EXPECT_EQ(desc.name, "unknown");
    EXPECT_FALSE(desc.available);
    EXPECT_FALSE(desc.compiled);
    EXPECT_EQ(desc.capabilities, backend_capability{});
}

TEST(BackendDiscovery, QueryIsCaseInsensitive) {
    auto lower = query_backend("mpi");
    auto upper = query_backend("MPI");
    auto mixed = query_backend("Mpi");
    EXPECT_EQ(lower.name, upper.name);
    EXPECT_EQ(lower.name, mixed.name);
    EXPECT_EQ(lower.capabilities, upper.capabilities);
}

// =============================================================================
// Convenience Query Tests
// =============================================================================

TEST(BackendDiscovery, HasAnyGpuBackendReturnsReasonableValue) {
    // This test just verifies the function doesn't crash.
    // The actual return value depends on the build configuration.
    [[maybe_unused]] bool result = has_any_gpu_backend();
}

TEST(BackendDiscovery, HasAnyCommBackendReturnsReasonableValue) {
    // Same — just verify it doesn't crash.
    [[maybe_unused]] bool result = has_any_comm_backend();
}

}  // namespace dtl::runtime::testing
