// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_resource.cpp
/// @brief Unit tests for dtl/utility/resource.hpp
/// @details Tests DTL initialization, capabilities, and RAII environment management.

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <dtl/utility/resource.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// Legacy resource API was moved to dtl::legacy:: in Phase 12.5.
// These tests exercise the legacy API, so bring it into scope.
using namespace dtl::legacy;

// Note: These tests run in isolation. Each test case handles its own
// initialization/finalization cycle because DTL uses global state.

// =============================================================================
// Init Options Tests
// =============================================================================

TEST(InitOptionsTest, DefaultValues) {
    init_options opts;

    EXPECT_EQ(opts.threading, init_options::thread_level::multiple);
    EXPECT_TRUE(opts.enable_mpi);
    EXPECT_TRUE(opts.enable_cuda);
    EXPECT_TRUE(opts.enable_shared_memory);
    EXPECT_FALSE(opts.verbose);
}

TEST(InitOptionsTest, ThreadLevelEnum) {
    // Verify thread levels are distinct
    EXPECT_NE(init_options::thread_level::single, init_options::thread_level::funneled);
    EXPECT_NE(init_options::thread_level::funneled, init_options::thread_level::serialized);
    EXPECT_NE(init_options::thread_level::serialized, init_options::thread_level::multiple);
}

// =============================================================================
// Capabilities Tests
// =============================================================================

TEST(CapabilitiesTest, DefaultValues) {
    capabilities caps;

    EXPECT_FALSE(caps.has_mpi);
    EXPECT_EQ(caps.num_ranks, 1);
    EXPECT_EQ(caps.my_rank, 0);
    EXPECT_FALSE(caps.has_cuda);
    EXPECT_EQ(caps.num_cuda_devices, 0);
    EXPECT_TRUE(caps.has_shared_memory);
    EXPECT_FALSE(caps.has_nccl);
}

TEST(CapabilitiesTest, IsDistributed) {
    capabilities caps;
    caps.num_ranks = 1;
    EXPECT_FALSE(caps.is_distributed());

    caps.num_ranks = 4;
    EXPECT_TRUE(caps.is_distributed());
}

TEST(CapabilitiesTest, HasGpu) {
    capabilities caps;
    caps.has_cuda = false;
    caps.num_cuda_devices = 0;
    EXPECT_FALSE(caps.has_gpu());

    caps.has_cuda = true;
    caps.num_cuda_devices = 0;
    EXPECT_FALSE(caps.has_gpu());

    caps.has_cuda = true;
    caps.num_cuda_devices = 2;
    EXPECT_TRUE(caps.has_gpu());
}

// =============================================================================
// Initialization Tests
// =============================================================================

class ResourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure clean state before each test
        if (legacy::is_initialized()) {
            legacy::finalize();
        }
    }

    void TearDown() override {
        // Clean up after each test
        if (legacy::is_initialized()) {
            legacy::finalize();
        }
    }
};

TEST_F(ResourceTest, InitWithOptions) {
    EXPECT_FALSE(legacy::is_initialized());

    init_options opts;
    opts.enable_mpi = false;  // Single process mode
    opts.enable_cuda = false;

    auto result = legacy::init(opts);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(legacy::is_initialized());

    // Check capabilities
    const auto& c = legacy::caps();
    EXPECT_EQ(c.my_rank, 0);
    EXPECT_EQ(c.num_ranks, 1);
    EXPECT_TRUE(c.has_shared_memory);
}

TEST_F(ResourceTest, InitFinalizeCycle) {
    EXPECT_FALSE(legacy::is_initialized());

    auto init_result = legacy::init();
    EXPECT_TRUE(init_result.has_value());
    EXPECT_TRUE(legacy::is_initialized());

    auto finalize_result = legacy::finalize();
    EXPECT_TRUE(finalize_result.has_value());
    EXPECT_FALSE(legacy::is_initialized());
}

TEST_F(ResourceTest, DoubleInitFails) {
    auto result1 = legacy::init();
    EXPECT_TRUE(result1.has_value());

    // Second init should fail
    auto result2 = legacy::init();
    EXPECT_TRUE(result2.has_error());
    EXPECT_EQ(result2.error().code(), status_code::invalid_state);
}

TEST_F(ResourceTest, FinalizeWithoutInitFails) {
    EXPECT_FALSE(legacy::is_initialized());

    auto result = legacy::finalize();
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), status_code::invalid_state);
}

// =============================================================================
// Rank and Size Functions Tests
// =============================================================================

TEST_F(ResourceTest, RankAndSizeFunctions) {
    legacy::init();

    EXPECT_EQ(legacy::rank(), 0);
    EXPECT_EQ(legacy::size(), 1);
}

// =============================================================================
// Finalize Callback Tests
// =============================================================================

TEST_F(ResourceTest, FinalizeCallbackInvoked) {
    legacy::init();

    bool callback_called = false;
    legacy::at_finalize([&]() {
        callback_called = true;
    });

    EXPECT_FALSE(callback_called);

    legacy::finalize();

    EXPECT_TRUE(callback_called);
}

TEST_F(ResourceTest, FinalizeCallbacksInReverseOrder) {
    legacy::init();

    std::vector<int> order;

    legacy::at_finalize([&]() { order.push_back(1); });
    legacy::at_finalize([&]() { order.push_back(2); });
    legacy::at_finalize([&]() { order.push_back(3); });

    legacy::finalize();

    // Should be in reverse order: 3, 2, 1
    ASSERT_EQ(order.size(), 3);
    EXPECT_EQ(order[0], 3);
    EXPECT_EQ(order[1], 2);
    EXPECT_EQ(order[2], 1);
}

TEST_F(ResourceTest, MultipleCallbacks) {
    legacy::init();

    int count = 0;
    legacy::at_finalize([&]() { count++; });
    legacy::at_finalize([&]() { count++; });
    legacy::at_finalize([&]() { count++; });

    legacy::finalize();

    EXPECT_EQ(count, 3);
}

// =============================================================================
// scoped_environment Tests
// =============================================================================

TEST(ScopedEnvironmentTest, RAIILifetime) {
    EXPECT_FALSE(legacy::is_initialized());

    {
        legacy::scoped_environment env;
        EXPECT_TRUE(legacy::is_initialized());
        EXPECT_EQ(env.rank(), 0);
        EXPECT_EQ(env.size(), 1);
    }

    EXPECT_FALSE(legacy::is_initialized());
}

TEST(ScopedEnvironmentTest, WithOptions) {
    EXPECT_FALSE(legacy::is_initialized());

    {
        init_options opts;
        opts.enable_mpi = false;
        opts.verbose = false;

        legacy::scoped_environment env(opts);
        EXPECT_TRUE(legacy::is_initialized());

        const auto& c = env.caps();
        EXPECT_TRUE(c.has_shared_memory);
    }

    EXPECT_FALSE(legacy::is_initialized());
}

TEST(ScopedEnvironmentTest, FinalizeCallbacksOnDestruction) {
    bool callback_called = false;

    {
        legacy::scoped_environment env;
        legacy::at_finalize([&]() {
            callback_called = true;
        });
        EXPECT_FALSE(callback_called);
    }

    EXPECT_TRUE(callback_called);
}

// =============================================================================
// Environment Handle Tests
// =============================================================================

TEST_F(ResourceTest, EnvironmentHandleNotInitialized) {
    EXPECT_FALSE(legacy::environment_handle::is_initialized());
}

TEST_F(ResourceTest, EnvironmentHandleAfterInit) {
    legacy::init();

    EXPECT_TRUE(legacy::environment_handle::is_initialized());
    EXPECT_EQ(legacy::environment_handle::rank(), 0);
    EXPECT_EQ(legacy::environment_handle::size(), 1);
}

}  // namespace dtl::test

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
