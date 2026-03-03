// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_connection_pool.cpp
/// @brief Tests for runtime connection pool
/// @since 0.1.0

#include <dtl/runtime/connection_pool.hpp>
#include <dtl/error/status.hpp>

#include <gtest/gtest.h>

namespace dtl::runtime::testing {

// =============================================================================
// Pool Metrics Tests
// =============================================================================

TEST(PoolMetrics, DefaultInitializesToZero) {
    pool_metrics m;
    EXPECT_EQ(m.total_acquired, 0u);
    EXPECT_EQ(m.total_released, 0u);
    EXPECT_EQ(m.current_active, 0u);
    EXPECT_EQ(m.high_water_mark, 0u);
    EXPECT_EQ(m.pool_size, 0u);
    EXPECT_EQ(m.pool_capacity, 0u);
}

// =============================================================================
// Pool Handle RAII Tests
// =============================================================================

TEST(PoolHandle, DefaultIsInvalid) {
    pool_handle h;
    EXPECT_FALSE(h.valid());
}

TEST(PoolHandle, ValidWithResource) {
    int resource = 42;
    pool_handle h(&resource, [](void*) {});
    EXPECT_TRUE(h.valid());
    EXPECT_EQ(*h.get<int>(), 42);
}

TEST(PoolHandle, ReleaseCallbackFiresOnDestruction) {
    int resource = 0;
    bool released = false;

    {
        pool_handle h(&resource, [&released](void*) { released = true; });
        EXPECT_TRUE(h.valid());
        EXPECT_FALSE(released);
    }
    // Destructor should have fired the callback
    EXPECT_TRUE(released);
}

TEST(PoolHandle, ExplicitRelease) {
    int resource = 0;
    bool released = false;

    pool_handle h(&resource, [&released](void*) { released = true; });
    h.release();
    EXPECT_TRUE(released);
    EXPECT_FALSE(h.valid());
}

TEST(PoolHandle, DoubleReleaseIsSafe) {
    int resource = 0;
    int release_count = 0;

    pool_handle h(&resource, [&release_count](void*) { release_count++; });
    h.release();
    h.release();  // Should be a no-op
    EXPECT_EQ(release_count, 1);
}

TEST(PoolHandle, MoveConstructor) {
    int resource = 0;
    bool released = false;

    pool_handle h1(&resource, [&released](void*) { released = true; });

    pool_handle h2(std::move(h1));
    EXPECT_FALSE(h1.valid());  // NOLINT: testing moved-from state
    EXPECT_TRUE(h2.valid());
    EXPECT_FALSE(released);

    h2.release();
    EXPECT_TRUE(released);
}

TEST(PoolHandle, MoveAssignment) {
    int r1 = 1, r2 = 2;
    bool released1 = false, released2 = false;

    pool_handle h1(&r1, [&released1](void*) { released1 = true; });
    pool_handle h2(&r2, [&released2](void*) { released2 = true; });

    // Moving h1 into h2 should release h2's old resource
    h2 = std::move(h1);
    EXPECT_TRUE(released2);   // Old h2 resource released
    EXPECT_FALSE(released1);  // h1's resource transferred to h2

    h2.release();
    EXPECT_TRUE(released1);
}

// =============================================================================
// Factory Tests
// =============================================================================

TEST(ConnectionPool, NcclPoolReturnsNotSupported) {
    auto result = make_communicator_pool("nccl");
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
}

TEST(ConnectionPool, UnknownBackendReturnsNotSupported) {
    auto result = make_communicator_pool("unknown_backend");
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::not_supported);
}

#if DTL_ENABLE_MPI
TEST(ConnectionPool, MpiPoolRequiresInitializedMPI) {
    auto result = make_communicator_pool("mpi");
    EXPECT_TRUE(result.has_error());
    EXPECT_EQ(result.error().code(), dtl::status_code::invalid_state);
}
#endif

// =============================================================================
// MPI Pool Tests (only run when MPI is available at build time)
// =============================================================================
// Note: These tests do NOT call MPI_Init — they cannot actually use
// MPI_Comm_dup in a non-MPI process. The MPI pool is tested in
// MPI-enabled integration tests. Here we test the factory returns
// a pool object when MPI is compiled in.

// The MPI pool integration tests are in tests/mpi/ and require
// mpirun to execute. Unit tests here verify the non-MPI factory path.

}  // namespace dtl::runtime::testing
