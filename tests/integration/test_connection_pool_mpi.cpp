// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_connection_pool_mpi.cpp
/// @brief MPI integration tests for runtime communicator pooling

#include <dtl/runtime/connection_pool.hpp>

#include <gtest/gtest.h>

#if DTL_ENABLE_MPI
#  include <mpi.h>
#endif

namespace dtl::test {

#if DTL_ENABLE_MPI

TEST(MpiConnectionPool, AcquireReleaseTracksMetrics) {
    auto pool_result = dtl::runtime::make_communicator_pool("mpi");
    ASSERT_FALSE(pool_result.has_error()) << pool_result.error().message();

    auto pool = std::move(pool_result.value());
    auto initial_metrics = pool->metrics();
    EXPECT_EQ(initial_metrics.current_active, 0u);

    auto handle_result = pool->acquire();
    ASSERT_FALSE(handle_result.has_error()) << handle_result.error().message();
    auto handle = std::move(handle_result.value());
    ASSERT_TRUE(handle.valid());

    auto active_metrics = pool->metrics();
    EXPECT_EQ(active_metrics.total_acquired, 1u);
    EXPECT_EQ(active_metrics.current_active, 1u);

    auto* comm = handle.get<MPI_Comm>();
    ASSERT_NE(comm, nullptr);
    int rank = -1;
    ASSERT_EQ(MPI_Comm_rank(*comm, &rank), MPI_SUCCESS);
    EXPECT_GE(rank, 0);

    handle.release();

    auto released_metrics = pool->metrics();
    EXPECT_EQ(released_metrics.total_released, 1u);
    EXPECT_EQ(released_metrics.current_active, 0u);
}

TEST(MpiConnectionPool, OutstandingHandleReleaseIsSafeAfterPoolDestruction) {
    auto pool_result = dtl::runtime::make_communicator_pool("mpi");
    ASSERT_FALSE(pool_result.has_error()) << pool_result.error().message();

    auto pool = std::move(pool_result.value());
    auto handle_result = pool->acquire();
    ASSERT_FALSE(handle_result.has_error()) << handle_result.error().message();
    auto handle = std::move(handle_result.value());
    ASSERT_TRUE(handle.valid());

    pool.reset();

    EXPECT_NO_THROW(handle.release());
    EXPECT_FALSE(handle.valid());
}

#endif

}  // namespace dtl::test
