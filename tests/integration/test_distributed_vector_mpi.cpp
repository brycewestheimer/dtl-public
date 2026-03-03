// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_vector_mpi.cpp
/// @brief MPI integration tests for distributed containers
/// @details Phase 08, Task 09: Verify container correctness across MPI ranks.
///          Tests partition correctness, data distribution, and basic collectives.
/// @note Run with: mpirun -np 2 ./dtl_mpi_tests
///       or:       mpirun -np 4 ./dtl_mpi_tests

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/core/environment.hpp>

#include <gtest/gtest.h>

#include <numeric>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Partition Correctness Tests
// =============================================================================

TEST(ContainerMpiTest, VectorGlobalSizeConsistent) {
    auto ctx = make_mpi_context();
    distributed_vector<int> vec(1000, ctx);

    // Global size should be consistent across all ranks
    EXPECT_EQ(vec.global_size(), 1000u);
}

TEST(ContainerMpiTest, VectorLocalSizesPartition) {
    auto ctx = make_mpi_context();
    const size_type global_n = 1000;
    distributed_vector<int> vec(global_n, ctx);

    // Each rank should have a positive local size
    EXPECT_GT(vec.local_size(), 0u);

    // Local size should not exceed global size
    EXPECT_LE(vec.local_size(), global_n);
}

TEST(ContainerMpiTest, VectorLocalDataAccess) {
    auto ctx = make_mpi_context();
    distributed_vector<int> vec(100, 42, ctx);

    // All local elements should be initialized to 42
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 42);
    }
}

TEST(ContainerMpiTest, VectorLocalModification) {
    auto ctx = make_mpi_context();
    distributed_vector<int> vec(100, ctx);

    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), ctx.rank() * 1000);

    // Verify local modifications are visible
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<int>(ctx.rank() * 1000 + i));
    }
}

TEST(ContainerMpiTest, VectorSyncStateWithMpi) {
    auto ctx = make_mpi_context();
    distributed_vector<int> vec(100, ctx);

    EXPECT_TRUE(vec.is_clean());

    vec.local_view()[0] = 99;
    vec.mark_local_modified();
    EXPECT_TRUE(vec.is_dirty());

    vec.mark_clean();
    EXPECT_TRUE(vec.is_clean());
}

#else  // !DTL_ENABLE_MPI

TEST(ContainerMpiTest, SkipWithoutMpi) {
    GTEST_SKIP() << "MPI not enabled — container MPI integration tests skipped";
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test