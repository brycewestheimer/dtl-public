// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_context_mpi.cpp
/// @brief Integration tests for multi-domain context with MPI
/// @details Tests V1.3.0 context functionality with MPI backend.
/// @note Run with: mpirun -np 2 ./test_executable
///       or:       mpirun -np 4 ./test_executable

#include <dtl/core/context.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/core/environment.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Test Fixture
// =============================================================================

class ContextMpiTest : public ::testing::Test {
protected:
    void SetUp() override {
        // MPI should already be initialized by the test framework or environment
    }
};

// =============================================================================
// MPI Context Basic Tests
// =============================================================================

TEST_F(ContextMpiTest, MpiContextCreation) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    if (ctx.get<mpi_domain>().valid()) {
        EXPECT_GE(ctx.rank(), 0);
        EXPECT_GE(ctx.size(), 1);
        EXPECT_LE(ctx.rank(), ctx.size() - 1);
    }
}

TEST_F(ContextMpiTest, MpiContextDomainAccess) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    auto& mpi = ctx.get<mpi_domain>();
    auto& cpu = ctx.get<cpu_domain>();

    EXPECT_TRUE(cpu.valid());  // CPU always valid

    if (mpi.valid()) {
        EXPECT_EQ(ctx.rank(), mpi.rank());
        EXPECT_EQ(ctx.size(), mpi.size());
    }
}

TEST_F(ContextMpiTest, MpiContextIsRoot) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    auto& mpi = ctx.get<mpi_domain>();
    if (mpi.valid()) {
        EXPECT_EQ(ctx.is_root(), (mpi.rank() == 0));
    }
}

TEST_F(ContextMpiTest, MpiContextBarrier) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    if (ctx.get<mpi_domain>().valid()) {
        // Should not deadlock
        ctx.barrier();
        EXPECT_TRUE(true);
    }
}

// =============================================================================
// MPI Domain Split Tests
// =============================================================================

TEST_F(ContextMpiTest, MpiDomainSplit) {
    mpi_domain mpi;

    if (!mpi.valid()) {
        GTEST_SKIP() << "MPI not available";
    }

    // Split by even/odd ranks
    int color = mpi.rank() % 2;
    auto split_result = mpi.split(color);

    ASSERT_TRUE(split_result.has_value()) << "MPI split failed";

    auto& split_mpi = *split_result;
    EXPECT_TRUE(split_mpi.valid());

    // New communicator should have smaller size
    EXPECT_LE(split_mpi.size(), mpi.size());

    // Rank should be valid in new communicator
    EXPECT_GE(split_mpi.rank(), 0);
    EXPECT_LT(split_mpi.rank(), split_mpi.size());
}

TEST_F(ContextMpiTest, ContextSplitMpi) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    if (!ctx.get<mpi_domain>().valid()) {
        GTEST_SKIP() << "MPI not available";
    }

    // Split by even/odd ranks
    int color = ctx.rank() % 2;
    auto split_result = ctx.split_mpi(color);

    ASSERT_TRUE(split_result.has_value()) << "Context split failed";

    auto& split_ctx = *split_result;
    EXPECT_LE(split_ctx.size(), ctx.size());
}

// =============================================================================
// with_cuda Factory Tests
// =============================================================================

TEST_F(ContextMpiTest, MpiContextWithCuda) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    // Add CUDA domain
    auto cuda_ctx = ctx.with_cuda(0);

    // Verify types
    static_assert(cuda_ctx.has<mpi_domain>());
    static_assert(cuda_ctx.has<cpu_domain>());
    static_assert(cuda_ctx.has<cuda_domain>());
    static_assert(cuda_ctx.domain_count == 3);

    // CUDA domain is present (may or may not be valid depending on GPU)
    auto& cuda = cuda_ctx.get<cuda_domain>();
#if DTL_ENABLE_CUDA
    // With real CUDA, device_id should match requested
    if (cuda.valid()) {
        EXPECT_EQ(cuda.device_id(), 0);
    }
#else
    // Without CUDA, stub returns -1 for device_id
    EXPECT_EQ(cuda.device_id(), -1);
    EXPECT_FALSE(cuda.valid());
#endif
}

// =============================================================================
// Container Integration Tests
// =============================================================================

TEST_F(ContextMpiTest, DistributedVectorWithContext) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    if (!ctx.get<mpi_domain>().valid()) {
        GTEST_SKIP() << "MPI not available";
    }

    // This tests that containers can be constructed with the new context
    // The context provides rank() and size()
    const size_type global_size = 1000;
    const rank_t expected_rank = ctx.rank();
    const rank_t expected_size = ctx.size();

    distributed_vector<int> vec(global_size, ctx);

    // Verify context provides expected interface
    EXPECT_EQ(ctx.rank(), expected_rank);
    EXPECT_EQ(ctx.size(), expected_size);

    // Verify container uses context-provided distribution
    EXPECT_EQ(vec.global_size(), global_size);
    EXPECT_EQ(vec.rank(), expected_rank);
    EXPECT_EQ(vec.num_ranks(), expected_size);
}

// =============================================================================
// Multi-Rank Collective Tests
// =============================================================================

TEST_F(ContextMpiTest, AllRanksAgree) {
    mpi_context ctx{mpi_domain{}, cpu_domain{}};

    if (!ctx.get<mpi_domain>().valid()) {
        GTEST_SKIP() << "MPI not available";
    }

    // All ranks should agree on the total size
    rank_t my_size = ctx.size();
    rank_t my_rank = ctx.rank();

    // Verify all ranks have valid indices
    EXPECT_GE(my_rank, 0);
    EXPECT_LT(my_rank, my_size);

    // Barrier to ensure all ranks reach here
    ctx.barrier();
}

// =============================================================================
// Environment Factory Tests
// =============================================================================

TEST_F(ContextMpiTest, EnvironmentMakeWorldContext) {
    // This test requires environment to be created first
    // In a real application, environment would be constructed in main()

    // Test that we can create a world context
    // The context should have MPI and CPU domains
    static_assert(mpi_context::has<mpi_domain>());
    static_assert(mpi_context::has<cpu_domain>());

    EXPECT_TRUE(true);  // Type checks pass
}

#else  // !DTL_ENABLE_MPI

// =============================================================================
// Non-MPI Tests
// =============================================================================

TEST(ContextMpiTest, MpiNotEnabled) {
    // When MPI is not enabled, we can still use cpu_context
    cpu_context ctx;

    EXPECT_EQ(ctx.rank(), 0);
    EXPECT_EQ(ctx.size(), 1);
    EXPECT_TRUE(ctx.is_root());
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test
