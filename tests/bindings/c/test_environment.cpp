// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_environment.cpp
 * @brief Unit tests for DTL C bindings environment operations
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl.h>

// ============================================================================
// Test Fixture
// ============================================================================

class EnvironmentTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Ensure no leaked environment handles after each test
        // (tests that create environments must destroy them)
    }
};

// ============================================================================
// Environment Lifecycle Tests
// ============================================================================

TEST_F(EnvironmentTest, EnvironmentCreateDestroy) {
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(env, nullptr);

    dtl_environment_destroy(env);
}

TEST_F(EnvironmentTest, EnvironmentCreateNullPointer) {
    dtl_status status = dtl_environment_create(nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);
}

TEST_F(EnvironmentTest, EnvironmentDestroyNull) {
    // Should not crash
    dtl_environment_destroy(nullptr);
}

// ============================================================================
// Environment State Query Tests
// ============================================================================

TEST_F(EnvironmentTest, EnvironmentIsInitialized) {
    // Before creation, should not be initialized
    // (Note: other tests may have left state, so we only check
    //  the positive case after our own creation)
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);

    EXPECT_NE(dtl_environment_is_initialized(), 0);

    dtl_environment_destroy(env);

    // After destroying the only handle, should not be initialized
    EXPECT_EQ(dtl_environment_is_initialized(), 0);
}

TEST_F(EnvironmentTest, EnvironmentRefCount) {
    dtl_environment_t env1 = nullptr;
    dtl_status status1 = dtl_environment_create(&env1);
    ASSERT_EQ(status1, DTL_SUCCESS);

    EXPECT_EQ(dtl_environment_ref_count(), 1u);

    dtl_environment_t env2 = nullptr;
    dtl_status status2 = dtl_environment_create(&env2);
    ASSERT_EQ(status2, DTL_SUCCESS);

    EXPECT_EQ(dtl_environment_ref_count(), 2u);

    dtl_environment_destroy(env2);
    EXPECT_EQ(dtl_environment_ref_count(), 1u);

    dtl_environment_destroy(env1);
    EXPECT_EQ(dtl_environment_ref_count(), 0u);
}

// ============================================================================
// Backend Availability Tests
// ============================================================================

TEST_F(EnvironmentTest, EnvironmentBackendQueries) {
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Each backend query should return a boolean value (0 or 1)
    int has_mpi = dtl_environment_has_mpi();
    EXPECT_TRUE(has_mpi == 0 || has_mpi == 1);

    int has_cuda = dtl_environment_has_cuda();
    EXPECT_TRUE(has_cuda == 0 || has_cuda == 1);

    int has_hip = dtl_environment_has_hip();
    EXPECT_TRUE(has_hip == 0 || has_hip == 1);

    int has_nccl = dtl_environment_has_nccl();
    EXPECT_TRUE(has_nccl == 0 || has_nccl == 1);

    int has_shmem = dtl_environment_has_shmem();
    EXPECT_TRUE(has_shmem == 0 || has_shmem == 1);

    dtl_environment_destroy(env);
}

TEST_F(EnvironmentTest, EnvironmentMpiThreadLevel) {
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);

    int level = dtl_environment_mpi_thread_level();
    // MPI thread levels: -1 (no MPI), 0 (SINGLE), 1 (FUNNELED),
    //                     2 (SERIALIZED), 3 (MULTIPLE)
    EXPECT_GE(level, -1);
    EXPECT_LE(level, 3);

    dtl_environment_destroy(env);
}

// ============================================================================
// Context Factory Tests
// ============================================================================

TEST_F(EnvironmentTest, EnvironmentMakeWorldContext) {
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_context_t ctx = nullptr;
    status = dtl_environment_make_world_context(env, &ctx);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    // Verify context has valid rank and size
    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);
    EXPECT_GE(rank, 0);
    EXPECT_GT(size, 0);
    EXPECT_LT(rank, size);

    dtl_context_destroy(ctx);
    dtl_environment_destroy(env);
}

TEST_F(EnvironmentTest, EnvironmentMakeWorldContextNullCtx) {
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);

    status = dtl_environment_make_world_context(env, nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);

    dtl_environment_destroy(env);
}

TEST_F(EnvironmentTest, EnvironmentMakeWorldContextInvalidEnv) {
    dtl_context_t ctx = nullptr;
    dtl_status status = dtl_environment_make_world_context(nullptr, &ctx);
    EXPECT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);
}

TEST_F(EnvironmentTest, EnvironmentMakeCpuContext) {
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_context_t ctx = nullptr;
    status = dtl_environment_make_cpu_context(env, &ctx);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    // CPU-only context should have rank=0 and size=1
    EXPECT_EQ(dtl_context_rank(ctx), 0);
    EXPECT_EQ(dtl_context_size(ctx), 1);

    dtl_context_destroy(ctx);
    dtl_environment_destroy(env);
}

TEST_F(EnvironmentTest, EnvironmentMakeWorldContextGpu) {
    dtl_environment_t env = nullptr;
    dtl_status status = dtl_environment_create(&env);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_context_t ctx = nullptr;
    status = dtl_environment_make_world_context_gpu(env, 0, &ctx);

    if (dtl_environment_has_cuda()) {
        // If CUDA is available, expect success
        EXPECT_EQ(status, DTL_SUCCESS);
        if (status == DTL_SUCCESS) {
            ASSERT_NE(ctx, nullptr);
            dtl_rank_t rank = dtl_context_rank(ctx);
            dtl_rank_t size = dtl_context_size(ctx);
            EXPECT_GE(rank, 0);
            EXPECT_GT(size, 0);
            dtl_context_destroy(ctx);
        }
    } else {
        // Without CUDA, expect an error (not success)
        EXPECT_NE(status, DTL_SUCCESS);
    }

    dtl_environment_destroy(env);
}
