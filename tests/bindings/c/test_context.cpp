// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_context.cpp
 * @brief Unit tests for DTL C bindings context operations
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>

#ifdef DTL_HAS_CUDA
#include <cuda_runtime.h>
#endif

// ============================================================================
// Context Creation Tests
// ============================================================================

TEST(CBindingsContext, CreateDefaultSucceeds) {
    dtl_context_t ctx = nullptr;
    dtl_status status = dtl_context_create_default(&ctx);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, CreateWithOptionsSucceeds) {
    dtl_context_options opts;
    dtl_context_options_init(&opts);
    opts.device_id = -1;  // CPU only

    dtl_context_t ctx = nullptr;
    dtl_status status = dtl_context_create(&ctx, &opts);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, CreateWithNullOptsUsesDefaults) {
    dtl_context_t ctx = nullptr;
    dtl_status status = dtl_context_create(&ctx, nullptr);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(ctx, nullptr);
    EXPECT_EQ(dtl_context_device_id(ctx), -1);  // Default is CPU

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, CreateWithNullContextFails) {
    dtl_status status = dtl_context_create(nullptr, nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);
}

// ============================================================================
// Context Destruction Tests
// ============================================================================

TEST(CBindingsContext, DestroyNullIsSafe) {
    // Should not crash
    dtl_context_destroy(nullptr);
}

TEST(CBindingsContext, DestroyAfterNullingHandleIsSafe) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);
    dtl_context_destroy(ctx);
    // After destruction, callers must not reuse the old handle.
    // Setting to NULL makes subsequent dtl_context_destroy() calls safe.
    ctx = nullptr;
    dtl_context_destroy(ctx);
}

// ============================================================================
// Context Query Tests
// ============================================================================

TEST(CBindingsContext, RankIsNonNegative) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_rank_t rank = dtl_context_rank(ctx);
    EXPECT_GE(rank, 0);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, SizeIsPositive) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_rank_t size = dtl_context_size(ctx);
    EXPECT_GT(size, 0);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, RankLessThanSize) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_rank_t rank = dtl_context_rank(ctx);
    dtl_rank_t size = dtl_context_size(ctx);
    EXPECT_LT(rank, size);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, NullContextRankReturnsNoRank) {
    EXPECT_EQ(dtl_context_rank(nullptr), DTL_NO_RANK);
}

TEST(CBindingsContext, NullContextSizeReturnsZero) {
    EXPECT_EQ(dtl_context_size(nullptr), 0);
}

TEST(CBindingsContext, DeviceIdDefaultIsCPU) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    int device_id = dtl_context_device_id(ctx);
    EXPECT_EQ(device_id, -1);  // CPU only

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, HasDeviceReturnsFalseForCPU) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    EXPECT_EQ(dtl_context_has_device(ctx), 0);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, IsRootForSingleProcess) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    // In single-process mode, we are always root
    if (dtl_context_size(ctx) == 1) {
        EXPECT_EQ(dtl_context_is_root(ctx), 1);
    }

    dtl_context_destroy(ctx);
}

// ============================================================================
// Context Validation Tests
// ============================================================================

TEST(CBindingsContext, IsValidReturnsTrueForValidContext) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    EXPECT_EQ(dtl_context_is_valid(ctx), 1);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, IsValidReturnsFalseForNull) {
    EXPECT_EQ(dtl_context_is_valid(nullptr), 0);
}

// ============================================================================
// Context Synchronization Tests
// ============================================================================

TEST(CBindingsContext, BarrierSucceeds) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_status status = dtl_context_barrier(ctx);
    EXPECT_EQ(status, DTL_SUCCESS);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, BarrierWithNullFails) {
    dtl_status status = dtl_context_barrier(nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsContext, FenceSucceeds) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_status status = dtl_context_fence(ctx);
    EXPECT_EQ(status, DTL_SUCCESS);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, FenceWithNullFails) {
    dtl_status status = dtl_context_fence(nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Context Duplication Tests
// ============================================================================

TEST(CBindingsContext, DupCreatesNewContext) {
    dtl_context_t ctx1 = nullptr;
    dtl_context_create_default(&ctx1);

    dtl_context_t ctx2 = nullptr;
    dtl_status status = dtl_context_dup(ctx1, &ctx2);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(ctx2, nullptr);
    EXPECT_NE(ctx1, ctx2);

    // Both should have same rank/size
    EXPECT_EQ(dtl_context_rank(ctx1), dtl_context_rank(ctx2));
    EXPECT_EQ(dtl_context_size(ctx1), dtl_context_size(ctx2));

    dtl_context_destroy(ctx2);
    dtl_context_destroy(ctx1);
}

TEST(CBindingsContext, DupWithNullSourceFails) {
    dtl_context_t ctx = nullptr;
    dtl_status status = dtl_context_dup(nullptr, &ctx);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST(CBindingsContext, DupWithNullDestFails) {
    dtl_context_t ctx = nullptr;
    dtl_context_create_default(&ctx);

    dtl_status status = dtl_context_dup(ctx, nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);

    dtl_context_destroy(ctx);
}

// ============================================================================
// Context Options Tests
// ============================================================================

TEST(CBindingsContext, OptionsInitSetsDefaults) {
    dtl_context_options opts;
    dtl_context_options_init(&opts);

    EXPECT_EQ(opts.device_id, -1);
    EXPECT_EQ(opts.init_mpi, 1);
    EXPECT_EQ(opts.finalize_mpi, 0);
}

TEST(CBindingsContext, OptionsInitNullIsSafe) {
    // Should not crash
    dtl_context_options_init(nullptr);
}

// =============================================================================
// Extended context API tests
// =============================================================================

TEST(CBindingsContext, HasMpiQuery) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    // has_mpi should return 0 or 1
    int has_mpi = dtl_context_has_mpi(ctx);
    EXPECT_TRUE(has_mpi == 0 || has_mpi == 1);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, HasCudaQuery) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    int has_cuda = dtl_context_has_cuda(ctx);
    EXPECT_TRUE(has_cuda == 0 || has_cuda == 1);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, HasNcclQuery) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    int has_nccl = dtl_context_has_nccl(ctx);
    EXPECT_TRUE(has_nccl == 0 || has_nccl == 1);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, HasShmemQuery) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    int has_shmem = dtl_context_has_shmem(ctx);
    EXPECT_TRUE(has_shmem == 0 || has_shmem == 1);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, SplitContext) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    dtl_context_t split_ctx = nullptr;
    dtl_status status = dtl_context_split(ctx, /*color=*/0, /*key=*/0, &split_ctx);
    // Split may fail in single-rank mode without MPI; that's acceptable
    if (status == DTL_SUCCESS) {
        EXPECT_NE(split_ctx, nullptr);
        EXPECT_GE(dtl_context_rank(split_ctx), 0);
        dtl_context_destroy(split_ctx);
    }

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, WithCuda) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    dtl_context_t cuda_ctx = nullptr;
    dtl_status status = dtl_context_with_cuda(ctx, /*device_id=*/0, &cuda_ctx);
    // May fail if CUDA not available; that's acceptable
    if (status == DTL_SUCCESS) {
        EXPECT_NE(cuda_ctx, nullptr);
        EXPECT_EQ(dtl_context_has_cuda(cuda_ctx), 1);
        dtl_context_destroy(cuda_ctx);
    }

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, WithCudaRejectsOutOfRangeDevice) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

#ifdef DTL_HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count < 0) {
        GTEST_SKIP() << "CUDA runtime not available for device validation";
    }

    const int invalid_device = device_count + 1;
    dtl_context_t cuda_ctx = nullptr;
    dtl_status status = dtl_context_with_cuda(ctx, invalid_device, &cuda_ctx);

    if (device_count == 0) {
        EXPECT_EQ(status, DTL_ERROR_BACKEND_UNAVAILABLE);
    } else {
        EXPECT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);
    }
    EXPECT_EQ(cuda_ctx, nullptr);
#else
    dtl_context_t cuda_ctx = nullptr;
    dtl_status status = dtl_context_with_cuda(ctx, /*device_id=*/0, &cuda_ctx);
    EXPECT_EQ(status, DTL_ERROR_BACKEND_UNAVAILABLE);
    EXPECT_EQ(cuda_ctx, nullptr);
#endif

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, WithNccl) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    dtl_context_t nccl_ctx = nullptr;
    dtl_status status = dtl_context_with_nccl(ctx, /*device_id=*/0, &nccl_ctx);
    if (status == DTL_SUCCESS) {
        EXPECT_NE(nccl_ctx, nullptr);
        EXPECT_EQ(dtl_context_has_nccl(nccl_ctx), 1);
        EXPECT_EQ(dtl_context_nccl_mode(nccl_ctx), DTL_NCCL_MODE_HYBRID_PARITY);
        dtl_context_destroy(nccl_ctx);
    } else {
        EXPECT_TRUE(status == DTL_ERROR_BACKEND_UNAVAILABLE ||
                    status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_INVALID_ARGUMENT ||
                    status == DTL_ERROR_MPI ||
                    status == DTL_ERROR_NCCL ||
                    status == DTL_ERROR_CUDA);
        EXPECT_EQ(nccl_ctx, nullptr);
    }

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, SplitNccl) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    dtl_context_t nccl_ctx = nullptr;
    dtl_status create_status = dtl_context_with_nccl(ctx, /*device_id=*/0, &nccl_ctx);
    if (create_status != DTL_SUCCESS) {
        dtl_context_destroy(ctx);
        GTEST_SKIP() << "NCCL context unavailable in this environment";
    }

    dtl_context_t split_ctx = nullptr;
    dtl_status status = dtl_context_split_nccl(nccl_ctx, /*color=*/0, /*key=*/0, &split_ctx);
    if (status == DTL_SUCCESS) {
        EXPECT_NE(split_ctx, nullptr);
        EXPECT_EQ(dtl_context_has_nccl(split_ctx), 1);
        dtl_context_destroy(split_ctx);
    } else {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE ||
                    status == DTL_ERROR_INVALID_ARGUMENT ||
                    status == DTL_ERROR_MPI ||
                    status == DTL_ERROR_NCCL ||
                    status == DTL_ERROR_CUDA);
        EXPECT_EQ(split_ctx, nullptr);
    }

    dtl_context_destroy(nccl_ctx);
    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, NcclCapabilityQueriesReturnStableValues) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    // Without NCCL, mode should be -1 and capability queries should be 0.
    EXPECT_EQ(dtl_context_nccl_mode(ctx), -1);
    EXPECT_EQ(dtl_context_nccl_supports_native(ctx, DTL_NCCL_OP_ALLREDUCE), 0);
    EXPECT_EQ(dtl_context_nccl_supports_hybrid(ctx, DTL_NCCL_OP_SCAN), 0);

    dtl_context_destroy(ctx);
}

TEST(CBindingsContext, WithNcclExNativeOnlyMode) {
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    dtl_context_t nccl_ctx = nullptr;
    dtl_status status = dtl_context_with_nccl_ex(
        ctx, /*device_id=*/0, DTL_NCCL_MODE_NATIVE_ONLY, &nccl_ctx);

    if (status == DTL_SUCCESS) {
        EXPECT_EQ(dtl_context_nccl_mode(nccl_ctx), DTL_NCCL_MODE_NATIVE_ONLY);
        EXPECT_EQ(dtl_context_nccl_supports_native(nccl_ctx, DTL_NCCL_OP_ALLREDUCE), 1);
        EXPECT_EQ(dtl_context_nccl_supports_hybrid(nccl_ctx, DTL_NCCL_OP_SCAN), 0);
        dtl_context_destroy(nccl_ctx);
    } else {
        EXPECT_TRUE(status == DTL_ERROR_BACKEND_UNAVAILABLE ||
                    status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_INVALID_ARGUMENT ||
                    status == DTL_ERROR_MPI ||
                    status == DTL_ERROR_NCCL ||
                    status == DTL_ERROR_CUDA);
    }

    dtl_context_destroy(ctx);
}
