// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_tensor_cuda.cpp
 * @brief CUDA placement tests for C binding tensors
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_tensor.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <cstring>
#include <vector>

#ifdef DTL_HAS_CUDA
#include <cuda_runtime.h>
static bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
#else
static bool cuda_available() { return false; }
#endif

class CudaTensorTest : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;

    void SetUp() override {
        if (!cuda_available()) {
            GTEST_SKIP() << "CUDA hardware not available";
        }
        ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);
    }

    void TearDown() override {
        if (ctx) dtl_context_destroy(ctx);
    }
};

// Tensor currently doesn't have create_with_options, so CUDA tensor tests
// are limited to verifying that the dispatch layer handles non-host placements
// via direct dispatch testing when the infrastructure supports it.

#ifdef DTL_HAS_CUDA

TEST_F(CudaTensorTest, DefaultTensorCreationIsHost) {
    dtl_shape shape = {};
    shape.ndim = 2;
    shape.dims[0] = 4;
    shape.dims[1] = 5;

    dtl_tensor_t tensor = nullptr;
    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape, &tensor);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(tensor, nullptr);

    // Default creation is HOST placement - data should be accessible
    EXPECT_NE(dtl_tensor_local_data(tensor), nullptr);

    EXPECT_EQ(dtl_tensor_ndim(tensor), 2);

    dtl_tensor_destroy(tensor);
}

TEST_F(CudaTensorTest, ShapeQueriesWork) {
    dtl_shape shape = {};
    shape.ndim = 3;
    shape.dims[0] = 2;
    shape.dims[1] = 3;
    shape.dims[2] = 4;

    dtl_tensor_t tensor = nullptr;
    ASSERT_EQ(dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shape, &tensor), DTL_SUCCESS);

    EXPECT_EQ(dtl_tensor_ndim(tensor), 3);
    EXPECT_EQ(dtl_tensor_dim(tensor, 0), 2u);
    EXPECT_EQ(dtl_tensor_dim(tensor, 1), 3u);
    EXPECT_EQ(dtl_tensor_dim(tensor, 2), 4u);

    dtl_tensor_destroy(tensor);
}

#endif  // DTL_HAS_CUDA
