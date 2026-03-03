// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_tensor.cpp
 * @brief Unit tests for DTL C bindings distributed tensor
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_tensor.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <vector>

// ============================================================================
// Test Fixture
// ============================================================================

class CBindingsTensor : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;

    void SetUp() override {
        dtl_status status = dtl_context_create_default(&ctx);
        ASSERT_EQ(status, DTL_SUCCESS);
    }

    void TearDown() override {
        if (ctx) {
            dtl_context_destroy(ctx);
        }
    }

    dtl_rank_t rank() { return dtl_context_rank(ctx); }
    dtl_rank_t size() { return dtl_context_size(ctx); }
};

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST_F(CBindingsTensor, Create1DSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(1000);

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape, &tensor);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(dtl_tensor_ndim(tensor), 1);
    EXPECT_EQ(dtl_tensor_dim(tensor, 0), 1000u);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, Create2DSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(100, 64);

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shape, &tensor);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_tensor_ndim(tensor), 2);
    EXPECT_EQ(dtl_tensor_dim(tensor, 0), 100u);
    EXPECT_EQ(dtl_tensor_dim(tensor, 1), 64u);
    EXPECT_EQ(dtl_tensor_global_size(tensor), 100u * 64u);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, Create3DSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_3d(32, 64, 128);

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_tensor_ndim(tensor), 3);
    EXPECT_EQ(dtl_tensor_dim(tensor, 0), 32u);
    EXPECT_EQ(dtl_tensor_dim(tensor, 1), 64u);
    EXPECT_EQ(dtl_tensor_dim(tensor, 2), 128u);
    EXPECT_EQ(dtl_tensor_global_size(tensor), 32u * 64u * 128u);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, CreateWithFillSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    float fill_value = 3.14f;

    dtl_status status = dtl_tensor_create_fill(ctx, DTL_DTYPE_FLOAT32, shape,
                                                &fill_value, &tensor);

    EXPECT_EQ(status, DTL_SUCCESS);

    // Verify fill
    const float* data = static_cast<const float*>(dtl_tensor_local_data(tensor));
    for (dtl_size_t i = 0; i < dtl_tensor_local_size(tensor); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, CreateWithNullContextFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);

    dtl_status status = dtl_tensor_create(nullptr, DTL_DTYPE_INT32, shape, &tensor);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsTensor, CreateWithInvalidNdimFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape;
    shape.ndim = 0;  // Invalid

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsTensor, CreateWithTooManyDimsFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape;
    shape.ndim = DTL_MAX_TENSOR_RANK + 1;

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Shape Query Tests
// ============================================================================

TEST_F(CBindingsTensor, ShapeReturnsCorrect) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_3d(10, 20, 30);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_shape returned = dtl_tensor_shape(tensor);
    EXPECT_EQ(returned.ndim, 3);
    EXPECT_EQ(returned.dims[0], 10u);
    EXPECT_EQ(returned.dims[1], 20u);
    EXPECT_EQ(returned.dims[2], 30u);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, LocalShapePartitionsFirstDim) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(100, 50);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_shape local_shape = dtl_tensor_local_shape(tensor);

    // Second dimension should be unchanged
    EXPECT_EQ(local_shape.dims[1], 50u);

    // First dimension should be partitioned
    if (size() == 1) {
        EXPECT_EQ(local_shape.dims[0], 100u);
    } else {
        EXPECT_LE(local_shape.dims[0], 100u / size() + 1);
    }

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, DtypeCorrect) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shape, &tensor);

    EXPECT_EQ(dtl_tensor_dtype(tensor), DTL_DTYPE_FLOAT64);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, DistributedDimIsZero) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(100, 50);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_distributed_dim(tensor), 0);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Local Data Access Tests
// ============================================================================

TEST_F(CBindingsTensor, LocalDataNotNull) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    const void* data = dtl_tensor_local_data(tensor);
    EXPECT_NE(data, nullptr);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, StridesAreRowMajor) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_3d(4, 5, 6);  // 4x5x6 tensor
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    // For row-major: stride[dim] = product of dims after dim
    // In local shape, dim 0 might be different, but dims 1,2 are same
    dtl_size_t stride2 = dtl_tensor_stride(tensor, 2);
    dtl_size_t stride1 = dtl_tensor_stride(tensor, 1);

    EXPECT_EQ(stride2, 1u);          // Innermost stride is 1
    EXPECT_EQ(stride1, 6u);          // stride[1] = dims[2] = 6

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, GetSetLocalNDSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    // Set element at (0, 5) in local coords
    dtl_index_t indices[2] = {0, 5};
    int32_t value = 42;
    dtl_status status = dtl_tensor_set_local_nd(tensor, indices, &value);
    EXPECT_EQ(status, DTL_SUCCESS);

    // Get it back
    int32_t retrieved = 0;
    status = dtl_tensor_get_local_nd(tensor, indices, &retrieved);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(retrieved, 42);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, GetSetLocalLinearSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape, &tensor);

    float value = 2.71828f;
    dtl_status status = dtl_tensor_set_local(tensor, 0, &value);
    EXPECT_EQ(status, DTL_SUCCESS);

    float retrieved = 0.0f;
    status = dtl_tensor_get_local(tensor, 0, &retrieved);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_FLOAT_EQ(retrieved, 2.71828f);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, GetLocalNDOutOfBoundsFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_index_t indices[2] = {0, 100};  // Out of bounds
    int32_t value = 0;
    dtl_status status = dtl_tensor_get_local_nd(tensor, indices, &value);
    EXPECT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, GetLocalLinearOutOfBoundsFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    int32_t value = 0;
    dtl_size_t local_size = dtl_tensor_local_size(tensor);
    dtl_status status = dtl_tensor_get_local(tensor, local_size, &value);
    EXPECT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Distribution Query Tests
// ============================================================================

TEST_F(CBindingsTensor, NumRanksCorrect) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_num_ranks(tensor), size());

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, RankCorrect) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_rank(tensor), rank());

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Collective Operation Tests
// ============================================================================

TEST_F(CBindingsTensor, FillLocalSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    int32_t fill = 77;
    dtl_status status = dtl_tensor_fill_local(tensor, &fill);
    EXPECT_EQ(status, DTL_SUCCESS);

    const int32_t* data = static_cast<const int32_t*>(dtl_tensor_local_data(tensor));
    for (dtl_size_t i = 0; i < dtl_tensor_local_size(tensor); ++i) {
        EXPECT_EQ(data[i], 77);
    }

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, BarrierSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_status status = dtl_tensor_barrier(tensor);
    EXPECT_EQ(status, DTL_SUCCESS);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(CBindingsTensor, IsValidAfterCreate) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_is_valid(tensor), 1);

    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsTensor, IsValidNullIsFalse) {
    EXPECT_EQ(dtl_tensor_is_valid(nullptr), 0);
}

TEST_F(CBindingsTensor, DestroyNullIsSafe) {
    dtl_tensor_destroy(nullptr);  // Should not crash
}

// ============================================================================
// Multi-dimensional Index Tests
// ============================================================================

TEST_F(CBindingsTensor, IndexConversionConsistent) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_3d(4, 5, 6);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    // Set values with ND indexing
    dtl_shape local_shape = dtl_tensor_local_shape(tensor);
    for (int i = 0; i < static_cast<int>(local_shape.dims[0]); ++i) {
        for (int j = 0; j < static_cast<int>(local_shape.dims[1]); ++j) {
            for (int k = 0; k < static_cast<int>(local_shape.dims[2]); ++k) {
                dtl_index_t indices[3] = {i, j, k};
                int32_t value = i * 100 + j * 10 + k;
                dtl_tensor_set_local_nd(tensor, indices, &value);
            }
        }
    }

    // Read back and verify
    for (int i = 0; i < static_cast<int>(local_shape.dims[0]); ++i) {
        for (int j = 0; j < static_cast<int>(local_shape.dims[1]); ++j) {
            for (int k = 0; k < static_cast<int>(local_shape.dims[2]); ++k) {
                dtl_index_t indices[3] = {i, j, k};
                int32_t expected = i * 100 + j * 10 + k;
                int32_t retrieved = 0;
                dtl_tensor_get_local_nd(tensor, indices, &retrieved);
                EXPECT_EQ(retrieved, expected);
            }
        }
    }

    dtl_tensor_destroy(tensor);
}
