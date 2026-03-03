// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_tensor_v2.cpp
 * @brief Unit tests for DTL C bindings distributed tensor V2 (vtable-based)
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl.h>
#include <chrono>
#include <thread>

// ============================================================================
// Test Fixture
// ============================================================================

class TensorV2Test : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;

    void SetUp() override {
        dtl_status status = dtl_context_create_default(&ctx);
        ASSERT_EQ(status, DTL_SUCCESS);
        ASSERT_NE(ctx, nullptr);
    }

    void TearDown() override {
        if (ctx) {
            dtl_context_destroy(ctx);
            ctx = nullptr;
        }
    }

    dtl_rank_t rank() { return dtl_context_rank(ctx); }
    dtl_rank_t size() { return dtl_context_size(ctx); }
};

// ============================================================================
// Creation Tests
// ============================================================================

TEST_F(TensorV2Test, Create1D) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(1000);

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape, &tensor);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(dtl_tensor_ndim(tensor), 1);
    EXPECT_EQ(dtl_tensor_dim(tensor, 0), 1000u);

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, Create2D) {
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

TEST_F(TensorV2Test, Create3D) {
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

TEST_F(TensorV2Test, CreateWithFill) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    float fill_value = 3.14f;

    dtl_status status = dtl_tensor_create_fill(ctx, DTL_DTYPE_FLOAT32, shape,
                                                &fill_value, &tensor);
    EXPECT_EQ(status, DTL_SUCCESS);

    const float* data = static_cast<const float*>(dtl_tensor_local_data(tensor));
    ASSERT_NE(data, nullptr);
    for (dtl_size_t i = 0; i < dtl_tensor_local_size(tensor); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, CreateInvalidNdimFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape;
    shape.ndim = 0;

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(TensorV2Test, CreateTooManyDimsFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape;
    shape.ndim = DTL_MAX_TENSOR_RANK + 1;

    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(TensorV2Test, CreateNullContextFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);

    dtl_status status = dtl_tensor_create(nullptr, DTL_DTYPE_INT32, shape, &tensor);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Shape Query Tests
// ============================================================================

TEST_F(TensorV2Test, ShapeReturnsCorrect) {
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

TEST_F(TensorV2Test, LocalShapePartitionsFirstDim) {
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

TEST_F(TensorV2Test, GlobalSizeIsProductOfDims) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_3d(4, 5, 6);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_global_size(tensor), 4u * 5u * 6u);

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, LocalSizeConsistentWithLocalShape) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(100, 50);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_shape local_shape = dtl_tensor_local_shape(tensor);
    dtl_size_t expected_size = 1;
    for (int i = 0; i < local_shape.ndim; ++i) {
        expected_size *= local_shape.dims[i];
    }
    EXPECT_EQ(dtl_tensor_local_size(tensor), expected_size);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Stride Tests
// ============================================================================

TEST_F(TensorV2Test, StridesAreRowMajor) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_3d(4, 5, 6);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_size_t stride2 = dtl_tensor_stride(tensor, 2);
    dtl_size_t stride1 = dtl_tensor_stride(tensor, 1);

    EXPECT_EQ(stride2, 1u);   // Innermost stride is 1
    EXPECT_EQ(stride1, 6u);   // stride[1] = dims[2] = 6

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, StridesInvalidDimReturnsZero) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_stride(tensor, -1), 0u);
    EXPECT_EQ(dtl_tensor_stride(tensor, 2), 0u);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// ND Indexing Tests
// ============================================================================

TEST_F(TensorV2Test, NDSetThenGet) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_index_t indices[2] = {0, 5};
    int32_t value = 42;
    dtl_status status = dtl_tensor_set_local_nd(tensor, indices, &value);
    EXPECT_EQ(status, DTL_SUCCESS);

    int32_t retrieved = 0;
    status = dtl_tensor_get_local_nd(tensor, indices, &retrieved);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(retrieved, 42);

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, NDOutOfBoundsFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_index_t indices[2] = {0, 100};
    int32_t value = 0;
    dtl_status status = dtl_tensor_get_local_nd(tensor, indices, &value);
    EXPECT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Linear Indexing Tests
// ============================================================================

TEST_F(TensorV2Test, LinearSetThenGet) {
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

TEST_F(TensorV2Test, LinearOutOfBoundsFails) {
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
// ND-to-Linear Consistency Tests
// ============================================================================

TEST_F(TensorV2Test, NDToLinearConsistency) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_3d(4, 5, 6);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_shape local_shape = dtl_tensor_local_shape(tensor);

    // Set values with ND indexing
    for (int i = 0; i < static_cast<int>(local_shape.dims[0]); ++i) {
        for (int j = 0; j < static_cast<int>(local_shape.dims[1]); ++j) {
            for (int k = 0; k < static_cast<int>(local_shape.dims[2]); ++k) {
                dtl_index_t indices[3] = {i, j, k};
                int32_t value = i * 100 + j * 10 + k;
                dtl_tensor_set_local_nd(tensor, indices, &value);
            }
        }
    }

    // Read back via linear index and verify
    dtl_size_t stride0 = dtl_tensor_stride(tensor, 0);
    dtl_size_t stride1 = dtl_tensor_stride(tensor, 1);
    dtl_size_t stride2 = dtl_tensor_stride(tensor, 2);

    for (int i = 0; i < static_cast<int>(local_shape.dims[0]); ++i) {
        for (int j = 0; j < static_cast<int>(local_shape.dims[1]); ++j) {
            for (int k = 0; k < static_cast<int>(local_shape.dims[2]); ++k) {
                dtl_size_t linear = i * stride0 + j * stride1 + k * stride2;
                int32_t expected = i * 100 + j * 10 + k;
                int32_t retrieved = 0;
                dtl_tensor_get_local(tensor, linear, &retrieved);
                EXPECT_EQ(retrieved, expected);
            }
        }
    }

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Fill Tests
// ============================================================================

TEST_F(TensorV2Test, FillLocal) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    int32_t fill = 77;
    dtl_status status = dtl_tensor_fill_local(tensor, &fill);
    EXPECT_EQ(status, DTL_SUCCESS);

    const int32_t* data = static_cast<const int32_t*>(dtl_tensor_local_data(tensor));
    ASSERT_NE(data, nullptr);
    for (dtl_size_t i = 0; i < dtl_tensor_local_size(tensor); ++i) {
        EXPECT_EQ(data[i], 77);
    }

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST_F(TensorV2Test, ReshapeSameTotalSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    // Reshape to 1D with same total
    dtl_shape new_shape = dtl_shape_1d(100);
    dtl_status status = dtl_tensor_reshape(tensor, new_shape);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_tensor_ndim(tensor), 1);
    EXPECT_EQ(dtl_tensor_global_size(tensor), 100u);

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, ReshapeDifferentTotalFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_shape new_shape = dtl_shape_1d(50);  // Different total
    dtl_status status = dtl_tensor_reshape(tensor, new_shape);
    EXPECT_NE(status, DTL_SUCCESS);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// AllDtypes Test
// ============================================================================

TEST_F(TensorV2Test, AllDtypes) {
    dtl_dtype dtypes[] = {
        DTL_DTYPE_INT8, DTL_DTYPE_INT16, DTL_DTYPE_INT32, DTL_DTYPE_INT64,
        DTL_DTYPE_UINT8, DTL_DTYPE_UINT16, DTL_DTYPE_UINT32, DTL_DTYPE_UINT64,
        DTL_DTYPE_FLOAT32, DTL_DTYPE_FLOAT64, DTL_DTYPE_BYTE, DTL_DTYPE_BOOL
    };

    dtl_shape shape = dtl_shape_2d(4, 4);

    for (auto dtype : dtypes) {
        dtl_tensor_t tensor = nullptr;
        dtl_status status = dtl_tensor_create(ctx, dtype, shape, &tensor);
        EXPECT_EQ(status, DTL_SUCCESS) << "Failed for dtype " << dtype;
        EXPECT_EQ(dtl_tensor_dtype(tensor), dtype);
        dtl_tensor_destroy(tensor);
    }
}

// ============================================================================
// Distribution Query Tests
// ============================================================================

TEST_F(TensorV2Test, NumRanks) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_num_ranks(tensor), size());

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, Rank) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_rank(tensor), rank());

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, DistributedDimIsZero) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(100, 50);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_distributed_dim(tensor), 0);

    dtl_tensor_destroy(tensor);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(TensorV2Test, IsValidAfterCreate) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_is_valid(tensor), 1);

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, IsValidNullIsFalse) {
    EXPECT_EQ(dtl_tensor_is_valid(nullptr), 0);
}

TEST_F(TensorV2Test, DestroyInvalidatesHandle) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_EQ(dtl_tensor_is_valid(tensor), 1);
    dtl_tensor_destroy(tensor);
    // After destroy, the handle pointer is dangling, but the test
    // just verifies destroy doesn't crash. We can't safely call
    // is_valid on a dangling pointer.
}

TEST_F(TensorV2Test, DestroyNullIsSafe) {
    dtl_tensor_destroy(nullptr);  // Should not crash
}

// ============================================================================
// Null Pointer Error Tests
// ============================================================================

TEST_F(TensorV2Test, CreateNullTensorOutputFails) {
    dtl_shape shape = dtl_shape_1d(100);
    dtl_status status = dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);
}

TEST_F(TensorV2Test, FillNullValueFails) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    dtl_status status = dtl_tensor_fill_local(tensor, nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, BarrierSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    auto start = std::chrono::steady_clock::now();
    if (size() > 1 && rank() == size() - 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    dtl_status status = dtl_tensor_barrier(tensor);
    EXPECT_EQ(status, DTL_SUCCESS);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    if (size() > 1 && rank() != size() - 1) {
        EXPECT_GE(elapsed.count(), 75);
    }

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, DtypeIsCorrect) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_1d(100);
    dtl_tensor_create(ctx, DTL_DTYPE_FLOAT64, shape, &tensor);

    EXPECT_EQ(dtl_tensor_dtype(tensor), DTL_DTYPE_FLOAT64);

    dtl_tensor_destroy(tensor);
}

TEST_F(TensorV2Test, LocalDataNotNull) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(10, 10);
    dtl_tensor_create(ctx, DTL_DTYPE_INT32, shape, &tensor);

    EXPECT_NE(dtl_tensor_local_data(tensor), nullptr);
    EXPECT_NE(dtl_tensor_local_data_mut(tensor), nullptr);

    dtl_tensor_destroy(tensor);
}
