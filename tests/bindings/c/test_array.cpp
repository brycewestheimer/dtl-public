// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_array.cpp
 * @brief Unit tests for DTL C bindings distributed array
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>

// ============================================================================
// Test Fixture
// ============================================================================

class CBindingsArray : public ::testing::Test {
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
// Array Creation Tests
// ============================================================================

TEST_F(CBindingsArray, CreateInt32Succeeds) {
    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create(ctx, DTL_DTYPE_INT32, 1000, &arr);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(arr, nullptr);
    EXPECT_EQ(dtl_array_global_size(arr), 1000u);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, CreateFloat64Succeeds) {
    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create(ctx, DTL_DTYPE_FLOAT64, 500, &arr);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_array_dtype(arr), DTL_DTYPE_FLOAT64);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, CreateWithFillSucceeds) {
    int32_t fill_value = 42;
    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_fill(ctx, DTL_DTYPE_INT32, 100,
                                               &fill_value, &arr);

    EXPECT_EQ(status, DTL_SUCCESS);

    // Check that values are filled
    const int32_t* data = static_cast<const int32_t*>(dtl_array_local_data(arr));
    for (dtl_size_t i = 0; i < dtl_array_local_size(arr); ++i) {
        EXPECT_EQ(data[i], 42);
    }

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, CreateZeroSizeSucceeds) {
    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create(ctx, DTL_DTYPE_INT32, 0, &arr);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_array_global_size(arr), 0u);
    EXPECT_EQ(dtl_array_local_size(arr), 0u);
    EXPECT_EQ(dtl_array_empty(arr), 1);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, CreateWithNullContextFails) {
    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create(nullptr, DTL_DTYPE_INT32, 100, &arr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsArray, CreateWithNullOutputFails) {
    dtl_status status = dtl_array_create(ctx, DTL_DTYPE_INT32, 100, nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);
}

TEST_F(CBindingsArray, CreateWithInvalidDtypeFails) {
    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create(ctx, static_cast<dtl_dtype>(999), 100, &arr);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Array Destruction Tests
// ============================================================================

TEST_F(CBindingsArray, DestroyNullIsSafe) {
    dtl_array_destroy(nullptr);  // Should not crash
}

TEST_F(CBindingsArray, IsValidAfterCreate) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    EXPECT_EQ(dtl_array_is_valid(arr), 1);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, IsValidNullIsFalse) {
    EXPECT_EQ(dtl_array_is_valid(nullptr), 0);
}

// ============================================================================
// Size Query Tests
// ============================================================================

TEST_F(CBindingsArray, LocalSizeSumsToGlobal) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 1000, &arr);

    // In single-process mode, local_size == global_size
    if (size() == 1) {
        EXPECT_EQ(dtl_array_local_size(arr), 1000u);
    } else {
        // local_size should be approximately global_size / num_ranks
        EXPECT_LE(dtl_array_local_size(arr), 1000u / size() + 1);
    }

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, LocalOffsetCorrect) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 1000, &arr);

    // First rank should have offset 0
    if (rank() == 0) {
        EXPECT_EQ(dtl_array_local_offset(arr), 0);
    }

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, EmptyReturnsFalseForNonEmpty) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    EXPECT_EQ(dtl_array_empty(arr), 0);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, GlobalSizeIsFixed) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    // Size should remain 100
    EXPECT_EQ(dtl_array_global_size(arr), 100u);

    // Note: No resize operation exists for arrays

    dtl_array_destroy(arr);
}

// ============================================================================
// Local Data Access Tests
// ============================================================================

TEST_F(CBindingsArray, LocalDataNotNull) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    const void* data = dtl_array_local_data(arr);
    EXPECT_NE(data, nullptr);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, LocalDataMutNotNull) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    void* data = dtl_array_local_data_mut(arr);
    EXPECT_NE(data, nullptr);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, GetSetLocalSucceeds) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    int32_t value = 42;
    dtl_status status = dtl_array_set_local(arr, 0, &value);
    EXPECT_EQ(status, DTL_SUCCESS);

    int32_t retrieved = 0;
    status = dtl_array_get_local(arr, 0, &retrieved);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(retrieved, 42);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, GetLocalOutOfBoundsFails) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    int32_t value = 0;
    dtl_size_t local_size = dtl_array_local_size(arr);
    dtl_status status = dtl_array_get_local(arr, local_size, &value);  // Out of bounds
    EXPECT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, SetLocalOutOfBoundsFails) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    int32_t value = 42;
    dtl_size_t local_size = dtl_array_local_size(arr);
    dtl_status status = dtl_array_set_local(arr, local_size, &value);  // Out of bounds
    EXPECT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_array_destroy(arr);
}

// ============================================================================
// Distribution Query Tests
// ============================================================================

TEST_F(CBindingsArray, NumRanksCorrect) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    EXPECT_EQ(dtl_array_num_ranks(arr), size());

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, RankCorrect) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    EXPECT_EQ(dtl_array_rank(arr), rank());

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, IsLocalForLocalIndices) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    dtl_index_t offset = dtl_array_local_offset(arr);
    dtl_size_t local_size = dtl_array_local_size(arr);

    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(dtl_array_is_local(arr, offset + i), 1);
    }

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, OwnerReturnsCorrectRank) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    dtl_index_t offset = dtl_array_local_offset(arr);
    dtl_rank_t owner = dtl_array_owner(arr, offset);
    EXPECT_EQ(owner, rank());

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, ToLocalConversion) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    dtl_index_t offset = dtl_array_local_offset(arr);
    dtl_index_t local_idx = dtl_array_to_local(arr, offset);
    EXPECT_EQ(local_idx, 0);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, ToGlobalConversion) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    dtl_index_t offset = dtl_array_local_offset(arr);
    dtl_index_t global_idx = dtl_array_to_global(arr, 0);
    EXPECT_EQ(global_idx, offset);

    dtl_array_destroy(arr);
}

// ============================================================================
// Local Operation Tests
// ============================================================================

TEST_F(CBindingsArray, FillLocalSucceeds) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    int32_t fill = 99;
    dtl_status status = dtl_array_fill_local(arr, &fill);
    EXPECT_EQ(status, DTL_SUCCESS);

    const int32_t* data = static_cast<const int32_t*>(dtl_array_local_data(arr));
    for (dtl_size_t i = 0; i < dtl_array_local_size(arr); ++i) {
        EXPECT_EQ(data[i], 99);
    }

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, BarrierSucceeds) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    auto start = std::chrono::steady_clock::now();
    if (size() > 1 && rank() == size() - 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    dtl_status status = dtl_array_barrier(arr);
    EXPECT_EQ(status, DTL_SUCCESS);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    if (size() > 1 && rank() != size() - 1) {
        EXPECT_GE(elapsed.count(), 75);
    }

    dtl_array_destroy(arr);
}

// ============================================================================
// Type-Specific Tests
// ============================================================================

TEST_F(CBindingsArray, Float32Operations) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_FLOAT32, 100, &arr);

    float value = 3.14f;
    dtl_array_set_local(arr, 0, &value);

    float retrieved = 0.0f;
    dtl_array_get_local(arr, 0, &retrieved);
    EXPECT_FLOAT_EQ(retrieved, 3.14f);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, Int64Operations) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT64, 100, &arr);

    int64_t value = 9223372036854775807LL;  // Max int64
    dtl_array_set_local(arr, 0, &value);

    int64_t retrieved = 0;
    dtl_array_get_local(arr, 0, &retrieved);
    EXPECT_EQ(retrieved, value);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsArray, Uint8Operations) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_UINT8, 100, &arr);

    uint8_t value = 255;
    dtl_array_set_local(arr, 0, &value);

    uint8_t retrieved = 0;
    dtl_array_get_local(arr, 0, &retrieved);
    EXPECT_EQ(retrieved, 255);

    dtl_array_destroy(arr);
}

// ============================================================================
// Array vs Vector Difference Tests
// ============================================================================

TEST_F(CBindingsArray, ArrayHasNoResize) {
    // This test documents that arrays have no resize function
    // The dtl_array_resize function does not exist in the API
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_INT32, 100, &arr);

    // Size remains fixed
    EXPECT_EQ(dtl_array_global_size(arr), 100u);

    // Unlike vector, there is no dtl_array_resize()

    dtl_array_destroy(arr);
}
