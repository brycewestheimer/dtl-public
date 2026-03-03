// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_vector.cpp
 * @brief Unit tests for DTL C bindings distributed vector
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>

// ============================================================================
// Test Fixture
// ============================================================================

class CBindingsVector : public ::testing::Test {
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
// Vector Creation Tests
// ============================================================================

TEST_F(CBindingsVector, CreateInt32Succeeds) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_INT32, 1000, &vec);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(vec, nullptr);
    EXPECT_EQ(dtl_vector_global_size(vec), 1000u);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, CreateFloat64Succeeds) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 500, &vec);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_vector_dtype(vec), DTL_DTYPE_FLOAT64);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, CreateWithFillSucceeds) {
    int32_t fill_value = 42;
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 100,
                                                &fill_value, &vec);

    EXPECT_EQ(status, DTL_SUCCESS);

    // Check that values are filled
    const int32_t* data = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    for (dtl_size_t i = 0; i < dtl_vector_local_size(vec); ++i) {
        EXPECT_EQ(data[i], 42);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, CreateZeroSizeSucceeds) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_INT32, 0, &vec);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_vector_global_size(vec), 0u);
    EXPECT_EQ(dtl_vector_local_size(vec), 0u);
    EXPECT_EQ(dtl_vector_empty(vec), 1);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, CreateWithNullContextFails) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(nullptr, DTL_DTYPE_INT32, 100, &vec);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsVector, CreateWithNullOutputFails) {
    dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);
}

TEST_F(CBindingsVector, CreateWithInvalidDtypeFails) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(ctx, static_cast<dtl_dtype>(999), 100, &vec);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Vector Destruction Tests
// ============================================================================

TEST_F(CBindingsVector, DestroyNullIsSafe) {
    dtl_vector_destroy(nullptr);  // Should not crash
}

TEST_F(CBindingsVector, IsValidAfterCreate) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    EXPECT_EQ(dtl_vector_is_valid(vec), 1);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, IsValidNullIsFalse) {
    EXPECT_EQ(dtl_vector_is_valid(nullptr), 0);
}

// ============================================================================
// Size Query Tests
// ============================================================================

TEST_F(CBindingsVector, LocalSizeSumsToGlobal) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 1000, &vec);

    // In single-process mode, local_size == global_size
    if (size() == 1) {
        EXPECT_EQ(dtl_vector_local_size(vec), 1000u);
    } else {
        // local_size should be approximately global_size / num_ranks
        EXPECT_LE(dtl_vector_local_size(vec), 1000u / size() + 1);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, LocalOffsetCorrect) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 1000, &vec);

    // First rank should have offset 0
    if (rank() == 0) {
        EXPECT_EQ(dtl_vector_local_offset(vec), 0);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, EmptyReturnsFalseForNonEmpty) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    EXPECT_EQ(dtl_vector_empty(vec), 0);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Local Data Access Tests
// ============================================================================

TEST_F(CBindingsVector, LocalDataNotNull) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    const void* data = dtl_vector_local_data(vec);
    EXPECT_NE(data, nullptr);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, LocalDataMutNotNull) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    void* data = dtl_vector_local_data_mut(vec);
    EXPECT_NE(data, nullptr);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, GetSetLocalSucceeds) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    int32_t value = 42;
    dtl_status status = dtl_vector_set_local(vec, 0, &value);
    EXPECT_EQ(status, DTL_SUCCESS);

    int32_t retrieved = 0;
    status = dtl_vector_get_local(vec, 0, &retrieved);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(retrieved, 42);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, GetLocalOutOfBoundsFails) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    int32_t value = 0;
    dtl_size_t local_size = dtl_vector_local_size(vec);
    dtl_status status = dtl_vector_get_local(vec, local_size, &value);  // Out of bounds
    EXPECT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, SetLocalOutOfBoundsFails) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    int32_t value = 42;
    dtl_size_t local_size = dtl_vector_local_size(vec);
    dtl_status status = dtl_vector_set_local(vec, local_size, &value);  // Out of bounds
    EXPECT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Distribution Query Tests
// ============================================================================

TEST_F(CBindingsVector, NumRanksCorrect) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    EXPECT_EQ(dtl_vector_num_ranks(vec), size());

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, RankCorrect) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    EXPECT_EQ(dtl_vector_rank(vec), rank());

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, IsLocalForLocalIndices) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    dtl_index_t offset = dtl_vector_local_offset(vec);
    dtl_size_t local_size = dtl_vector_local_size(vec);

    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(dtl_vector_is_local(vec, offset + i), 1);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, OwnerReturnsCorrectRank) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    dtl_index_t offset = dtl_vector_local_offset(vec);
    dtl_rank_t owner = dtl_vector_owner(vec, offset);
    EXPECT_EQ(owner, rank());

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, ToLocalConversion) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    dtl_index_t offset = dtl_vector_local_offset(vec);
    dtl_index_t local_idx = dtl_vector_to_local(vec, offset);
    EXPECT_EQ(local_idx, 0);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, ToGlobalConversion) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    dtl_index_t offset = dtl_vector_local_offset(vec);
    dtl_index_t global_idx = dtl_vector_to_global(vec, 0);
    EXPECT_EQ(global_idx, offset);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Collective Operation Tests
// ============================================================================

TEST_F(CBindingsVector, ResizeSucceeds) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    dtl_status status = dtl_vector_resize(vec, 200);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(dtl_vector_global_size(vec), 200u);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, FillLocalSucceeds) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    int32_t fill = 99;
    dtl_status status = dtl_vector_fill_local(vec, &fill);
    EXPECT_EQ(status, DTL_SUCCESS);

    const int32_t* data = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    for (dtl_size_t i = 0; i < dtl_vector_local_size(vec); ++i) {
        EXPECT_EQ(data[i], 99);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, BarrierSucceeds) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec);

    auto start = std::chrono::steady_clock::now();
    if (size() > 1 && rank() == size() - 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    dtl_status status = dtl_vector_barrier(vec);
    EXPECT_EQ(status, DTL_SUCCESS);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    if (size() > 1 && rank() != size() - 1) {
        EXPECT_GE(elapsed.count(), 75);
    }

    dtl_vector_destroy(vec);
}

// ============================================================================
// Type-Specific Tests
// ============================================================================

TEST_F(CBindingsVector, Float32Operations) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_FLOAT32, 100, &vec);

    float value = 3.14f;
    dtl_vector_set_local(vec, 0, &value);

    float retrieved = 0.0f;
    dtl_vector_get_local(vec, 0, &retrieved);
    EXPECT_FLOAT_EQ(retrieved, 3.14f);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, Int64Operations) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_INT64, 100, &vec);

    int64_t value = 9223372036854775807LL;  // Max int64
    dtl_vector_set_local(vec, 0, &value);

    int64_t retrieved = 0;
    dtl_vector_get_local(vec, 0, &retrieved);
    EXPECT_EQ(retrieved, value);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsVector, Uint8Operations) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_UINT8, 100, &vec);

    uint8_t value = 255;
    dtl_vector_set_local(vec, 0, &value);

    uint8_t retrieved = 0;
    dtl_vector_get_local(vec, 0, &retrieved);
    EXPECT_EQ(retrieved, 255);

    dtl_vector_destroy(vec);
}
