// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_algorithms.cpp
 * @brief Unit tests for DTL C algorithm bindings
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl.h>

#include <cmath>
#include <cstring>
#include <vector>

// ============================================================================
// Test Fixture
// ============================================================================

class AlgorithmTest : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;

    void SetUp() override {
        ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);
    }

    void TearDown() override {
        if (ctx) {
            dtl_context_destroy(ctx);
        }
    }
};

// ============================================================================
// For-Each Tests
// ============================================================================

namespace {

// Callback that doubles each element
void double_element(void* elem, dtl_size_t idx, void* user_data) {
    (void)idx;
    (void)user_data;
    double* val = static_cast<double*>(elem);
    *val *= 2.0;
}

// Callback that accumulates sum
void sum_element(const void* elem, dtl_size_t idx, void* user_data) {
    (void)idx;
    const double* val = static_cast<const double*>(elem);
    double* sum = static_cast<double*>(user_data);
    *sum += *val;
}

// Callback that counts elements (const version)
void count_element(const void* elem, dtl_size_t idx, void* user_data) {
    (void)elem;
    (void)idx;
    int* count = static_cast<int*>(user_data);
    (*count)++;
}

// Callback that counts elements (mutable version for for_each)
void count_element_mut(void* elem, dtl_size_t idx, void* user_data) {
    (void)elem;
    (void)idx;
    int* count = static_cast<int*>(user_data);
    (*count)++;
}

} // namespace

TEST_F(AlgorithmTest, ForEachVectorModify) {
    dtl_vector_t vec;
    double fill = 5.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &vec), DTL_SUCCESS);

    // Double all elements
    ASSERT_EQ(dtl_for_each_vector(vec, double_element, nullptr), DTL_SUCCESS);

    // Verify
    const double* data = static_cast<const double*>(dtl_vector_local_data(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 10.0);
    }

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, ForEachVectorConst) {
    dtl_vector_t vec;
    double fill = 3.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &vec), DTL_SUCCESS);

    // Sum all elements
    double sum = 0.0;
    ASSERT_EQ(dtl_for_each_vector_const(vec, sum_element, &sum), DTL_SUCCESS);

    // Total should be local_size * 3.0
    dtl_size_t local_size = dtl_vector_local_size(vec);
    EXPECT_DOUBLE_EQ(sum, local_size * 3.0);

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, ForEachArray) {
    dtl_array_t arr;
    int32_t fill = 7;
    ASSERT_EQ(dtl_array_create_fill(ctx, DTL_DTYPE_INT32, 20, &fill, &arr), DTL_SUCCESS);

    // Count elements
    int count = 0;
    ASSERT_EQ(dtl_for_each_array_const(arr, count_element, &count), DTL_SUCCESS);

    EXPECT_EQ(static_cast<dtl_size_t>(count), dtl_array_local_size(arr));

    dtl_array_destroy(arr);
}

TEST_F(AlgorithmTest, ForEachNullFunc) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 10, &vec), DTL_SUCCESS);

    EXPECT_EQ(dtl_for_each_vector(vec, nullptr, nullptr), DTL_ERROR_NULL_POINTER);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Transform Tests
// ============================================================================

namespace {

void square_transform(const void* in, void* out, dtl_size_t idx, void* user_data) {
    (void)idx;
    (void)user_data;
    const double* input = static_cast<const double*>(in);
    double* output = static_cast<double*>(out);
    *output = (*input) * (*input);
}

void add_offset_transform(const void* in, void* out, dtl_size_t idx, void* user_data) {
    const int32_t* input = static_cast<const int32_t*>(in);
    int32_t* output = static_cast<int32_t*>(out);
    int32_t offset = user_data ? *static_cast<int32_t*>(user_data) : 0;
    *output = *input + offset + static_cast<int32_t>(idx);
}

} // namespace

TEST_F(AlgorithmTest, TransformVectorSquare) {
    dtl_vector_t src, dst;
    double fill = 3.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &src), DTL_SUCCESS);
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 10, &dst), DTL_SUCCESS);

    ASSERT_EQ(dtl_transform_vector(src, dst, square_transform, nullptr), DTL_SUCCESS);

    const double* data = static_cast<const double*>(dtl_vector_local_data(dst));
    dtl_size_t local_size = dtl_vector_local_size(dst);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 9.0);
    }

    dtl_vector_destroy(src);
    dtl_vector_destroy(dst);
}

TEST_F(AlgorithmTest, TransformVectorInPlace) {
    dtl_vector_t vec;
    double fill = 4.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &vec), DTL_SUCCESS);

    // In-place transform
    ASSERT_EQ(dtl_transform_vector(vec, vec, square_transform, nullptr), DTL_SUCCESS);

    const double* data = static_cast<const double*>(dtl_vector_local_data(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 16.0);
    }

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, TransformArray) {
    dtl_array_t src, dst;
    int32_t fill = 10;
    int32_t offset = 100;
    ASSERT_EQ(dtl_array_create_fill(ctx, DTL_DTYPE_INT32, 5, &fill, &src), DTL_SUCCESS);
    ASSERT_EQ(dtl_array_create(ctx, DTL_DTYPE_INT32, 5, &dst), DTL_SUCCESS);

    ASSERT_EQ(dtl_transform_array(src, dst, add_offset_transform, &offset), DTL_SUCCESS);

    const int32_t* data = static_cast<const int32_t*>(dtl_array_local_data(dst));
    dtl_size_t local_size = dtl_array_local_size(dst);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(data[i], 10 + 100 + static_cast<int32_t>(i));
    }

    dtl_array_destroy(src);
    dtl_array_destroy(dst);
}

// ============================================================================
// Copy/Fill Tests
// ============================================================================

TEST_F(AlgorithmTest, CopyVector) {
    dtl_vector_t src, dst;
    double fill = 42.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &src), DTL_SUCCESS);
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 10, &dst), DTL_SUCCESS);

    ASSERT_EQ(dtl_copy_vector(src, dst), DTL_SUCCESS);

    const double* data = static_cast<const double*>(dtl_vector_local_data(dst));
    dtl_size_t local_size = dtl_vector_local_size(dst);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 42.0);
    }

    dtl_vector_destroy(src);
    dtl_vector_destroy(dst);
}

TEST_F(AlgorithmTest, CopyArray) {
    dtl_array_t src, dst;
    int64_t fill = 12345;
    ASSERT_EQ(dtl_array_create_fill(ctx, DTL_DTYPE_INT64, 8, &fill, &src), DTL_SUCCESS);
    ASSERT_EQ(dtl_array_create(ctx, DTL_DTYPE_INT64, 8, &dst), DTL_SUCCESS);

    ASSERT_EQ(dtl_copy_array(src, dst), DTL_SUCCESS);

    const int64_t* data = static_cast<const int64_t*>(dtl_array_local_data(dst));
    dtl_size_t local_size = dtl_array_local_size(dst);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(data[i], 12345);
    }

    dtl_array_destroy(src);
    dtl_array_destroy(dst);
}

TEST_F(AlgorithmTest, CopyVectorTypeMismatch) {
    dtl_vector_t src, dst;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 10, &src), DTL_SUCCESS);
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 10, &dst), DTL_SUCCESS);

    EXPECT_EQ(dtl_copy_vector(src, dst), DTL_ERROR_INVALID_ARGUMENT);

    dtl_vector_destroy(src);
    dtl_vector_destroy(dst);
}

TEST_F(AlgorithmTest, FillVector) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT32, 15, &vec), DTL_SUCCESS);

    float fill = 3.14f;
    ASSERT_EQ(dtl_fill_vector(vec, &fill), DTL_SUCCESS);

    const float* data = static_cast<const float*>(dtl_vector_local_data(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, FillArray) {
    dtl_array_t arr;
    ASSERT_EQ(dtl_array_create(ctx, DTL_DTYPE_INT16, 20, &arr), DTL_SUCCESS);

    int16_t fill = 999;
    ASSERT_EQ(dtl_fill_array(arr, &fill), DTL_SUCCESS);

    const int16_t* data = static_cast<const int16_t*>(dtl_array_local_data(arr));
    dtl_size_t local_size = dtl_array_local_size(arr);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(data[i], 999);
    }

    dtl_array_destroy(arr);
}

// ============================================================================
// Find Tests
// ============================================================================

TEST_F(AlgorithmTest, FindVectorFound) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 10, &vec), DTL_SUCCESS);

    // Fill with sequential values
    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<int32_t>(i * 2);
    }

    // Find value that exists (should find 4 at index 2)
    int32_t target = 4;
    if (local_size > 2) {
        dtl_index_t idx = dtl_find_vector(vec, &target);
        EXPECT_EQ(idx, 2);
    }

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, FindVectorNotFound) {
    dtl_vector_t vec;
    int32_t fill = 0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 10, &fill, &vec), DTL_SUCCESS);

    int32_t target = 999;
    dtl_index_t idx = dtl_find_vector(vec, &target);
    EXPECT_EQ(idx, -1);

    dtl_vector_destroy(vec);
}

namespace {

int is_positive(const void* elem, void* user_data) {
    (void)user_data;
    const int32_t* val = static_cast<const int32_t*>(elem);
    return *val > 0 ? 1 : 0;
}

int is_greater_than(const void* elem, void* user_data) {
    const double* val = static_cast<const double*>(elem);
    const double* threshold = static_cast<const double*>(user_data);
    return *val > *threshold ? 1 : 0;
}

} // namespace

TEST_F(AlgorithmTest, FindIfVector) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 10, &vec), DTL_SUCCESS);

    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    // Fill with negative values, then one positive
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = -static_cast<int32_t>(i + 1);
    }
    if (local_size > 3) {
        data[3] = 42;  // First positive at index 3
    }

    dtl_index_t idx = dtl_find_if_vector(vec, is_positive, nullptr);
    if (local_size > 3) {
        EXPECT_EQ(idx, 3);
    }

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, FindArray) {
    dtl_array_t arr;
    double fill = 1.0;
    ASSERT_EQ(dtl_array_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &arr), DTL_SUCCESS);

    double* data = static_cast<double*>(dtl_array_local_data_mut(arr));
    dtl_size_t local_size = dtl_array_local_size(arr);
    if (local_size > 5) {
        data[5] = 99.0;
    }

    double target = 99.0;
    dtl_index_t idx = dtl_find_array(arr, &target);
    if (local_size > 5) {
        EXPECT_EQ(idx, 5);
    }

    dtl_array_destroy(arr);
}

// ============================================================================
// Count Tests
// ============================================================================

TEST_F(AlgorithmTest, CountVector) {
    dtl_vector_t vec;
    int32_t fill = 5;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 10, &fill, &vec), DTL_SUCCESS);

    // All elements should match
    dtl_size_t count = dtl_count_vector(vec, &fill);
    EXPECT_EQ(count, dtl_vector_local_size(vec));

    // No elements should match
    int32_t other = 99;
    count = dtl_count_vector(vec, &other);
    EXPECT_EQ(count, 0u);

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, CountIfVector) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 10, &vec), DTL_SUCCESS);

    double* data = static_cast<double*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    // Fill with values: 0.5, 1.5, 2.5, 3.5, ...
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = i + 0.5;
    }

    // Count values > 2.0 (should be all values >= 2.5)
    double threshold = 2.0;
    dtl_size_t count = dtl_count_if_vector(vec, is_greater_than, &threshold);
    // Values > 2.0: 2.5, 3.5, 4.5, ... up to (local_size - 1) + 0.5
    dtl_size_t expected = (local_size > 2) ? (local_size - 2) : 0;
    EXPECT_EQ(count, expected);

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, CountArray) {
    dtl_array_t arr;
    uint8_t fill = 0;
    ASSERT_EQ(dtl_array_create_fill(ctx, DTL_DTYPE_UINT8, 20, &fill, &arr), DTL_SUCCESS);

    uint8_t* data = static_cast<uint8_t*>(dtl_array_local_data_mut(arr));
    dtl_size_t local_size = dtl_array_local_size(arr);
    // Set some elements to 1
    for (dtl_size_t i = 0; i < local_size; i += 3) {
        data[i] = 1;
    }

    uint8_t target = 1;
    dtl_size_t count = dtl_count_array(arr, &target);
    dtl_size_t expected = (local_size + 2) / 3;  // Ceiling division
    EXPECT_EQ(count, expected);

    dtl_array_destroy(arr);
}

// ============================================================================
// Reduce Tests
// ============================================================================

TEST_F(AlgorithmTest, ReduceLocalVectorSum) {
    dtl_vector_t vec;
    double fill = 1.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &vec), DTL_SUCCESS);

    double result = 0.0;
    ASSERT_EQ(dtl_reduce_local_vector(vec, DTL_OP_SUM, &result), DTL_SUCCESS);

    dtl_size_t local_size = dtl_vector_local_size(vec);
    EXPECT_DOUBLE_EQ(result, static_cast<double>(local_size));

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, ReduceLocalVectorProd) {
    dtl_vector_t vec;
    double fill = 2.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 5, &fill, &vec), DTL_SUCCESS);

    double result = 0.0;
    ASSERT_EQ(dtl_reduce_local_vector(vec, DTL_OP_PROD, &result), DTL_SUCCESS);

    dtl_size_t local_size = dtl_vector_local_size(vec);
    double expected = std::pow(2.0, static_cast<double>(local_size));
    EXPECT_DOUBLE_EQ(result, expected);

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, ReduceLocalVectorMinMax) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 10, &vec), DTL_SUCCESS);

    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<int32_t>(i * 10 - 50);  // -50, -40, -30, ...
    }

    int32_t min_result = 0;
    int32_t max_result = 0;
    ASSERT_EQ(dtl_reduce_local_vector(vec, DTL_OP_MIN, &min_result), DTL_SUCCESS);
    ASSERT_EQ(dtl_reduce_local_vector(vec, DTL_OP_MAX, &max_result), DTL_SUCCESS);

    EXPECT_EQ(min_result, -50);
    EXPECT_EQ(max_result, static_cast<int32_t>((local_size - 1) * 10 - 50));

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, ReduceLocalArray) {
    dtl_array_t arr;
    float fill = 0.5f;
    ASSERT_EQ(dtl_array_create_fill(ctx, DTL_DTYPE_FLOAT32, 8, &fill, &arr), DTL_SUCCESS);

    float result = 0.0f;
    ASSERT_EQ(dtl_reduce_local_array(arr, DTL_OP_SUM, &result), DTL_SUCCESS);

    dtl_size_t local_size = dtl_array_local_size(arr);
    EXPECT_FLOAT_EQ(result, local_size * 0.5f);

    dtl_array_destroy(arr);
}

namespace {

void custom_sum(const void* a, const void* b, void* result, void* user_data) {
    (void)user_data;
    const double* av = static_cast<const double*>(a);
    const double* bv = static_cast<const double*>(b);
    double* rv = static_cast<double*>(result);
    *rv = *av + *bv;
}

} // namespace

TEST_F(AlgorithmTest, ReduceLocalVectorCustom) {
    dtl_vector_t vec;
    double fill = 2.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 5, &fill, &vec), DTL_SUCCESS);

    double identity = 0.0;
    double result = 0.0;
    ASSERT_EQ(dtl_reduce_local_vector_func(vec, custom_sum, &identity, &result, nullptr), DTL_SUCCESS);

    dtl_size_t local_size = dtl_vector_local_size(vec);
    EXPECT_DOUBLE_EQ(result, local_size * 2.0);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Sort Tests
// ============================================================================

TEST_F(AlgorithmTest, SortVectorAscending) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 10, &vec), DTL_SUCCESS);

    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    // Fill in reverse order
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<int32_t>(local_size - i);
    }

    ASSERT_EQ(dtl_sort_vector(vec), DTL_SUCCESS);

    // Verify sorted ascending
    const int32_t* sorted = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    for (dtl_size_t i = 1; i < local_size; ++i) {
        EXPECT_LE(sorted[i - 1], sorted[i]);
    }

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, SortVectorDescending) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 10, &vec), DTL_SUCCESS);

    double* data = static_cast<double*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<double>(i);
    }

    ASSERT_EQ(dtl_sort_vector_descending(vec), DTL_SUCCESS);

    const double* sorted = static_cast<const double*>(dtl_vector_local_data(vec));
    for (dtl_size_t i = 1; i < local_size; ++i) {
        EXPECT_GE(sorted[i - 1], sorted[i]);
    }

    dtl_vector_destroy(vec);
}

namespace {

int reverse_compare(const void* a, const void* b, void* user_data) {
    (void)user_data;
    const int32_t* av = static_cast<const int32_t*>(a);
    const int32_t* bv = static_cast<const int32_t*>(b);
    return (*bv < *av) ? -1 : ((*bv > *av) ? 1 : 0);
}

} // namespace

TEST_F(AlgorithmTest, SortVectorCustom) {
    dtl_vector_t vec;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 10, &vec), DTL_SUCCESS);

    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<int32_t>(i);
    }

    // Sort descending using custom comparator
    ASSERT_EQ(dtl_sort_vector_func(vec, reverse_compare, nullptr), DTL_SUCCESS);

    const int32_t* sorted = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    for (dtl_size_t i = 1; i < local_size; ++i) {
        EXPECT_GE(sorted[i - 1], sorted[i]);
    }

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, SortArrayAscending) {
    dtl_array_t arr;
    ASSERT_EQ(dtl_array_create(ctx, DTL_DTYPE_INT64, 15, &arr), DTL_SUCCESS);

    int64_t* data = static_cast<int64_t*>(dtl_array_local_data_mut(arr));
    dtl_size_t local_size = dtl_array_local_size(arr);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<int64_t>((local_size - i) * 100);
    }

    ASSERT_EQ(dtl_sort_array(arr), DTL_SUCCESS);

    const int64_t* sorted = static_cast<const int64_t*>(dtl_array_local_data(arr));
    for (dtl_size_t i = 1; i < local_size; ++i) {
        EXPECT_LE(sorted[i - 1], sorted[i]);
    }

    dtl_array_destroy(arr);
}

TEST_F(AlgorithmTest, SortArrayDescending) {
    dtl_array_t arr;
    ASSERT_EQ(dtl_array_create(ctx, DTL_DTYPE_FLOAT32, 12, &arr), DTL_SUCCESS);

    float* data = static_cast<float*>(dtl_array_local_data_mut(arr));
    dtl_size_t local_size = dtl_array_local_size(arr);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<float>(i * 0.1f);
    }

    ASSERT_EQ(dtl_sort_array_descending(arr), DTL_SUCCESS);

    const float* sorted = static_cast<const float*>(dtl_array_local_data(arr));
    for (dtl_size_t i = 1; i < local_size; ++i) {
        EXPECT_GE(sorted[i - 1], sorted[i]);
    }

    dtl_array_destroy(arr);
}

// ============================================================================
// MinMax Tests
// ============================================================================

TEST_F(AlgorithmTest, MinMaxVector) {
    dtl_vector_t vec;
    // Use create_fill to ensure all elements are 100.0
    double fill_val = 100.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill_val, &vec), DTL_SUCCESS);

    dtl_size_t local_size = dtl_vector_local_size(vec);
    ASSERT_GT(local_size, 0u);

    // Verify we can read back the fill values
    double check_val;
    ASSERT_EQ(dtl_vector_get_local(vec, 0, &check_val), DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(check_val, 100.0);  // Should be fill value

    // Set first element to minimum
    double min_expected = -25.0;
    ASSERT_EQ(dtl_vector_set_local(vec, 0, &min_expected), DTL_SUCCESS);

    // Verify it was set
    ASSERT_EQ(dtl_vector_get_local(vec, 0, &check_val), DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(check_val, min_expected);

    // Set second-to-last element to maximum (to avoid edge cases)
    double max_expected = 200.0;  // Larger than fill value
    dtl_size_t max_idx = local_size > 2 ? local_size - 2 : local_size - 1;
    ASSERT_EQ(dtl_vector_set_local(vec, max_idx, &max_expected), DTL_SUCCESS);

    // Verify it was set
    ASSERT_EQ(dtl_vector_get_local(vec, max_idx, &check_val), DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(check_val, max_expected);

    double min_val = 0.0, max_val = 0.0;
    ASSERT_EQ(dtl_minmax_vector(vec, &min_val, &max_val), DTL_SUCCESS);

    EXPECT_DOUBLE_EQ(min_val, min_expected);
    EXPECT_DOUBLE_EQ(max_val, max_expected);

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, MinMaxVectorOnlyMin) {
    dtl_vector_t vec;
    double fill = 42.0;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 10, &fill, &vec), DTL_SUCCESS);

    double min_val = 0.0;
    ASSERT_EQ(dtl_minmax_vector(vec, &min_val, nullptr), DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(min_val, 42.0);

    dtl_vector_destroy(vec);
}

TEST_F(AlgorithmTest, MinMaxArray) {
    dtl_array_t arr;
    ASSERT_EQ(dtl_array_create(ctx, DTL_DTYPE_INT32, 20, &arr), DTL_SUCCESS);

    int32_t* data = static_cast<int32_t*>(dtl_array_local_data_mut(arr));
    dtl_size_t local_size = dtl_array_local_size(arr);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<int32_t>(i * i - 100);
    }

    int32_t min_val = 0, max_val = 0;
    ASSERT_EQ(dtl_minmax_array(arr, &min_val, &max_val), DTL_SUCCESS);

    // Find expected min/max
    int32_t expected_min = data[0], expected_max = data[0];
    for (dtl_size_t i = 1; i < local_size; ++i) {
        if (data[i] < expected_min) expected_min = data[i];
        if (data[i] > expected_max) expected_max = data[i];
    }

    EXPECT_EQ(min_val, expected_min);
    EXPECT_EQ(max_val, expected_max);

    dtl_array_destroy(arr);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(AlgorithmTest, AlgorithmsWithNullVector) {
    EXPECT_EQ(dtl_for_each_vector(nullptr, count_element_mut, nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_transform_vector(nullptr, nullptr, square_transform, nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_copy_vector(nullptr, nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_sort_vector(nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_find_vector(nullptr, nullptr), -1);
    EXPECT_EQ(dtl_count_vector(nullptr, nullptr), 0u);
}

TEST_F(AlgorithmTest, AlgorithmsWithNullArray) {
    EXPECT_EQ(dtl_for_each_array(nullptr, count_element_mut, nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_transform_array(nullptr, nullptr, add_offset_transform, nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_copy_array(nullptr, nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_sort_array(nullptr), DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(dtl_find_array(nullptr, nullptr), -1);
    EXPECT_EQ(dtl_count_array(nullptr, nullptr), 0u);
}
