// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_span.cpp
 * @brief Unit tests for DTL C bindings distributed span
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_span.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_tensor.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_types.h>

#include <vector>

class CBindingsSpan : public ::testing::Test {
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

    dtl_rank_t rank() const { return dtl_context_rank(ctx); }
    dtl_rank_t size() const { return dtl_context_size(ctx); }
};

TEST_F(CBindingsSpan, CreateFromRawSucceeds) {
    std::vector<int32_t> data{1, 2, 3, 4};
    dtl_span_t span = nullptr;

    dtl_status status = dtl_span_create(
        DTL_DTYPE_INT32,
        data.data(),
        static_cast<dtl_size_t>(data.size()),
        16,
        rank(),
        size(),
        &span);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(span, nullptr);
    EXPECT_EQ(dtl_span_size(span), 16u);
    EXPECT_EQ(dtl_span_local_size(span), data.size());
    EXPECT_EQ(dtl_span_dtype(span), DTL_DTYPE_INT32);
    EXPECT_EQ(dtl_span_rank(span), rank());
    EXPECT_EQ(dtl_span_num_ranks(span), size());

    dtl_span_destroy(span);
}

TEST_F(CBindingsSpan, CreateFromRawRejectsNullOutHandle) {
    std::vector<int32_t> data{1, 2, 3, 4};
    dtl_status status = dtl_span_create(
        DTL_DTYPE_INT32,
        data.data(),
        static_cast<dtl_size_t>(data.size()),
        4,
        rank(),
        size(),
        nullptr);
    EXPECT_EQ(status, DTL_ERROR_NULL_POINTER);
}

TEST_F(CBindingsSpan, CreateFromVectorSucceeds) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 64, &vec), DTL_SUCCESS);

    int32_t* local = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    ASSERT_NE(local, nullptr);
    for (dtl_size_t i = 0; i < dtl_vector_local_size(vec); ++i) {
        local[i] = static_cast<int32_t>(i + 10);
    }

    dtl_span_t span = nullptr;
    ASSERT_EQ(dtl_span_from_vector(vec, &span), DTL_SUCCESS);
    ASSERT_EQ(dtl_span_size(span), dtl_vector_global_size(vec));
    ASSERT_EQ(dtl_span_local_size(span), dtl_vector_local_size(vec));
    ASSERT_EQ(dtl_span_dtype(span), dtl_vector_dtype(vec));

    int32_t value = 0;
    ASSERT_EQ(dtl_span_get_local(span, 0, &value), DTL_SUCCESS);
    EXPECT_EQ(value, 10);

    value = 77;
    ASSERT_EQ(dtl_span_set_local(span, 1, &value), DTL_SUCCESS);
    EXPECT_EQ(local[1], 77);

    dtl_span_destroy(span);
    dtl_vector_destroy(vec);
}

TEST_F(CBindingsSpan, CreateFromArraySucceeds) {
    dtl_array_t arr = nullptr;
    ASSERT_EQ(dtl_array_create(ctx, DTL_DTYPE_FLOAT64, 32, &arr), DTL_SUCCESS);

    dtl_span_t span = nullptr;
    ASSERT_EQ(dtl_span_from_array(arr, &span), DTL_SUCCESS);
    EXPECT_EQ(dtl_span_size(span), dtl_array_global_size(arr));
    EXPECT_EQ(dtl_span_local_size(span), dtl_array_local_size(arr));
    EXPECT_EQ(dtl_span_dtype(span), dtl_array_dtype(arr));

    dtl_span_destroy(span);
    dtl_array_destroy(arr);
}

TEST_F(CBindingsSpan, CreateFromTensorSucceeds) {
    dtl_tensor_t tensor = nullptr;
    dtl_shape shape = dtl_shape_2d(8, 4);
    ASSERT_EQ(dtl_tensor_create(ctx, DTL_DTYPE_FLOAT32, shape, &tensor), DTL_SUCCESS);

    dtl_span_t span = nullptr;
    ASSERT_EQ(dtl_span_from_tensor(tensor, &span), DTL_SUCCESS);
    EXPECT_EQ(dtl_span_size(span), dtl_tensor_global_size(tensor));
    EXPECT_EQ(dtl_span_local_size(span), dtl_tensor_local_size(tensor));
    EXPECT_EQ(dtl_span_dtype(span), dtl_tensor_dtype(tensor));

    dtl_span_destroy(span);
    dtl_tensor_destroy(tensor);
}

TEST_F(CBindingsSpan, SizeBytesMatchesDtype) {
    std::vector<double> data(7, 0.0);
    dtl_span_t span = nullptr;
    ASSERT_EQ(dtl_span_create(
                  DTL_DTYPE_FLOAT64,
                  data.data(),
                  static_cast<dtl_size_t>(data.size()),
                  7,
                  rank(),
                  size(),
                  &span),
              DTL_SUCCESS);

    EXPECT_EQ(dtl_span_size_bytes(span), data.size() * sizeof(double));
    dtl_span_destroy(span);
}

TEST_F(CBindingsSpan, FirstLastSubspanCreateViews) {
    std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
    dtl_span_t span = nullptr;
    ASSERT_EQ(dtl_span_create(
                  DTL_DTYPE_INT32,
                  data.data(),
                  static_cast<dtl_size_t>(data.size()),
                  data.size(),
                  rank(),
                  size(),
                  &span),
              DTL_SUCCESS);

    dtl_span_t first = nullptr;
    ASSERT_EQ(dtl_span_first(span, 2, &first), DTL_SUCCESS);
    EXPECT_EQ(dtl_span_local_size(first), 2u);
    int32_t v = 0;
    ASSERT_EQ(dtl_span_get_local(first, 1, &v), DTL_SUCCESS);
    EXPECT_EQ(v, 2);

    dtl_span_t last = nullptr;
    ASSERT_EQ(dtl_span_last(span, 2, &last), DTL_SUCCESS);
    ASSERT_EQ(dtl_span_get_local(last, 0, &v), DTL_SUCCESS);
    EXPECT_EQ(v, 5);

    dtl_span_t sub = nullptr;
    ASSERT_EQ(dtl_span_subspan(span, 2, 3, &sub), DTL_SUCCESS);
    ASSERT_EQ(dtl_span_get_local(sub, 2, &v), DTL_SUCCESS);
    EXPECT_EQ(v, 5);

    dtl_span_t tail = nullptr;
    ASSERT_EQ(dtl_span_subspan(span, 4, DTL_SPAN_NPOS, &tail), DTL_SUCCESS);
    EXPECT_EQ(dtl_span_local_size(tail), 2u);
    ASSERT_EQ(dtl_span_get_local(tail, 1, &v), DTL_SUCCESS);
    EXPECT_EQ(v, 6);

    dtl_span_destroy(tail);
    dtl_span_destroy(sub);
    dtl_span_destroy(last);
    dtl_span_destroy(first);
    dtl_span_destroy(span);
}

TEST_F(CBindingsSpan, OutOfBoundsIsReported) {
    std::vector<int32_t> data{1, 2, 3};
    dtl_span_t span = nullptr;
    ASSERT_EQ(dtl_span_create(
                  DTL_DTYPE_INT32,
                  data.data(),
                  static_cast<dtl_size_t>(data.size()),
                  data.size(),
                  rank(),
                  size(),
                  &span),
              DTL_SUCCESS);

    int32_t out = 0;
    EXPECT_EQ(dtl_span_get_local(span, 3, &out), DTL_ERROR_OUT_OF_BOUNDS);
    EXPECT_EQ(dtl_span_set_local(span, 3, &out), DTL_ERROR_OUT_OF_BOUNDS);

    dtl_span_t bad = nullptr;
    EXPECT_EQ(dtl_span_subspan(span, 4, 1, &bad), DTL_ERROR_OUT_OF_BOUNDS);

    dtl_span_destroy(span);
}

TEST_F(CBindingsSpan, EmptyAndValidity) {
    dtl_span_t span = nullptr;
    ASSERT_EQ(dtl_span_create(
                  DTL_DTYPE_INT8,
                  nullptr,
                  0,
                  0,
                  rank(),
                  size(),
                  &span),
              DTL_SUCCESS);
    EXPECT_EQ(dtl_span_is_valid(span), 1);
    EXPECT_EQ(dtl_span_empty(span), 1);
    dtl_span_destroy(span);
    EXPECT_EQ(dtl_span_is_valid(nullptr), 0);
}

TEST_F(CBindingsSpan, DestroyNullIsSafe) {
    dtl_span_destroy(nullptr);
}
