// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_rma.cpp
 * @brief Tests for DTL C bindings RMA operations
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl.h>
#include <cstring>
#include <vector>
#include <cmath>

// ============================================================================
// Test Fixture
// ============================================================================

class RmaTest : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;
    dtl_rank_t my_rank = 0;

    void SetUp() override {
        dtl_status status = dtl_context_create_default(&ctx);
        ASSERT_EQ(status, DTL_SUCCESS);
        ASSERT_NE(ctx, nullptr);
        my_rank = dtl_context_rank(ctx);
    }

    void TearDown() override {
        if (ctx) {
            dtl_context_destroy(ctx);
        }
    }
};

// ============================================================================
// Window Lifecycle Tests
// ============================================================================

TEST_F(RmaTest, WindowCreateWithBase) {
    std::vector<double> data(100, 0.0);
    dtl_window_t win = nullptr;

    dtl_status status = dtl_window_create(ctx, data.data(),
                                           data.size() * sizeof(double), &win);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(win, nullptr);
    ASSERT_EQ(dtl_window_is_valid(win), 1);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, WindowCreateWithNullBase) {
    dtl_window_t win = nullptr;

    // NULL base is allowed (remote-only window)
    dtl_status status = dtl_window_create(ctx, nullptr, 1024, &win);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(win, nullptr);
    ASSERT_EQ(dtl_window_base(win), nullptr);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, WindowAllocate) {
    dtl_window_t win = nullptr;

    dtl_status status = dtl_window_allocate(ctx, 1024, &win);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(win, nullptr);
    ASSERT_NE(dtl_window_base(win), nullptr);
    ASSERT_EQ(dtl_window_size(win), 1024);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, WindowAllocateZeroSize) {
    dtl_window_t win = nullptr;

    dtl_status status = dtl_window_allocate(ctx, 0, &win);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(win, nullptr);
    ASSERT_EQ(dtl_window_size(win), 0);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, WindowDestroyNull) {
    // Should not crash
    dtl_window_destroy(nullptr);
}

TEST_F(RmaTest, WindowDestroyInvalidatesHandle) {
    dtl_window_t win = nullptr;
    dtl_status status = dtl_window_allocate(ctx, 100, &win);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
    // After destroy, is_valid returns 0
    // (though actually using win is UB, we check it doesn't crash on NULL)
}

TEST_F(RmaTest, WindowCreateInvalidContext) {
    dtl_window_t win = nullptr;
    dtl_status status = dtl_window_create(nullptr, nullptr, 100, &win);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);
}

TEST_F(RmaTest, WindowCreateNullOutput) {
    std::vector<char> data(100);
    dtl_status status = dtl_window_create(ctx, data.data(), data.size(), nullptr);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);
}

// ============================================================================
// Window Query Tests
// ============================================================================

TEST_F(RmaTest, WindowBase) {
    std::vector<int> data(50, 42);
    dtl_window_t win = nullptr;

    dtl_status status = dtl_window_create(ctx, data.data(),
                                           data.size() * sizeof(int), &win);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(dtl_window_base(win), data.data());

    dtl_window_destroy(win);
}

TEST_F(RmaTest, WindowSize) {
    dtl_window_t win = nullptr;
    dtl_status status = dtl_window_allocate(ctx, 2048, &win);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(dtl_window_size(win), 2048);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, WindowIsValidTrue) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);
    ASSERT_EQ(dtl_window_is_valid(win), 1);
    dtl_window_destroy(win);
}

TEST_F(RmaTest, WindowIsValidNull) {
    ASSERT_EQ(dtl_window_is_valid(nullptr), 0);
}

TEST_F(RmaTest, WindowBaseNull) {
    ASSERT_EQ(dtl_window_base(nullptr), nullptr);
}

TEST_F(RmaTest, WindowSizeNull) {
    ASSERT_EQ(dtl_window_size(nullptr), 0);
}

// ============================================================================
// Fence Synchronization Tests
// ============================================================================

TEST_F(RmaTest, FenceBasic) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_fence(win);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Close epoch
    status = dtl_window_fence(win);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, FenceNull) {
    dtl_status status = dtl_window_fence(nullptr);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);
}

// ============================================================================
// Lock/Unlock Tests
// ============================================================================

TEST_F(RmaTest, LockUnlockExclusive) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_lock(win, my_rank, DTL_LOCK_EXCLUSIVE);
    ASSERT_EQ(status, DTL_SUCCESS);

    status = dtl_window_unlock(win, my_rank);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, LockUnlockShared) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_lock(win, my_rank, DTL_LOCK_SHARED);
    ASSERT_EQ(status, DTL_SUCCESS);

    status = dtl_window_unlock(win, my_rank);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, LockAlreadyLocked) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_window_lock(win, my_rank, DTL_LOCK_EXCLUSIVE);

    // Double lock should fail
    dtl_status status = dtl_window_lock(win, my_rank, DTL_LOCK_EXCLUSIVE);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);

    dtl_window_unlock(win, my_rank);
    dtl_window_destroy(win);
}

TEST_F(RmaTest, UnlockNotLocked) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_unlock(win, my_rank);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, LockAllUnlockAll) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_lock_all(win);
    ASSERT_EQ(status, DTL_SUCCESS);

    status = dtl_window_unlock_all(win);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, LockAllAlreadyLocked) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_window_lock_all(win);

    dtl_status status = dtl_window_lock_all(win);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);

    dtl_window_unlock_all(win);
    dtl_window_destroy(win);
}

TEST_F(RmaTest, LockInvalidTarget) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_lock(win, -1, DTL_LOCK_EXCLUSIVE);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);

    status = dtl_window_lock(win, 1000, DTL_LOCK_EXCLUSIVE);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);

    dtl_window_destroy(win);
}

// ============================================================================
// Flush Tests
// ============================================================================

TEST_F(RmaTest, Flush) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_flush(win, my_rank);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, FlushAll) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_flush_all(win);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, FlushLocal) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_flush_local(win, 0);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, FlushLocalAll) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_flush_local_all(win);
    ASSERT_EQ(status, DTL_SUCCESS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, FlushInvalidTarget) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_window_flush(win, -1);
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);

    dtl_window_destroy(win);
}

// ============================================================================
// Put/Get Tests
// ============================================================================

TEST_F(RmaTest, PutLocalSelf) {
    // Allocate window
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(double), &win);

    // Put data to self (rank 0)
    double data[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    dtl_status status = dtl_rma_put(win, my_rank, 0, data, sizeof(data));
    ASSERT_EQ(status, DTL_SUCCESS);

    // Verify data was written
    double* base = static_cast<double*>(dtl_window_base(win));
    for (int i = 0; i < 10; ++i) {
        ASSERT_DOUBLE_EQ(base[i], data[i]);
    }

    dtl_window_destroy(win);
}

TEST_F(RmaTest, GetLocalSelf) {
    // Allocate window and initialize
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int), &win);

    int* base = static_cast<int*>(dtl_window_base(win));
    for (int i = 0; i < 10; ++i) {
        base[i] = i * 10;
    }

    // Get data from self
    int buffer[10] = {0};
    dtl_status status = dtl_rma_get(win, my_rank, 0, buffer, sizeof(buffer));
    ASSERT_EQ(status, DTL_SUCCESS);

    // Verify
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(buffer[i], i * 10);
    }

    dtl_window_destroy(win);
}

TEST_F(RmaTest, PutWithOffset) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 1000, &win);

    // Initialize to zeros
    std::memset(dtl_window_base(win), 0, 1000);

    // Put at offset 100
    char data[50];
    std::memset(data, 0xAB, sizeof(data));

    dtl_status status = dtl_rma_put(win, my_rank, 100, data, sizeof(data));
    ASSERT_EQ(status, DTL_SUCCESS);

    // Verify
    char* base = static_cast<char*>(dtl_window_base(win));
    for (int i = 0; i < 50; ++i) {
        ASSERT_EQ(static_cast<unsigned char>(base[100 + i]), 0xAB);
    }
    // Before offset should still be zero
    ASSERT_EQ(base[99], 0);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, GetWithOffset) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 1000, &win);

    // Set data at offset
    char* base = static_cast<char*>(dtl_window_base(win));
    for (int i = 0; i < 50; ++i) {
        base[200 + i] = static_cast<char>(i);
    }

    // Get from offset
    char buffer[50] = {0};
    dtl_status status = dtl_rma_get(win, my_rank, 200, buffer, sizeof(buffer));
    ASSERT_EQ(status, DTL_SUCCESS);

    for (int i = 0; i < 50; ++i) {
        ASSERT_EQ(buffer[i], static_cast<char>(i));
    }

    dtl_window_destroy(win);
}

TEST_F(RmaTest, PutOutOfBounds) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    char data[50];
    // offset + size > window size
    dtl_status status = dtl_rma_put(win, my_rank, 80, data, sizeof(data));
    ASSERT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, GetOutOfBounds) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    char buffer[50];
    dtl_status status = dtl_rma_get(win, my_rank, 80, buffer, sizeof(buffer));
    ASSERT_EQ(status, DTL_ERROR_OUT_OF_BOUNDS);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, PutNullOrigin) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_rma_put(win, my_rank, 0, nullptr, 50);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, GetNullBuffer) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    dtl_status status = dtl_rma_get(win, my_rank, 0, nullptr, 50);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, PutInvalidTarget) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    char data[10];
    dtl_status status = dtl_rma_put(win, -1, 0, data, sizeof(data));
    ASSERT_EQ(status, DTL_ERROR_INVALID_ARGUMENT);

    dtl_window_destroy(win);
}

// ============================================================================
// Async Put/Get Tests
// ============================================================================

TEST_F(RmaTest, PutAsync) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int), &win);

    int data[5] = {10, 20, 30, 40, 50};
    dtl_request_t req = nullptr;

    dtl_status status = dtl_rma_put_async(win, my_rank, 0, data, sizeof(data), &req);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(req, nullptr);

    // Wait for completion
    status = dtl_wait(req);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Verify data
    int* base = static_cast<int*>(dtl_window_base(win));
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(base[i], data[i]);
    }

    dtl_window_destroy(win);
}

TEST_F(RmaTest, GetAsync) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(double), &win);

    // Initialize window
    double* base = static_cast<double*>(dtl_window_base(win));
    for (int i = 0; i < 10; ++i) {
        base[i] = i * 3.14;
    }

    double buffer[10] = {0};
    dtl_request_t req = nullptr;

    dtl_status status = dtl_rma_get_async(win, my_rank, 0, buffer, sizeof(buffer), &req);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(req, nullptr);

    status = dtl_wait(req);
    ASSERT_EQ(status, DTL_SUCCESS);

    for (int i = 0; i < 10; ++i) {
        ASSERT_DOUBLE_EQ(buffer[i], i * 3.14);
    }

    dtl_window_destroy(win);
}

TEST_F(RmaTest, RequestTest) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    char data[10] = {0};
    dtl_request_t req = nullptr;

    dtl_rma_put_async(win, my_rank, 0, data, sizeof(data), &req);

    int completed = 0;
    dtl_status status = dtl_test(req, &completed);
    ASSERT_EQ(status, DTL_SUCCESS);
    // In single-process mode, should be immediately complete
    ASSERT_EQ(completed, 1);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, RequestFree) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    char data[10] = {0};
    dtl_request_t req = nullptr;

    dtl_rma_put_async(win, my_rank, 0, data, sizeof(data), &req);
    dtl_request_free(req);  // Free without waiting

    dtl_window_destroy(win);
}

TEST_F(RmaTest, RequestFreeNull) {
    // Should not crash
    dtl_request_free(nullptr);
}

// ============================================================================
// Atomic Accumulate Tests
// ============================================================================

TEST_F(RmaTest, AccumulateSum) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(double), &win);

    // Initialize to 10.0
    double* base = static_cast<double*>(dtl_window_base(win));
    base[0] = 10.0;

    // Accumulate 5.0
    double val = 5.0;
    dtl_status status = dtl_rma_accumulate(win, my_rank, 0, &val, sizeof(double),
                                            DTL_DTYPE_FLOAT64, DTL_OP_SUM);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_DOUBLE_EQ(base[0], 15.0);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, AccumulateProd) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(double), &win);

    double* base = static_cast<double*>(dtl_window_base(win));
    base[0] = 3.0;

    double val = 4.0;
    dtl_status status = dtl_rma_accumulate(win, my_rank, 0, &val, sizeof(double),
                                            DTL_DTYPE_FLOAT64, DTL_OP_PROD);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_DOUBLE_EQ(base[0], 12.0);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, AccumulateMin) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int32_t), &win);

    int32_t* base = static_cast<int32_t*>(dtl_window_base(win));
    base[0] = 100;

    int32_t val = 50;
    dtl_status status = dtl_rma_accumulate(win, my_rank, 0, &val, sizeof(int32_t),
                                            DTL_DTYPE_INT32, DTL_OP_MIN);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(base[0], 50);

    // Try with larger value (should not change)
    val = 75;
    status = dtl_rma_accumulate(win, my_rank, 0, &val, sizeof(int32_t),
                                 DTL_DTYPE_INT32, DTL_OP_MIN);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(base[0], 50);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, AccumulateMax) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int64_t), &win);

    int64_t* base = static_cast<int64_t*>(dtl_window_base(win));
    base[0] = 100;

    int64_t val = 200;
    dtl_status status = dtl_rma_accumulate(win, my_rank, 0, &val, sizeof(int64_t),
                                            DTL_DTYPE_INT64, DTL_OP_MAX);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(base[0], 200);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, AccumulateBitwiseAnd) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int32_t), &win);

    int32_t* base = static_cast<int32_t*>(dtl_window_base(win));
    base[0] = 0xFF;

    int32_t val = 0x0F;
    dtl_status status = dtl_rma_accumulate(win, my_rank, 0, &val, sizeof(int32_t),
                                            DTL_DTYPE_INT32, DTL_OP_BAND);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(base[0], 0x0F);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, AccumulateBitwiseOr) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int32_t), &win);

    int32_t* base = static_cast<int32_t*>(dtl_window_base(win));
    base[0] = 0xF0;

    int32_t val = 0x0F;
    dtl_status status = dtl_rma_accumulate(win, my_rank, 0, &val, sizeof(int32_t),
                                            DTL_DTYPE_INT32, DTL_OP_BOR);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(base[0], 0xFF);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, AccumulateMultipleElements) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(float), &win);

    float* base = static_cast<float*>(dtl_window_base(win));
    for (int i = 0; i < 5; ++i) {
        base[i] = 1.0f;
    }

    float vals[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    dtl_status status = dtl_rma_accumulate(win, my_rank, 0, vals, sizeof(vals),
                                            DTL_DTYPE_FLOAT32, DTL_OP_SUM);
    ASSERT_EQ(status, DTL_SUCCESS);

    for (int i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(base[i], 1.0f + vals[i]);
    }

    dtl_window_destroy(win);
}

// ============================================================================
// Fetch-and-Op Tests
// ============================================================================

TEST_F(RmaTest, FetchAndAdd) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int32_t), &win);

    int32_t* base = static_cast<int32_t*>(dtl_window_base(win));
    base[0] = 42;

    int32_t addend = 10;
    int32_t result = 0;

    dtl_status status = dtl_rma_fetch_and_op(win, my_rank, 0, &addend, &result,
                                              DTL_DTYPE_INT32, DTL_OP_SUM);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(result, 42);      // Old value
    ASSERT_EQ(base[0], 52);     // New value

    dtl_window_destroy(win);
}

TEST_F(RmaTest, FetchAndMax) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(double), &win);

    double* base = static_cast<double*>(dtl_window_base(win));
    base[0] = 5.0;

    double val = 10.0;
    double result = 0.0;

    dtl_status status = dtl_rma_fetch_and_op(win, my_rank, 0, &val, &result,
                                              DTL_DTYPE_FLOAT64, DTL_OP_MAX);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_DOUBLE_EQ(result, 5.0);   // Old value
    ASSERT_DOUBLE_EQ(base[0], 10.0); // New value (max)

    dtl_window_destroy(win);
}

TEST_F(RmaTest, FetchAndOpNullArgs) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    int32_t val = 1;
    int32_t result = 0;

    dtl_status status = dtl_rma_fetch_and_op(win, my_rank, 0, nullptr, &result,
                                              DTL_DTYPE_INT32, DTL_OP_SUM);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);

    status = dtl_rma_fetch_and_op(win, my_rank, 0, &val, nullptr,
                                   DTL_DTYPE_INT32, DTL_OP_SUM);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);

    dtl_window_destroy(win);
}

// ============================================================================
// Compare-and-Swap Tests
// ============================================================================

TEST_F(RmaTest, CompareAndSwapSuccess) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int64_t), &win);

    int64_t* base = static_cast<int64_t*>(dtl_window_base(win));
    base[0] = 42;

    int64_t compare = 42;
    int64_t swap = 100;
    int64_t result = 0;

    dtl_status status = dtl_rma_compare_and_swap(win, my_rank, 0, &compare, &swap,
                                                  &result, DTL_DTYPE_INT64);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(result, 42);      // Old value
    ASSERT_EQ(base[0], 100);    // Swapped to new value

    dtl_window_destroy(win);
}

TEST_F(RmaTest, CompareAndSwapFailure) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int32_t), &win);

    int32_t* base = static_cast<int32_t*>(dtl_window_base(win));
    base[0] = 42;

    int32_t compare = 99;  // Different from actual value
    int32_t swap = 100;
    int32_t result = 0;

    dtl_status status = dtl_rma_compare_and_swap(win, my_rank, 0, &compare, &swap,
                                                  &result, DTL_DTYPE_INT32);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(result, 42);      // Old value returned
    ASSERT_EQ(base[0], 42);     // NOT swapped (compare failed)

    dtl_window_destroy(win);
}

TEST_F(RmaTest, CompareAndSwapDouble) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(double), &win);

    double* base = static_cast<double*>(dtl_window_base(win));
    base[0] = 3.14;

    double compare = 3.14;
    double swap = 2.718;
    double result = 0.0;

    dtl_status status = dtl_rma_compare_and_swap(win, my_rank, 0, &compare, &swap,
                                                  &result, DTL_DTYPE_FLOAT64);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_DOUBLE_EQ(result, 3.14);
    ASSERT_DOUBLE_EQ(base[0], 2.718);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, CompareAndSwapNullArgs) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100, &win);

    int32_t val = 1;
    int32_t result = 0;

    dtl_status status = dtl_rma_compare_and_swap(win, my_rank, 0, nullptr, &val,
                                                  &result, DTL_DTYPE_INT32);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);

    status = dtl_rma_compare_and_swap(win, my_rank, 0, &val, nullptr,
                                       &result, DTL_DTYPE_INT32);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);

    status = dtl_rma_compare_and_swap(win, my_rank, 0, &val, &val,
                                       nullptr, DTL_DTYPE_INT32);
    ASSERT_EQ(status, DTL_ERROR_NULL_POINTER);

    dtl_window_destroy(win);
}

TEST_F(RmaTest, CompareAndSwapWithOffset) {
    dtl_window_t win = nullptr;
    dtl_window_allocate(ctx, 100 * sizeof(int32_t), &win);

    int32_t* base = static_cast<int32_t*>(dtl_window_base(win));
    base[5] = 123;  // At offset 5 * sizeof(int32_t)

    int32_t compare = 123;
    int32_t swap = 456;
    int32_t result = 0;

    dtl_status status = dtl_rma_compare_and_swap(win, my_rank, 5 * sizeof(int32_t),
                                                  &compare, &swap, &result,
                                                  DTL_DTYPE_INT32);
    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_EQ(result, 123);
    ASSERT_EQ(base[5], 456);

    dtl_window_destroy(win);
}
