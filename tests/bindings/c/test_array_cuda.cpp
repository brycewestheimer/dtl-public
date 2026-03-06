// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_array_cuda.cpp
 * @brief CUDA placement tests for C binding arrays
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_array.h>
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

class CudaArrayTest : public ::testing::Test {
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

#ifdef DTL_HAS_CUDA

TEST_F(CudaArrayTest, CreateDevicePlacement) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_DEVICE;
    opts.device_id = 0;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(ctx, DTL_DTYPE_FLOAT32, 50, &opts, &arr);

    if (status != DTL_SUCCESS) {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        return;
    }

    ASSERT_NE(arr, nullptr);
    EXPECT_EQ(dtl_array_placement_policy(arr), DTL_PLACEMENT_DEVICE);

    // Device array: local_data should be null
    EXPECT_EQ(dtl_array_local_data(arr), nullptr);

    dtl_array_destroy(arr);
}

TEST_F(CudaArrayTest, CreateUnifiedPlacement) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_UNIFIED;
    opts.device_id = 0;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(ctx, DTL_DTYPE_FLOAT32, 50, &opts, &arr);

    if (status != DTL_SUCCESS) {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        return;
    }

    ASSERT_NE(arr, nullptr);

    // Unified: both host and device accessible
    EXPECT_NE(dtl_array_local_data(arr), nullptr);

    dtl_array_destroy(arr);
}

TEST_F(CudaArrayTest, DeviceCopyRoundTrip) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_DEVICE;
    opts.device_id = 0;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(ctx, DTL_DTYPE_INT32, 10, &opts, &arr);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    std::size_t local_size = dtl_array_local_size(arr);
    std::vector<int32_t> src(local_size);
    for (std::size_t i = 0; i < local_size; ++i) src[i] = static_cast<int32_t>(i * 7);

    ASSERT_EQ(dtl_array_copy_from_host(arr, src.data(), local_size), DTL_SUCCESS);

    std::vector<int32_t> dst(local_size, 0);
    ASSERT_EQ(dtl_array_copy_to_host(arr, dst.data(), local_size), DTL_SUCCESS);

    for (std::size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(dst[i], static_cast<int32_t>(i * 7));
    }

    dtl_array_destroy(arr);
}

TEST_F(CudaArrayTest, FixedSizeSemantics) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_DEVICE;
    opts.device_id = 0;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(ctx, DTL_DTYPE_FLOAT64, 10, &opts, &arr);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    EXPECT_EQ(dtl_array_global_size(arr), 10u);

    dtl_array_destroy(arr);
}

#endif  // DTL_HAS_CUDA
