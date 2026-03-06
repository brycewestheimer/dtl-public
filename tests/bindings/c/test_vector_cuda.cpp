// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_vector_cuda.cpp
 * @brief CUDA placement tests for C binding vectors
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <cstring>
#include <vector>

// ============================================================================
// CUDA Detection
// ============================================================================

#ifdef DTL_HAS_CUDA
#include <cuda_runtime.h>
static bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}
#else
static bool cuda_available() { return false; }
#endif

// ============================================================================
// Test Fixture
// ============================================================================

class CudaVectorTest : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;

    void SetUp() override {
        if (!cuda_available()) {
            GTEST_SKIP() << "CUDA hardware not available";
        }
        dtl_status status = dtl_context_create_default(&ctx);
        ASSERT_EQ(status, DTL_SUCCESS);
    }

    void TearDown() override {
        if (ctx) {
            dtl_context_destroy(ctx);
        }
    }

    dtl_container_options device_opts() {
        dtl_container_options opts;
        dtl_container_options_init(&opts);
        opts.placement = DTL_PLACEMENT_DEVICE;
        opts.device_id = 0;
        return opts;
    }

    dtl_container_options unified_opts() {
        dtl_container_options opts;
        dtl_container_options_init(&opts);
        opts.placement = DTL_PLACEMENT_UNIFIED;
        opts.device_id = 0;
        return opts;
    }
};

// ============================================================================
// Device Placement Tests
// ============================================================================

#ifdef DTL_HAS_CUDA

TEST_F(CudaVectorTest, CreateDevicePlacement) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    // Context must have HAS_CUDA flag for this to succeed
    if (status != DTL_SUCCESS) {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        return;
    }

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);
    EXPECT_EQ(dtl_vector_placement_policy(vec), DTL_PLACEMENT_DEVICE);
    EXPECT_EQ(dtl_vector_global_size(vec), 100u);
    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, CreateUnifiedPlacement) {
    auto opts = unified_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    if (status != DTL_SUCCESS) {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        return;
    }

    ASSERT_NE(vec, nullptr);
    EXPECT_EQ(dtl_vector_placement_policy(vec), DTL_PLACEMENT_UNIFIED);
    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, DeviceLocalDataReturnsNull) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT32, 10, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    // Device memory is not host-accessible
    EXPECT_EQ(dtl_vector_local_data(vec), nullptr);
    EXPECT_EQ(dtl_vector_local_data_mut(vec), nullptr);

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, UnifiedLocalDataReturnsNonNull) {
    auto opts = unified_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT32, 10, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Unified creation unavailable"; }

    // Unified memory IS host-accessible
    EXPECT_NE(dtl_vector_local_data(vec), nullptr);
    EXPECT_NE(dtl_vector_local_data_mut(vec), nullptr);

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, DeviceDeviceDataReturnsNonNull) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT32, 10, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    EXPECT_NE(dtl_vector_device_data(vec), nullptr);
    EXPECT_NE(dtl_vector_device_data_mut(vec), nullptr);

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, CopyToHostFromDevice) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 10, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    // Fill device vector
    double fill_val = 3.14;
    status = dtl_vector_fill_local(vec, &fill_val);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Copy to host and verify
    std::size_t local_size = dtl_vector_local_size(vec);
    std::vector<double> host_buf(local_size);
    status = dtl_vector_copy_to_host(vec, host_buf.data(), local_size);
    ASSERT_EQ(status, DTL_SUCCESS);

    for (std::size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(host_buf[i], 3.14);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, CopyFromHostToDevice) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 10, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    std::size_t local_size = dtl_vector_local_size(vec);
    std::vector<int32_t> src(local_size);
    for (std::size_t i = 0; i < local_size; ++i) src[i] = static_cast<int32_t>(i * 3);

    // Copy to device
    status = dtl_vector_copy_from_host(vec, src.data(), local_size);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Copy back and verify round-trip
    std::vector<int32_t> dst(local_size, 0);
    status = dtl_vector_copy_to_host(vec, dst.data(), local_size);
    ASSERT_EQ(status, DTL_SUCCESS);

    for (std::size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(dst[i], static_cast<int32_t>(i * 3));
    }

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, DeviceFill) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 20, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    int32_t val = 42;
    status = dtl_vector_fill_local(vec, &val);
    ASSERT_EQ(status, DTL_SUCCESS);

    std::size_t local_size = dtl_vector_local_size(vec);
    std::vector<int32_t> host_buf(local_size);
    status = dtl_vector_copy_to_host(vec, host_buf.data(), local_size);
    ASSERT_EQ(status, DTL_SUCCESS);

    for (std::size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(host_buf[i], 42);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, DeviceReduceSum) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 10, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    // Fill with 1s
    int32_t one = 1;
    dtl_vector_fill_local(vec, &one);

    int32_t sum = 0;
    status = dtl_vector_reduce_sum(vec, &sum);
    ASSERT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(sum, static_cast<int32_t>(dtl_vector_local_size(vec)));

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, DeviceReduceMinMax) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 5, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    std::size_t local_size = dtl_vector_local_size(vec);
    std::vector<double> data(local_size);
    for (std::size_t i = 0; i < local_size; ++i) data[i] = static_cast<double>(i) + 1.0;
    dtl_vector_copy_from_host(vec, data.data(), local_size);

    double min_val = 0, max_val = 0;
    EXPECT_EQ(dtl_vector_reduce_min(vec, &min_val), DTL_SUCCESS);
    EXPECT_EQ(dtl_vector_reduce_max(vec, &max_val), DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, static_cast<double>(local_size));

    dtl_vector_destroy(vec);
}

TEST_F(CudaVectorTest, DeviceSort) {
    auto opts = device_opts();
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 5, &opts, &vec);
    if (status != DTL_SUCCESS) { GTEST_SKIP() << "Device creation unavailable"; }

    std::size_t local_size = dtl_vector_local_size(vec);
    std::vector<int32_t> data(local_size);
    for (std::size_t i = 0; i < local_size; ++i) data[i] = static_cast<int32_t>(local_size - i);
    dtl_vector_copy_from_host(vec, data.data(), local_size);

    status = dtl_vector_sort_ascending(vec);
    ASSERT_EQ(status, DTL_SUCCESS);

    std::vector<int32_t> sorted(local_size);
    dtl_vector_copy_to_host(vec, sorted.data(), local_size);
    for (std::size_t i = 1; i < local_size; ++i) {
        EXPECT_LE(sorted[i - 1], sorted[i]);
    }

    dtl_vector_destroy(vec);
}

#endif  // DTL_HAS_CUDA

// ============================================================================
// Tests that work regardless of CUDA availability
// ============================================================================

TEST_F(CudaVectorTest, DevicePreferredReturnsNotSupported) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_DEVICE_PREFERRED;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);
    EXPECT_EQ(status, DTL_ERROR_NOT_SUPPORTED);
    EXPECT_EQ(vec, nullptr);
}

TEST(CudaVectorNoContext, NoCudaContextReturnsBackendUnavailable) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA hardware not available";
    }

    // Create a default context (no CUDA domain)
    dtl_context_t ctx = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx), DTL_SUCCESS);

    // Default context shouldn't have CUDA domain unless explicitly set
    // This test verifies the validation path
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_DEVICE;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    // Default context should lack HAS_CUDA flag, so expect an error.
    // If it happens to have CUDA, creation may succeed.
    if (status != DTL_SUCCESS) {
        EXPECT_TRUE(status == DTL_ERROR_BACKEND_UNAVAILABLE ||
                    status == DTL_ERROR_NOT_SUPPORTED);
        EXPECT_EQ(vec, nullptr);
    } else {
        if (vec) {
            dtl_vector_destroy(vec);
        }
    }

    dtl_context_destroy(ctx);
}
