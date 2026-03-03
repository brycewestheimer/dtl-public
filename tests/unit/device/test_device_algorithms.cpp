// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_algorithms.cpp
/// @brief Unit tests for dtl::device:: algorithm functions
/// @details Tests the vendor-agnostic dtl::device:: namespace. Runs with
///          either CUDA or HIP backend depending on build configuration.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA || DTL_ENABLE_HIP

#include <dtl/device/algorithms.hpp>
#include <dtl/device/buffer.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#define GPU_MEMCPY(dst, src, size, kind) cudaMemcpy(dst, src, size, kind)
#define GPU_MEMCPY_H2D cudaMemcpyHostToDevice
#define GPU_MEMCPY_D2H cudaMemcpyDeviceToHost
#define GPU_DEVICE_COUNT(count) cudaGetDeviceCount(count)
#elif DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#define GPU_MEMCPY(dst, src, size, kind) hipMemcpy(dst, src, size, kind)
#define GPU_MEMCPY_H2D hipMemcpyHostToDevice
#define GPU_MEMCPY_D2H hipMemcpyDeviceToHost
#define GPU_DEVICE_COUNT(count) hipGetDeviceCount(count)
#endif

#include <gtest/gtest.h>

#include <vector>
#include <numeric>

namespace dtl::test {

class DeviceAlgorithms : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        GPU_DEVICE_COUNT(&count);
        if (count == 0) {
            GTEST_SKIP() << "No GPU available";
        }
    }
};

TEST_F(DeviceAlgorithms, FillDevice) {
    constexpr size_t N = 1024;
    dtl::device::device_buffer<float> buf(N);

    dtl::device::fill_device(buf.data(), N, 42.0f);
    dtl::device::synchronize_device();

    std::vector<float> host(N);
    GPU_MEMCPY(host.data(), buf.data(), N * sizeof(float), GPU_MEMCPY_D2H);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host[i], 42.0f) << "index=" << i;
    }
}

TEST_F(DeviceAlgorithms, ReduceSumDevice) {
    constexpr size_t N = 256;
    std::vector<float> host(N, 1.0f);

    dtl::device::device_buffer<float> buf(N);
    GPU_MEMCPY(buf.data(), host.data(), N * sizeof(float), GPU_MEMCPY_H2D);

    float sum = dtl::device::reduce_sum_device(buf.data(), N);
    EXPECT_FLOAT_EQ(sum, static_cast<float>(N));
}

TEST_F(DeviceAlgorithms, SortDevice) {
    constexpr size_t N = 128;
    std::vector<int> host(N);
    for (size_t i = 0; i < N; ++i) {
        host[i] = static_cast<int>(N - i);
    }

    dtl::device::device_buffer<int> buf(N);
    GPU_MEMCPY(buf.data(), host.data(), N * sizeof(int), GPU_MEMCPY_H2D);

    dtl::device::sort_device(buf.data(), N);
    dtl::device::synchronize_device();

    GPU_MEMCPY(host.data(), buf.data(), N * sizeof(int), GPU_MEMCPY_D2H);

    for (size_t i = 1; i < N; ++i) {
        EXPECT_LE(host[i - 1], host[i]) << "index=" << i;
    }
}

TEST_F(DeviceAlgorithms, CopyDevice) {
    constexpr size_t N = 512;
    std::vector<double> host(N);
    std::iota(host.begin(), host.end(), 0.0);

    dtl::device::device_buffer<double> src(N);
    dtl::device::device_buffer<double> dst(N);
    GPU_MEMCPY(src.data(), host.data(), N * sizeof(double), GPU_MEMCPY_H2D);

    dtl::device::copy_device(src.data(), dst.data(), N);
    dtl::device::synchronize_device();

    std::vector<double> result(N, -1.0);
    GPU_MEMCPY(result.data(), dst.data(), N * sizeof(double), GPU_MEMCPY_D2H);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(result[i], static_cast<double>(i)) << "index=" << i;
    }
}

TEST_F(DeviceAlgorithms, CountDevice) {
    constexpr size_t N = 100;
    std::vector<int> host(N, 0);
    for (size_t i = 0; i < N; i += 3) {
        host[i] = 7;
    }

    dtl::device::device_buffer<int> buf(N);
    GPU_MEMCPY(buf.data(), host.data(), N * sizeof(int), GPU_MEMCPY_H2D);

    auto count = dtl::device::count_device(buf.data(), N, 7);
    EXPECT_EQ(count, 34u);
}

}  // namespace dtl::test

#else  // !DTL_ENABLE_CUDA && !DTL_ENABLE_HIP

#include <gtest/gtest.h>

TEST(DeviceAlgorithms, SkippedNoGpu) {
    GTEST_SKIP() << "Neither CUDA nor HIP enabled — skipping device algorithm tests";
}

#endif  // DTL_ENABLE_CUDA || DTL_ENABLE_HIP
