// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_hip_algorithms.cpp
/// @brief Compile-only and basic unit tests for dtl::hip:: algorithms
/// @details Verifies HIP algorithm functions compile and link correctly.
///          When HIP is available, runs basic functional tests.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_HIP
#include <dtl/hip/hip_algorithms.hpp>
#include <dtl/hip/device_buffer.hpp>
#include <hip/hip_runtime.h>
#endif

#include <gtest/gtest.h>

#include <vector>

namespace dtl::test {

#if DTL_ENABLE_HIP

class HipAlgorithmsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        hipGetDeviceCount(&count);
        if (count == 0) {
            GTEST_SKIP() << "No HIP device available";
        }
    }
};

TEST_F(HipAlgorithmsTest, FillDevice) {
    constexpr size_t N = 512;
    dtl::hip::device_buffer<float> buf(N);

    dtl::hip::fill_device(buf.data(), N, 3.14f);
    hipDeviceSynchronize();

    std::vector<float> host(N);
    hipMemcpy(host.data(), buf.data(), N * sizeof(float), hipMemcpyDeviceToHost);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(host[i], 3.14f) << "index=" << i;
    }
}

TEST_F(HipAlgorithmsTest, ReduceSumDevice) {
    constexpr size_t N = 256;
    std::vector<float> host(N, 1.0f);

    dtl::hip::device_buffer<float> buf(N);
    hipMemcpy(buf.data(), host.data(), N * sizeof(float), hipMemcpyHostToDevice);

    float sum = dtl::hip::reduce_sum_device(buf.data(), N);
    EXPECT_FLOAT_EQ(sum, static_cast<float>(N));
}

TEST_F(HipAlgorithmsTest, SortDevice) {
    constexpr size_t N = 128;
    std::vector<int> host(N);
    for (size_t i = 0; i < N; ++i) {
        host[i] = static_cast<int>(N - i);
    }

    dtl::hip::device_buffer<int> buf(N);
    hipMemcpy(buf.data(), host.data(), N * sizeof(int), hipMemcpyHostToDevice);

    dtl::hip::sort_device(buf.data(), N);
    hipDeviceSynchronize();

    hipMemcpy(host.data(), buf.data(), N * sizeof(int), hipMemcpyDeviceToHost);

    for (size_t i = 1; i < N; ++i) {
        EXPECT_LE(host[i - 1], host[i]) << "index=" << i;
    }
}

TEST_F(HipAlgorithmsTest, CopyDevice) {
    constexpr size_t N = 256;
    std::vector<double> host(N);
    for (size_t i = 0; i < N; ++i) host[i] = static_cast<double>(i);

    dtl::hip::device_buffer<double> src(N);
    dtl::hip::device_buffer<double> dst(N);
    hipMemcpy(src.data(), host.data(), N * sizeof(double), hipMemcpyHostToDevice);

    dtl::hip::copy_device(src.data(), dst.data(), N);
    hipDeviceSynchronize();

    std::vector<double> result(N, -1.0);
    hipMemcpy(result.data(), dst.data(), N * sizeof(double), hipMemcpyDeviceToHost);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(result[i], static_cast<double>(i)) << "index=" << i;
    }
}

TEST_F(HipAlgorithmsTest, CountDevice) {
    constexpr size_t N = 100;
    std::vector<int> host(N, 0);
    for (size_t i = 0; i < N; i += 3) {
        host[i] = 7;
    }

    dtl::hip::device_buffer<int> buf(N);
    hipMemcpy(buf.data(), host.data(), N * sizeof(int), hipMemcpyHostToDevice);

    auto count = dtl::hip::count_device(buf.data(), N, 7);
    EXPECT_EQ(count, 34u);
}

TEST_F(HipAlgorithmsTest, SynchronizeHelpers) {
    auto stream_result = dtl::hip::synchronize_stream();
    EXPECT_TRUE(stream_result.has_value());

    auto device_result = dtl::hip::synchronize_device();
    EXPECT_TRUE(device_result.has_value());
}

#else  // !DTL_ENABLE_HIP

TEST(HipAlgorithmsTest, FillDevicePlaceholder) {
    SUCCEED();
}

TEST(HipAlgorithmsTest, ReduceSumDevicePlaceholder) {
    SUCCEED();
}

TEST(HipAlgorithmsTest, SortDevicePlaceholder) {
    SUCCEED();
}

TEST(HipAlgorithmsTest, CopyDevicePlaceholder) {
    SUCCEED();
}

TEST(HipAlgorithmsTest, CountDevicePlaceholder) {
    SUCCEED();
}

TEST(HipAlgorithmsTest, SynchronizeHelpersPlaceholder) {
    SUCCEED();
}

#endif  // DTL_ENABLE_HIP

}  // namespace dtl::test
