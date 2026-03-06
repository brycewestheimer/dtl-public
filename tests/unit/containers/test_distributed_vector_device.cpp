// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_vector_device.cpp
/// @brief Unit tests for distributed_vector with device_only_runtime placement

#include <gtest/gtest.h>
#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/policies.hpp>
#include <dtl/algorithms/device/device_fill.hpp>
#include <dtl/algorithms/device/device_reduce.hpp>
#include <dtl/algorithms/device/device_sort.hpp>
#include <vector>

namespace {

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct test_device_context {
    dtl::rank_t rank() const noexcept { return 0; }
    dtl::rank_t size() const noexcept { return 1; }

    template <typename Domain>
    bool has() const noexcept { return true; }

    template <typename Domain>
    auto get() const noexcept {
        struct domain_adapter {
            bool valid() const noexcept { return true; }
            int device_id() const noexcept { return 0; }
        };
        return domain_adapter{};
    }
};

}  // namespace

TEST(DistributedVectorDevice, DeviceViewReturnsNonNullData) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_device_context ctx;
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(100, ctx);

    auto dv = vec.device_view();
    EXPECT_NE(dv.data(), nullptr);
    EXPECT_EQ(dv.size(), vec.local_size());
}

TEST(DistributedVectorDevice, CopyToHostRoundTrip) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_device_context ctx;
    dtl::distributed_vector<int, dtl::device_only_runtime> vec(10, ctx);

    // Write data to device
    auto dv = vec.device_view();
    std::vector<int> src(dv.size());
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<int>(i * 5);
    cudaMemcpy(dv.data(), src.data(), src.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Read back
    std::vector<int> dst(dv.size(), 0);
    cudaMemcpy(dst.data(), dv.data(), dst.size() * sizeof(int), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < dst.size(); ++i) {
        EXPECT_EQ(dst[i], static_cast<int>(i * 5));
    }
}

TEST(DistributedVectorDevice, ResizePreservesDeviceAffinity) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_device_context ctx;
    dtl::distributed_vector<double, dtl::device_only_runtime> vec(10, ctx);
    EXPECT_EQ(vec.local_size(), 10u);

    auto result = vec.resize(20);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(vec.local_size(), 20u);
    EXPECT_NE(vec.device_view().data(), nullptr);
}

TEST(DistributedVectorDevice, DeviceIdMatchesContext) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_device_context ctx;
    dtl::distributed_vector<float, dtl::device_only_runtime> vec(10, ctx);
    EXPECT_EQ(vec.device_id(), 0);
}

#endif  // DTL_ENABLE_CUDA
