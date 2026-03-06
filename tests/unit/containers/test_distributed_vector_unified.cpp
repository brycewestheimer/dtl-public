// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_vector_unified.cpp
/// @brief Unit tests for distributed_vector with unified_memory placement

#include <gtest/gtest.h>
#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/policies.hpp>
#include <vector>

namespace {

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct test_unified_context {
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

TEST(DistributedVectorUnified, LocalViewAccessible) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_unified_context ctx;
    dtl::distributed_vector<float, dtl::unified_memory> vec(50, ctx);

    auto lv = vec.local_view();
    EXPECT_NE(lv.data(), nullptr);
    EXPECT_EQ(lv.size(), vec.local_size());

    // Can write directly via host pointer
    for (std::size_t i = 0; i < lv.size(); ++i) {
        lv[i] = static_cast<float>(i);
    }
    EXPECT_FLOAT_EQ(lv[0], 0.0f);
    EXPECT_FLOAT_EQ(lv[1], 1.0f);
}

TEST(DistributedVectorUnified, DeviceViewAccessible) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_unified_context ctx;
    dtl::distributed_vector<int, dtl::unified_memory> vec(20, ctx);

    auto dv = vec.device_view();
    EXPECT_NE(dv.data(), nullptr);
    EXPECT_EQ(dv.size(), vec.local_size());
}

TEST(DistributedVectorUnified, HostAndDeviceShareMemory) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_unified_context ctx;
    dtl::distributed_vector<int, dtl::unified_memory> vec(10, ctx);

    // Write via local view (host)
    auto lv = vec.local_view();
    for (std::size_t i = 0; i < lv.size(); ++i) {
        lv[i] = static_cast<int>(i * 3);
    }

    // Read via device view pointer copy back to host
    auto dv = vec.device_view();
    std::vector<int> check(dv.size());
    cudaDeviceSynchronize();
    cudaMemcpy(check.data(), dv.data(), check.size() * sizeof(int), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < check.size(); ++i) {
        EXPECT_EQ(check[i], static_cast<int>(i * 3));
    }
}

TEST(DistributedVectorUnified, ResizeWorks) {
    if (!has_cuda_device()) { GTEST_SKIP() << "No CUDA device"; }

    test_unified_context ctx;
    dtl::distributed_vector<double, dtl::unified_memory> vec(10, ctx);
    EXPECT_EQ(vec.local_size(), 10u);

    auto result = vec.resize(25);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(vec.local_size(), 25u);
}

#endif  // DTL_ENABLE_CUDA
