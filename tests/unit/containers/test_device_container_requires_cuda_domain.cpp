// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_device_container_requires_cuda_domain.cpp
/// @brief Unit tests for device container requirements
/// @details Verifies that device_only_runtime containers require a context
///          with a cuda_domain (or hip_domain) and fail appropriately otherwise.
/// @since 0.1.0

#include <dtl/core/config.hpp>
#include <dtl/dtl.hpp>
#include <dtl/policies/placement/device_only_runtime.hpp>
#include <dtl/core/runtime_device_context.hpp>

#include <gtest/gtest.h>

namespace {

// ============================================================================
// Policy Validation Tests
// ============================================================================

TEST(DeviceContainerRequirements, RuntimePolicyValidatesContextWithCuda) {
#if DTL_ENABLE_CUDA
    auto base_ctx = dtl::make_cpu_context();
    auto cuda_ctx = base_ctx.with_cuda(0);
    
    // Context with CUDA domain should validate
    EXPECT_TRUE(dtl::device_only_runtime::validate_context(cuda_ctx));
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

TEST(DeviceContainerRequirements, RuntimePolicyRejectsCpuOnlyContext) {
    auto cpu_ctx = dtl::make_cpu_context();
    
    // CPU-only context should fail validation
    EXPECT_FALSE(dtl::device_only_runtime::validate_context(cpu_ctx));
}

TEST(DeviceContainerRequirements, ExtractDeviceIdFromCudaContext) {
#if DTL_ENABLE_CUDA
    auto cuda_ctx = dtl::make_cpu_context().with_cuda(0);
    
    int device_id = dtl::device_only_runtime::extract_device_id(cuda_ctx);
    EXPECT_EQ(device_id, 0);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

TEST(DeviceContainerRequirements, ExtractDeviceIdFromCpuContextReturnsNegative) {
    auto cpu_ctx = dtl::make_cpu_context();
    
    int device_id = dtl::device_only_runtime::extract_device_id(cpu_ctx);
    EXPECT_EQ(device_id, -1);
}

// ============================================================================
// Context Utility Tests
// ============================================================================

TEST(ContextUtilities, CtxGpuDeviceIdWithCuda) {
#if DTL_ENABLE_CUDA
    auto cuda_ctx = dtl::make_cpu_context().with_cuda(0);
    
    auto device_id = dtl::detail::ctx_gpu_device_id(cuda_ctx);
    ASSERT_TRUE(device_id.has_value());
    EXPECT_EQ(*device_id, 0);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

TEST(ContextUtilities, CtxGpuDeviceIdWithoutCuda) {
    auto cpu_ctx = dtl::make_cpu_context();
    
    auto device_id = dtl::detail::ctx_gpu_device_id(cpu_ctx);
    EXPECT_FALSE(device_id.has_value());
}

TEST(ContextUtilities, CtxHasGpuDomainWithCuda) {
#if DTL_ENABLE_CUDA
    auto cuda_ctx = dtl::make_cpu_context().with_cuda(0);
    EXPECT_TRUE(dtl::detail::ctx_has_gpu_domain(cuda_ctx));
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

TEST(ContextUtilities, CtxHasGpuDomainWithoutCuda) {
    auto cpu_ctx = dtl::make_cpu_context();
    EXPECT_FALSE(dtl::detail::ctx_has_gpu_domain(cpu_ctx));
}

TEST(ContextUtilities, CtxTryGpuDeviceIdSuccess) {
#if DTL_ENABLE_CUDA
    auto cuda_ctx = dtl::make_cpu_context().with_cuda(0);
    
    auto result = dtl::detail::ctx_try_gpu_device_id(cuda_ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
#else
    GTEST_SKIP() << "CUDA not enabled";
#endif
}

TEST(ContextUtilities, CtxTryGpuDeviceIdFailure) {
    auto cpu_ctx = dtl::make_cpu_context();
    
    auto result = dtl::detail::ctx_try_gpu_device_id(cpu_ctx);
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Type Trait Tests
// ============================================================================

TEST(DevicePolicyTraits, IsRuntimeDevicePolicy) {
    EXPECT_TRUE(dtl::is_runtime_device_policy_v<dtl::device_only_runtime>);
    EXPECT_FALSE(dtl::is_runtime_device_policy_v<dtl::device_only<0>>);
    EXPECT_FALSE(dtl::is_runtime_device_policy_v<dtl::host_only>);
}

TEST(DevicePolicyTraits, IsCompileTimeDevicePolicy) {
    EXPECT_FALSE(dtl::is_compile_time_device_policy_v<dtl::device_only_runtime>);
    EXPECT_TRUE(dtl::is_compile_time_device_policy_v<dtl::device_only<0>>);
    EXPECT_TRUE(dtl::is_compile_time_device_policy_v<dtl::device_only<1>>);
    EXPECT_FALSE(dtl::is_compile_time_device_policy_v<dtl::host_only>);
}

TEST(DevicePolicyTraits, IsDevicePlacementPolicy) {
    EXPECT_TRUE(dtl::is_device_placement_policy_v<dtl::device_only_runtime>);
    EXPECT_TRUE(dtl::is_device_placement_policy_v<dtl::device_only<0>>);
    EXPECT_FALSE(dtl::is_device_placement_policy_v<dtl::host_only>);
}

// ============================================================================
// Placement Policy Interface Tests
// ============================================================================

TEST(DevicePolicyInterface, RuntimePolicyHasCorrectLocation) {
    EXPECT_EQ(dtl::device_only_runtime::preferred_location(), dtl::memory_location::device);
}

TEST(DevicePolicyInterface, RuntimePolicyAccessibility) {
    EXPECT_FALSE(dtl::device_only_runtime::is_host_accessible());
    EXPECT_TRUE(dtl::device_only_runtime::is_device_accessible());
}

TEST(DevicePolicyInterface, RuntimePolicyCopyRequirements) {
    EXPECT_TRUE(dtl::device_only_runtime::requires_host_copy());
    EXPECT_FALSE(dtl::device_only_runtime::requires_device_copy());
}

TEST(DevicePolicyInterface, RuntimePolicyDeviceFlags) {
    EXPECT_FALSE(dtl::device_only_runtime::is_compile_time_device());
    EXPECT_TRUE(dtl::device_only_runtime::is_runtime_device());
}

TEST(DevicePolicyInterface, CompileTimePolicyComparison) {
    // device_only<0> should have compile-time device
    EXPECT_TRUE(dtl::device_only<0>::is_compile_time_device());
    EXPECT_EQ(dtl::device_only<0>::device_id(), 0);
    
    // Different device IDs produce different types
    static_assert(!std::is_same_v<dtl::device_only<0>, dtl::device_only<1>>);
}

}  // namespace
