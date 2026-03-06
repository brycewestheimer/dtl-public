// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_policies.cpp
 * @brief Unit tests for DTL C bindings policy system
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <cstring>

// ============================================================================
// Test Fixture
// ============================================================================

class CBindingsPolicies : public ::testing::Test {
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
};

// ============================================================================
// Partition Policy Tests
// ============================================================================

TEST(PolicyNames, PartitionPolicyNames) {
    EXPECT_STREQ(dtl_partition_policy_name(DTL_PARTITION_BLOCK), "block");
    EXPECT_STREQ(dtl_partition_policy_name(DTL_PARTITION_CYCLIC), "cyclic");
    EXPECT_STREQ(dtl_partition_policy_name(DTL_PARTITION_BLOCK_CYCLIC), "block_cyclic");
    EXPECT_STREQ(dtl_partition_policy_name(DTL_PARTITION_HASH), "hash");
    EXPECT_STREQ(dtl_partition_policy_name(DTL_PARTITION_REPLICATED), "replicated");
    EXPECT_STREQ(dtl_partition_policy_name(static_cast<dtl_partition_policy>(99)), "unknown");
}

// ============================================================================
// Placement Policy Tests
// ============================================================================

TEST(PolicyNames, PlacementPolicyNames) {
    EXPECT_STREQ(dtl_placement_policy_name(DTL_PLACEMENT_HOST), "host");
    EXPECT_STREQ(dtl_placement_policy_name(DTL_PLACEMENT_DEVICE), "device");
    EXPECT_STREQ(dtl_placement_policy_name(DTL_PLACEMENT_UNIFIED), "unified");
    EXPECT_STREQ(dtl_placement_policy_name(DTL_PLACEMENT_DEVICE_PREFERRED), "device_preferred");
    EXPECT_STREQ(dtl_placement_policy_name(static_cast<dtl_placement_policy>(99)), "unknown");
}

TEST(PolicyAvailability, HostAlwaysAvailable) {
    EXPECT_EQ(dtl_placement_available(DTL_PLACEMENT_HOST), 1);
}

TEST(PolicyAvailability, DevicePoliciesConditional) {
    // Availability is a build-time property, not a guarantee that the V2 C
    // container layer has a real implementation for that placement.
    int device_available = dtl_placement_available(DTL_PLACEMENT_DEVICE);
    int unified_available = dtl_placement_available(DTL_PLACEMENT_UNIFIED);
    int device_preferred_available = dtl_placement_available(DTL_PLACEMENT_DEVICE_PREFERRED);

#if DTL_ENABLE_CUDA
    EXPECT_EQ(device_available, 1);
    EXPECT_EQ(unified_available, 1);
#ifdef DTL_C_ABI_ENABLE_DEVICE_PREFERRED
    EXPECT_EQ(device_preferred_available, 1);
#else
    EXPECT_EQ(device_preferred_available, 0);
#endif
#else
    EXPECT_EQ(device_available, 0);
    EXPECT_EQ(unified_available, 0);
    EXPECT_EQ(device_preferred_available, 0);
#endif
}

TEST(PolicyAvailability, InvalidPolicyNotAvailable) {
    EXPECT_EQ(dtl_placement_available(static_cast<dtl_placement_policy>(99)), 0);
}

// ============================================================================
// Execution Policy Tests
// ============================================================================

TEST(PolicyNames, ExecutionPolicyNames) {
    EXPECT_STREQ(dtl_execution_policy_name(DTL_EXEC_SEQ), "seq");
    EXPECT_STREQ(dtl_execution_policy_name(DTL_EXEC_PAR), "par");
    EXPECT_STREQ(dtl_execution_policy_name(DTL_EXEC_ASYNC), "async");
    EXPECT_STREQ(dtl_execution_policy_name(static_cast<dtl_execution_policy>(99)), "unknown");
}

// ============================================================================
// Container Options Tests
// ============================================================================

TEST(ContainerOptions, InitSetsDefaults) {
    dtl_container_options opts;
    // Fill with garbage first
    std::memset(&opts, 0xFF, sizeof(opts));

    dtl_container_options_init(&opts);

    EXPECT_EQ(opts.partition, DTL_PARTITION_BLOCK);
    EXPECT_EQ(opts.placement, DTL_PLACEMENT_HOST);
    EXPECT_EQ(opts.execution, DTL_EXEC_SEQ);
    EXPECT_EQ(opts.device_id, 0);
    EXPECT_EQ(opts.block_size, 1u);
}

TEST(ContainerOptions, InitHandlesNull) {
    // Should not crash
    dtl_container_options_init(nullptr);
}

// ============================================================================
// Policy-Aware Vector Creation Tests
// ============================================================================

TEST_F(CBindingsPolicies, VectorCreateWithNullOptionsSucceeds) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, nullptr, &vec);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(vec, nullptr);
    EXPECT_EQ(dtl_vector_global_size(vec), 100u);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsPolicies, VectorCreateWithDefaultOptions) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(vec, nullptr);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsPolicies, VectorCreateWithCyclicPartition) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_CYCLIC;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(vec, nullptr);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsPolicies, VectorCreateWithUnavailablePlacementNoSilentFallback) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_DEVICE;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    if (status == DTL_SUCCESS) {
        ASSERT_NE(vec, nullptr);
        EXPECT_EQ(dtl_vector_placement_policy(vec), DTL_PLACEMENT_DEVICE);
        dtl_vector_destroy(vec);
    } else {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        EXPECT_EQ(vec, nullptr);
    }
}

TEST_F(CBindingsPolicies, VectorCreateWithUnifiedPlacementNoSilentFallback) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_UNIFIED;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    if (status == DTL_SUCCESS) {
        ASSERT_NE(vec, nullptr);
        EXPECT_EQ(dtl_vector_placement_policy(vec), DTL_PLACEMENT_UNIFIED);
        dtl_vector_destroy(vec);
    } else {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        EXPECT_EQ(vec, nullptr);
    }
}

// ============================================================================
// Policy-Aware Array Creation Tests
// ============================================================================

TEST_F(CBindingsPolicies, ArrayCreateWithNullOptionsSucceeds) {
    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, nullptr, &arr);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(arr, nullptr);
    EXPECT_EQ(dtl_array_global_size(arr), 100u);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsPolicies, ArrayCreateWithDefaultOptions) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &arr);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(arr, nullptr);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsPolicies, ArrayCreateWithBlockCyclicPartition) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_BLOCK_CYCLIC;
    opts.block_size = 16;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &arr);

    if (status == DTL_ERROR_NOT_SUPPORTED) {
        GTEST_SKIP() << "Block-cyclic partition not enabled in this build";
    }

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_NE(arr, nullptr);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsPolicies, ArrayCreateWithDevicePlacementNoSilentFallback) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_DEVICE;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &arr);

    if (status == DTL_SUCCESS) {
        ASSERT_NE(arr, nullptr);
        EXPECT_EQ(dtl_array_placement_policy(arr), DTL_PLACEMENT_DEVICE);
        dtl_array_destroy(arr);
    } else {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        EXPECT_EQ(arr, nullptr);
    }
}

TEST_F(CBindingsPolicies, ArrayCreateWithUnifiedPlacementNoSilentFallback) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_UNIFIED;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &arr);

    if (status == DTL_SUCCESS) {
        ASSERT_NE(arr, nullptr);
        EXPECT_EQ(dtl_array_placement_policy(arr), DTL_PLACEMENT_UNIFIED);
        dtl_array_destroy(arr);
    } else {
        EXPECT_TRUE(status == DTL_ERROR_NOT_SUPPORTED ||
                    status == DTL_ERROR_BACKEND_UNAVAILABLE);
        EXPECT_EQ(arr, nullptr);
    }
}

// ============================================================================
// Policy Query Tests
// ============================================================================

TEST_F(CBindingsPolicies, VectorQueryPartitionPolicy) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 100, &vec);

    dtl_partition_policy policy = dtl_vector_partition_policy(vec);
    EXPECT_EQ(policy, DTL_PARTITION_BLOCK);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsPolicies, VectorQueryPlacementPolicy) {
    dtl_vector_t vec = nullptr;
    dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 100, &vec);

    dtl_placement_policy policy = dtl_vector_placement_policy(vec);
    EXPECT_EQ(policy, DTL_PLACEMENT_HOST);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsPolicies, VectorQueryPolicyOnNullReturnsError) {
    EXPECT_EQ(dtl_vector_partition_policy(nullptr), static_cast<dtl_partition_policy>(-1));
    EXPECT_EQ(dtl_vector_placement_policy(nullptr), static_cast<dtl_placement_policy>(-1));
}

TEST_F(CBindingsPolicies, ArrayQueryPartitionPolicy) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_FLOAT64, 100, &arr);

    dtl_partition_policy policy = dtl_array_partition_policy(arr);
    EXPECT_EQ(policy, DTL_PARTITION_BLOCK);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsPolicies, ArrayQueryPlacementPolicy) {
    dtl_array_t arr = nullptr;
    dtl_array_create(ctx, DTL_DTYPE_FLOAT64, 100, &arr);

    dtl_placement_policy policy = dtl_array_placement_policy(arr);
    EXPECT_EQ(policy, DTL_PLACEMENT_HOST);

    dtl_array_destroy(arr);
}

TEST_F(CBindingsPolicies, ArrayQueryPolicyOnNullReturnsError) {
    EXPECT_EQ(dtl_array_partition_policy(nullptr), static_cast<dtl_partition_policy>(-1));
    EXPECT_EQ(dtl_array_placement_policy(nullptr), static_cast<dtl_placement_policy>(-1));
}
