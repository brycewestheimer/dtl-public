// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_policies_dispatch.cpp
 * @brief Unit tests for DTL C bindings policy dispatch (Phase 04)
 * @since 0.1.0
 *
 * These tests validate that policy options are actually honored
 * and not silently ignored.
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_vector.h>
#include <dtl/bindings/c/dtl_array.h>
#include <dtl/bindings/c/dtl_algorithms.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Test Fixture
// ============================================================================

class PolicyDispatchTest : public ::testing::Test {
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
// Consistency Policy Name Tests
// ============================================================================

TEST(PolicyNames, ConsistencyPolicyNames) {
    EXPECT_STREQ(dtl_consistency_policy_name(DTL_CONSISTENCY_BULK_SYNCHRONOUS), "bulk_synchronous");
    EXPECT_STREQ(dtl_consistency_policy_name(DTL_CONSISTENCY_RELAXED), "relaxed");
    EXPECT_STREQ(dtl_consistency_policy_name(DTL_CONSISTENCY_RELEASE_ACQUIRE), "release_acquire");
    EXPECT_STREQ(dtl_consistency_policy_name(DTL_CONSISTENCY_SEQUENTIAL), "sequential");
    EXPECT_STREQ(dtl_consistency_policy_name(static_cast<dtl_consistency_policy>(99)), "unknown");
}

// ============================================================================
// Error Policy Name Tests
// ============================================================================

TEST(PolicyNames, ErrorPolicyNames) {
    EXPECT_STREQ(dtl_error_policy_name(DTL_ERROR_POLICY_RETURN_STATUS), "return_status");
    EXPECT_STREQ(dtl_error_policy_name(DTL_ERROR_POLICY_CALLBACK), "callback");
    EXPECT_STREQ(dtl_error_policy_name(DTL_ERROR_POLICY_TERMINATE), "terminate");
    EXPECT_STREQ(dtl_error_policy_name(static_cast<dtl_error_policy>(99)), "unknown");
}

// ============================================================================
// Container Options Extended Tests
// ============================================================================

TEST(ContainerOptions, ConsistencyAndErrorDefaults) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    // Check that consistency defaults to bulk_synchronous
    EXPECT_EQ(dtl_container_options_consistency(&opts), DTL_CONSISTENCY_BULK_SYNCHRONOUS);

    // Check that error defaults to return_status
    EXPECT_EQ(dtl_container_options_error(&opts), DTL_ERROR_POLICY_RETURN_STATUS);
}

TEST(ContainerOptions, SetAndGetConsistency) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    dtl_container_options_set_consistency(&opts, DTL_CONSISTENCY_RELAXED);
    EXPECT_EQ(dtl_container_options_consistency(&opts), DTL_CONSISTENCY_RELAXED);

    dtl_container_options_set_consistency(&opts, DTL_CONSISTENCY_SEQUENTIAL);
    EXPECT_EQ(dtl_container_options_consistency(&opts), DTL_CONSISTENCY_SEQUENTIAL);
}

TEST(ContainerOptions, SetAndGetError) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    dtl_container_options_set_error(&opts, DTL_ERROR_POLICY_CALLBACK);
    EXPECT_EQ(dtl_container_options_error(&opts), DTL_ERROR_POLICY_CALLBACK);

    dtl_container_options_set_error(&opts, DTL_ERROR_POLICY_TERMINATE);
    EXPECT_EQ(dtl_container_options_error(&opts), DTL_ERROR_POLICY_TERMINATE);
}

// ============================================================================
// Vector Partition Policy Query Tests
// ============================================================================

TEST_F(PolicyDispatchTest, VectorBlockPartitionQueryReturnsBlock) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_BLOCK;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    // Query should return actual partition, not hardcoded default
    EXPECT_EQ(dtl_vector_partition_policy(vec), DTL_PARTITION_BLOCK);

    dtl_vector_destroy(vec);
}

TEST_F(PolicyDispatchTest, VectorCyclicPartitionQueryReturnsCyclic) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_CYCLIC;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    // Query should return CYCLIC, not default BLOCK
    EXPECT_EQ(dtl_vector_partition_policy(vec), DTL_PARTITION_CYCLIC);

    dtl_vector_destroy(vec);
}

TEST_F(PolicyDispatchTest, VectorReplicatedPartitionQueryReturnsReplicated) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_REPLICATED;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    EXPECT_EQ(dtl_vector_partition_policy(vec), DTL_PARTITION_REPLICATED);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Vector Placement Policy Query Tests
// ============================================================================

TEST_F(PolicyDispatchTest, VectorHostPlacementQueryReturnsHost) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_HOST;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    EXPECT_EQ(dtl_vector_placement_policy(vec), DTL_PLACEMENT_HOST);

    dtl_vector_destroy(vec);
}

TEST_F(PolicyDispatchTest, VectorDevicePlacementNoSilentFallback) {
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

TEST_F(PolicyDispatchTest, VectorUnifiedPlacementNoSilentFallback) {
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

TEST_F(PolicyDispatchTest, ArrayUnifiedPlacementNoSilentFallback) {
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
// Vector Execution Policy Query Tests
// ============================================================================

TEST_F(PolicyDispatchTest, VectorExecutionPolicyQuery) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.execution = DTL_EXEC_PAR;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    EXPECT_EQ(dtl_vector_execution_policy(vec), DTL_EXEC_PAR);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Vector Device ID Query Tests
// ============================================================================

TEST_F(PolicyDispatchTest, VectorDeviceIdQuery) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.device_id = 2;  // Non-default device ID

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    EXPECT_EQ(dtl_vector_device_id(vec), 2);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Vector Consistency/Error Policy Query Tests
// ============================================================================

TEST_F(PolicyDispatchTest, VectorConsistencyPolicyQueryDefault) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 100, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    // Default should be bulk_synchronous
    EXPECT_EQ(dtl_vector_consistency_policy(vec), DTL_CONSISTENCY_BULK_SYNCHRONOUS);

    dtl_vector_destroy(vec);
}

TEST_F(PolicyDispatchTest, VectorErrorPolicyQueryDefault) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 100, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    // Default should be return_status
    EXPECT_EQ(dtl_vector_error_policy(vec), DTL_ERROR_POLICY_RETURN_STATUS);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Array Policy Query Tests
// ============================================================================

TEST_F(PolicyDispatchTest, ArrayPartitionPolicyQuery) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_CYCLIC;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &arr);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(arr, nullptr);

    EXPECT_EQ(dtl_array_partition_policy(arr), DTL_PARTITION_CYCLIC);

    dtl_array_destroy(arr);
}

TEST_F(PolicyDispatchTest, ArrayPlacementPolicyQuery) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.placement = DTL_PLACEMENT_HOST;

    dtl_array_t arr = nullptr;
    dtl_status status = dtl_array_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &arr);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(arr, nullptr);

    EXPECT_EQ(dtl_array_placement_policy(arr), DTL_PLACEMENT_HOST);

    dtl_array_destroy(arr);
}

// ============================================================================
// Unsupported Policy Combination Tests
// ============================================================================

TEST_F(PolicyDispatchTest, UnsupportedPartitionReturnsNotSupported) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_HASH;  // Hash is for maps only

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    EXPECT_EQ(status, DTL_ERROR_NOT_SUPPORTED);
    EXPECT_EQ(vec, nullptr);
}

TEST_F(PolicyDispatchTest, BlockCyclicWithZeroBlockSizeReturnsError) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_BLOCK_CYCLIC;
    opts.block_size = 0;  // Invalid block size

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    // Should fail with invalid argument (block_size must be > 0 for block-cyclic)
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Cyclic Partition Behavior Tests
// ============================================================================

TEST_F(PolicyDispatchTest, CyclicPartitionOwnershipRoundRobin) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_CYCLIC;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_INT32, 10, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    dtl_rank_t num_ranks = dtl_vector_num_ranks(vec);
    dtl_rank_t my_rank = dtl_vector_rank(vec);

    // In cyclic partition, element i belongs to rank (i % num_ranks)
    for (dtl_index_t i = 0; i < 10; ++i) {
        dtl_rank_t expected_owner = static_cast<dtl_rank_t>(i % num_ranks);
        EXPECT_EQ(dtl_vector_owner(vec, i), expected_owner)
            << "Element " << i << " should be owned by rank " << expected_owner;

        // Check is_local is consistent with owner
        bool should_be_local = (expected_owner == my_rank);
        EXPECT_EQ(dtl_vector_is_local(vec, i), should_be_local ? 1 : 0);
    }

    dtl_vector_destroy(vec);
}

TEST_F(PolicyDispatchTest, ReplicatedPartitionAllLocal) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);
    opts.partition = DTL_PARTITION_REPLICATED;

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_INT32, 10, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    // In replicated partition, all elements are local
    EXPECT_EQ(dtl_vector_local_size(vec), 10u);

    for (dtl_index_t i = 0; i < 10; ++i) {
        EXPECT_EQ(dtl_vector_is_local(vec, i), 1)
            << "Element " << i << " should be local in replicated partition";
    }

    dtl_vector_destroy(vec);
}

// ============================================================================
// Copy Helper Tests
// ============================================================================

TEST_F(PolicyDispatchTest, VectorCopyToHostWorks) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_FLOAT64, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    // Fill with a value
    double fill_value = 42.0;
    status = dtl_vector_fill_local(vec, &fill_value);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Copy to host buffer
    std::size_t local_size = dtl_vector_local_size(vec);
    std::vector<double> host_buffer(local_size);

    status = dtl_vector_copy_to_host(vec, host_buffer.data(), local_size);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Verify values
    for (std::size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(host_buffer[i], 42.0);
    }

    dtl_vector_destroy(vec);
}

TEST_F(PolicyDispatchTest, VectorCopyFromHostWorks) {
    dtl_container_options opts;
    dtl_container_options_init(&opts);

    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create_with_options(
        ctx, DTL_DTYPE_INT32, 100, &opts, &vec);

    ASSERT_EQ(status, DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    // Create source data
    std::size_t local_size = dtl_vector_local_size(vec);
    std::vector<int32_t> source(local_size);
    for (std::size_t i = 0; i < local_size; ++i) {
        source[i] = static_cast<int32_t>(i * 2);
    }

    // Copy from host
    status = dtl_vector_copy_from_host(vec, source.data(), local_size);
    ASSERT_EQ(status, DTL_SUCCESS);

    // Verify via direct data access
    const int32_t* data = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(data[i], static_cast<int32_t>(i * 2));
    }

    dtl_vector_destroy(vec);
}

// ============================================================================
// Error Handler Callback Tests
// ============================================================================

namespace {
    bool g_callback_invoked = false;
    dtl_status g_callback_status = DTL_SUCCESS;

    void test_error_handler(dtl_status status, const char* /*message*/, void* /*user_data*/) {
        g_callback_invoked = true;
        g_callback_status = status;
    }
}

TEST_F(PolicyDispatchTest, ErrorHandlerRegistration) {
    g_callback_invoked = false;
    g_callback_status = DTL_SUCCESS;

    dtl_status status = dtl_context_set_error_handler(ctx, test_error_handler, nullptr);
    EXPECT_EQ(status, DTL_SUCCESS);

    // Clear for future tests
    dtl_context_set_error_handler(ctx, nullptr, nullptr);
}

TEST_F(PolicyDispatchTest, ErrorHandlerInvokedPerContext) {
    struct callback_state {
        bool invoked = false;
        dtl_status status = DTL_SUCCESS;
    };

    auto handler = [](dtl_status status, const char* /*message*/, void* user_data) {
        auto* s = static_cast<callback_state*>(user_data);
        s->invoked = true;
        s->status = status;
    };

    // Create a second context to ensure handlers are not stored in thread-local state.
    dtl_context_t ctx2 = nullptr;
    ASSERT_EQ(dtl_context_create_default(&ctx2), DTL_SUCCESS);

    callback_state s1{};
    callback_state s2{};

    ASSERT_EQ(dtl_context_set_error_handler(ctx, handler, &s1), DTL_SUCCESS);
    ASSERT_EQ(dtl_context_set_error_handler(ctx2, handler, &s2), DTL_SUCCESS);

    dtl_container_options opts;
    dtl_container_options_init(&opts);
    dtl_container_options_set_error(&opts, DTL_ERROR_POLICY_CALLBACK);
    opts.partition = static_cast<dtl_partition_policy>(99);  // Invalid -> triggers error policy path

    // Trigger an error on ctx (handler must be s1, not s2)
    dtl_vector_t vec = nullptr;
    dtl_status st1 = dtl_vector_create_with_options(ctx, DTL_DTYPE_FLOAT64, 10, &opts, &vec);
    EXPECT_NE(st1, DTL_SUCCESS);
    EXPECT_EQ(vec, nullptr);
    EXPECT_TRUE(s1.invoked);
    EXPECT_FALSE(s2.invoked);
    EXPECT_EQ(s1.status, st1);

    // Trigger an error on ctx2 (handler must be s2)
    s1.invoked = false;
    s1.status = DTL_SUCCESS;
    s2.invoked = false;
    s2.status = DTL_SUCCESS;

    vec = nullptr;
    dtl_status st2 = dtl_vector_create_with_options(ctx2, DTL_DTYPE_FLOAT64, 10, &opts, &vec);
    EXPECT_NE(st2, DTL_SUCCESS);
    EXPECT_EQ(vec, nullptr);
    EXPECT_FALSE(s1.invoked);
    EXPECT_TRUE(s2.invoked);
    EXPECT_EQ(s2.status, st2);

    // Cleanup
    dtl_context_set_error_handler(ctx, nullptr, nullptr);
    dtl_context_set_error_handler(ctx2, nullptr, nullptr);
    dtl_context_destroy(ctx2);
}

TEST_F(PolicyDispatchTest, ErrorHandlerInvokedOnVectorSetLocalOutOfBounds) {
    g_callback_invoked = false;
    g_callback_status = DTL_SUCCESS;

    ASSERT_EQ(dtl_context_set_error_handler(ctx, test_error_handler, nullptr), DTL_SUCCESS);

    dtl_container_options opts;
    dtl_container_options_init(&opts);
    dtl_container_options_set_error(&opts, DTL_ERROR_POLICY_CALLBACK);

    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 1, &opts, &vec), DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    const int32_t v = 7;
    dtl_status st = dtl_vector_set_local(vec, 1, &v);  // local_size == 1 -> OOB
    EXPECT_EQ(st, DTL_ERROR_OUT_OF_BOUNDS);
    EXPECT_TRUE(g_callback_invoked);
    EXPECT_EQ(g_callback_status, st);

    dtl_vector_destroy(vec);
    dtl_context_set_error_handler(ctx, nullptr, nullptr);
}

TEST_F(PolicyDispatchTest, ErrorHandlerInvokedOnVectorCopyToHostNullBuffer) {
    g_callback_invoked = false;
    g_callback_status = DTL_SUCCESS;

    ASSERT_EQ(dtl_context_set_error_handler(ctx, test_error_handler, nullptr), DTL_SUCCESS);

    dtl_container_options opts;
    dtl_container_options_init(&opts);
    dtl_container_options_set_error(&opts, DTL_ERROR_POLICY_CALLBACK);

    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 1, &opts, &vec), DTL_SUCCESS);
    ASSERT_NE(vec, nullptr);

    dtl_status st = dtl_vector_copy_to_host(vec, nullptr, 0);
    EXPECT_EQ(st, DTL_ERROR_NULL_POINTER);
    EXPECT_TRUE(g_callback_invoked);
    EXPECT_EQ(g_callback_status, st);

    dtl_vector_destroy(vec);
    dtl_context_set_error_handler(ctx, nullptr, nullptr);
}

TEST_F(PolicyDispatchTest, ErrorHandlerInvokedOnTransformVectorSizeMismatch) {
    g_callback_invoked = false;
    g_callback_status = DTL_SUCCESS;

    ASSERT_EQ(dtl_context_set_error_handler(ctx, test_error_handler, nullptr), DTL_SUCCESS);

    dtl_container_options opts_default;
    dtl_container_options_init(&opts_default);

    dtl_container_options opts_cb;
    dtl_container_options_init(&opts_cb);
    dtl_container_options_set_error(&opts_cb, DTL_ERROR_POLICY_CALLBACK);

    dtl_size_t world = static_cast<dtl_size_t>(dtl_context_size(ctx));
    ASSERT_GE(world, 1u);

    // Choose sizes so every rank sees a local-size mismatch under block partition:
    // src local size = 1, dst local size = 2.
    dtl_vector_t src = nullptr;
    dtl_vector_t dst = nullptr;
    ASSERT_EQ(dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, world, &opts_default, &src), DTL_SUCCESS);
    ASSERT_EQ(dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 2 * world, &opts_cb, &dst), DTL_SUCCESS);

    auto transform = [](const void* in, void* out, dtl_size_t /*idx*/, void* /*ud*/) {
        *static_cast<int32_t*>(out) = *static_cast<const int32_t*>(in);
    };

    dtl_status st = dtl_transform_vector(src, dst, transform, nullptr);
    EXPECT_EQ(st, DTL_ERROR_INVALID_ARGUMENT);
    EXPECT_TRUE(g_callback_invoked);
    EXPECT_EQ(g_callback_status, st);

    dtl_vector_destroy(src);
    dtl_vector_destroy(dst);
    dtl_context_set_error_handler(ctx, nullptr, nullptr);
}

TEST_F(PolicyDispatchTest, ErrorHandlerInvokedOnArraySetLocalOutOfBounds) {
    g_callback_invoked = false;
    g_callback_status = DTL_SUCCESS;

    ASSERT_EQ(dtl_context_set_error_handler(ctx, test_error_handler, nullptr), DTL_SUCCESS);

    dtl_container_options opts;
    dtl_container_options_init(&opts);
    dtl_container_options_set_error(&opts, DTL_ERROR_POLICY_CALLBACK);

    dtl_array_t arr = nullptr;
    ASSERT_EQ(dtl_array_create_with_options(ctx, DTL_DTYPE_INT32, 1, &opts, &arr), DTL_SUCCESS);
    ASSERT_NE(arr, nullptr);

    const int32_t v = 7;
    dtl_status st = dtl_array_set_local(arr, 1, &v);  // local_size == 1 -> OOB
    EXPECT_EQ(st, DTL_ERROR_OUT_OF_BOUNDS);
    EXPECT_TRUE(g_callback_invoked);
    EXPECT_EQ(g_callback_status, st);

    dtl_array_destroy(arr);
    dtl_context_set_error_handler(ctx, nullptr, nullptr);
}

TEST(PolicyDispatchDeathTest, TerminateErrorPolicyAbortsOnError) {
    const char* mpi_size_env = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (mpi_size_env && std::strcmp(mpi_size_env, "1") != 0) {
        GTEST_SKIP() << "Skipping death test under MPI multi-rank launch";
    }
#if GTEST_HAS_DEATH_TEST
    ASSERT_DEATH(
        {
            dtl_context_t ctx = nullptr;
            (void)dtl_context_create_default(&ctx);

            dtl_container_options opts;
            dtl_container_options_init(&opts);
            dtl_container_options_set_error(&opts, DTL_ERROR_POLICY_TERMINATE);

            dtl_vector_t vec = nullptr;
            (void)dtl_vector_create_with_options(ctx, DTL_DTYPE_INT32, 1, &opts, &vec);

            const int32_t v = 7;
            (void)dtl_vector_set_local(vec, 1, &v);  // local_size == 1 -> OOB -> abort()
        },
        "");
#else
    GTEST_SKIP() << "Death tests are not supported on this platform/build";
#endif
}

// ============================================================================
// Invalid Handle Tests
// ============================================================================

TEST(PolicyQueryInvalidHandle, VectorNullReturnsNegative) {
    EXPECT_EQ(dtl_vector_partition_policy(nullptr), static_cast<dtl_partition_policy>(-1));
    EXPECT_EQ(dtl_vector_placement_policy(nullptr), static_cast<dtl_placement_policy>(-1));
    EXPECT_EQ(dtl_vector_execution_policy(nullptr), static_cast<dtl_execution_policy>(-1));
    EXPECT_EQ(dtl_vector_device_id(nullptr), -1);
    EXPECT_EQ(dtl_vector_consistency_policy(nullptr), static_cast<dtl_consistency_policy>(-1));
    EXPECT_EQ(dtl_vector_error_policy(nullptr), static_cast<dtl_error_policy>(-1));
}

TEST(PolicyQueryInvalidHandle, ArrayNullReturnsNegative) {
    EXPECT_EQ(dtl_array_partition_policy(nullptr), static_cast<dtl_partition_policy>(-1));
    EXPECT_EQ(dtl_array_placement_policy(nullptr), static_cast<dtl_placement_policy>(-1));
    EXPECT_EQ(dtl_array_execution_policy(nullptr), static_cast<dtl_execution_policy>(-1));
    EXPECT_EQ(dtl_array_device_id(nullptr), -1);
    EXPECT_EQ(dtl_array_consistency_policy(nullptr), static_cast<dtl_consistency_policy>(-1));
    EXPECT_EQ(dtl_array_error_policy(nullptr), static_cast<dtl_error_policy>(-1));
}
