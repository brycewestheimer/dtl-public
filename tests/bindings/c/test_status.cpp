// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_status.cpp
 * @brief Unit tests for DTL C bindings status codes
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_status.h>
#include <cstring>

// ============================================================================
// Status OK Tests
// ============================================================================

TEST(CBindingsStatus, SuccessIsZero) {
    EXPECT_EQ(DTL_SUCCESS, 0);
}

TEST(CBindingsStatus, StatusOkReturnsTrue) {
    EXPECT_EQ(dtl_status_ok(DTL_SUCCESS), 1);
    EXPECT_EQ(dtl_status_ok(0), 1);
}

TEST(CBindingsStatus, StatusOkReturnsFalseForErrors) {
    EXPECT_EQ(dtl_status_ok(DTL_ERROR_COMMUNICATION), 0);
    EXPECT_EQ(dtl_status_ok(DTL_ERROR_MEMORY), 0);
    EXPECT_EQ(dtl_status_ok(DTL_ERROR_BOUNDS), 0);
    EXPECT_EQ(dtl_status_ok(DTL_ERROR_UNKNOWN), 0);
}

TEST(CBindingsStatus, StatusIsErrorReturnsFalseForSuccess) {
    EXPECT_EQ(dtl_status_is_error(DTL_SUCCESS), 0);
}

TEST(CBindingsStatus, StatusIsErrorReturnsTrueForErrors) {
    EXPECT_EQ(dtl_status_is_error(DTL_ERROR_COMMUNICATION), 1);
    EXPECT_EQ(dtl_status_is_error(DTL_ERROR_MEMORY), 1);
    EXPECT_EQ(dtl_status_is_error(DTL_ERROR_NOT_IMPLEMENTED), 1);
}

// ============================================================================
// Status Message Tests
// ============================================================================

TEST(CBindingsStatus, SuccessMessageNotNull) {
    const char* msg = dtl_status_message(DTL_SUCCESS);
    EXPECT_NE(msg, nullptr);
    EXPECT_GT(strlen(msg), 0u);
}

TEST(CBindingsStatus, ErrorMessagesNotNull) {
    // Test a sample of error codes
    EXPECT_NE(dtl_status_message(DTL_ERROR_COMMUNICATION), nullptr);
    EXPECT_NE(dtl_status_message(DTL_ERROR_MEMORY), nullptr);
    EXPECT_NE(dtl_status_message(DTL_ERROR_BOUNDS), nullptr);
    EXPECT_NE(dtl_status_message(DTL_ERROR_NOT_IMPLEMENTED), nullptr);
}

TEST(CBindingsStatus, UnknownCodeHasMessage) {
    const char* msg = dtl_status_message(12345);
    EXPECT_NE(msg, nullptr);
    EXPECT_GT(strlen(msg), 0u);
}

// ============================================================================
// Status Name Tests
// ============================================================================

TEST(CBindingsStatus, SuccessNameIsOk) {
    EXPECT_STREQ(dtl_status_name(DTL_SUCCESS), "ok");
}

TEST(CBindingsStatus, ErrorNamesNotEmpty) {
    EXPECT_GT(strlen(dtl_status_name(DTL_ERROR_COMMUNICATION)), 0u);
    EXPECT_GT(strlen(dtl_status_name(DTL_ERROR_MEMORY)), 0u);
    EXPECT_GT(strlen(dtl_status_name(DTL_ERROR_NOT_IMPLEMENTED)), 0u);
}

TEST(CBindingsStatus, SpecificErrorNames) {
    EXPECT_STREQ(dtl_status_name(DTL_ERROR_COMMUNICATION), "communication_error");
    EXPECT_STREQ(dtl_status_name(DTL_ERROR_MEMORY), "memory_error");
    EXPECT_STREQ(dtl_status_name(DTL_ERROR_ALLOCATION_FAILED), "allocation_failed");
    EXPECT_STREQ(dtl_status_name(DTL_ERROR_NOT_IMPLEMENTED), "not_implemented");
}

// ============================================================================
// Status Category Tests
// ============================================================================

TEST(CBindingsStatus, SuccessCategoryIsSuccess) {
    EXPECT_STREQ(dtl_status_category(DTL_SUCCESS), "success");
}

TEST(CBindingsStatus, CommunicationCategoryCorrect) {
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_COMMUNICATION), "communication");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_SEND_FAILED), "communication");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_BARRIER_FAILED), "communication");
}

TEST(CBindingsStatus, MemoryCategoryCorrect) {
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_MEMORY), "memory");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_ALLOCATION_FAILED), "memory");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_OUT_OF_MEMORY), "memory");
}

TEST(CBindingsStatus, BoundsCategoryCorrect) {
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_BOUNDS), "bounds");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_OUT_OF_BOUNDS), "bounds");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_INVALID_ARGUMENT), "bounds");
}

TEST(CBindingsStatus, BackendCategoryCorrect) {
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_BACKEND), "backend");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_MPI), "backend");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_CUDA), "backend");
}

TEST(CBindingsStatus, InternalCategoryCorrect) {
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_INTERNAL), "internal");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_NOT_IMPLEMENTED), "internal");
    EXPECT_STREQ(dtl_status_category(DTL_ERROR_UNKNOWN), "internal");
}

// ============================================================================
// Category Code Tests
// ============================================================================

TEST(CBindingsStatus, CategoryCodeSuccess) {
    EXPECT_EQ(dtl_status_category_code(DTL_SUCCESS), 0);
}

TEST(CBindingsStatus, CategoryCodeCommunication) {
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_COMMUNICATION), 1);
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_SEND_FAILED), 1);
}

TEST(CBindingsStatus, CategoryCodeMemory) {
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_MEMORY), 2);
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_ALLOCATION_FAILED), 2);
}

TEST(CBindingsStatus, CategoryCodeBounds) {
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_BOUNDS), 4);
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_INVALID_ARGUMENT), 4);
}

TEST(CBindingsStatus, CategoryCodeInternal) {
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_INTERNAL), 9);
    EXPECT_EQ(dtl_status_category_code(DTL_ERROR_NOT_IMPLEMENTED), 9);
}

// ============================================================================
// Is Category Tests
// ============================================================================

TEST(CBindingsStatus, IsCategorySuccess) {
    EXPECT_EQ(dtl_status_is_category(DTL_SUCCESS, DTL_CATEGORY_SUCCESS), 1);
    EXPECT_EQ(dtl_status_is_category(DTL_SUCCESS, DTL_CATEGORY_COMMUNICATION), 0);
}

TEST(CBindingsStatus, IsCategoryCommunication) {
    EXPECT_EQ(dtl_status_is_category(DTL_ERROR_COMMUNICATION, DTL_CATEGORY_COMMUNICATION), 1);
    EXPECT_EQ(dtl_status_is_category(DTL_ERROR_SEND_FAILED, DTL_CATEGORY_COMMUNICATION), 1);
    EXPECT_EQ(dtl_status_is_category(DTL_ERROR_COMMUNICATION, DTL_CATEGORY_MEMORY), 0);
}

TEST(CBindingsStatus, IsCategoryMemory) {
    EXPECT_EQ(dtl_status_is_category(DTL_ERROR_MEMORY, DTL_CATEGORY_MEMORY), 1);
    EXPECT_EQ(dtl_status_is_category(DTL_ERROR_ALLOCATION_FAILED, DTL_CATEGORY_MEMORY), 1);
    EXPECT_EQ(dtl_status_is_category(DTL_ERROR_MEMORY, DTL_CATEGORY_BACKEND), 0);
}

// ============================================================================
// Error Code Range Tests
// ============================================================================

TEST(CBindingsStatus, CommunicationErrorsInRange) {
    EXPECT_GE(DTL_ERROR_COMMUNICATION, 100);
    EXPECT_LT(DTL_ERROR_COMMUNICATION, 200);
    EXPECT_GE(DTL_ERROR_COLLECTIVE_FAILED, 100);
    EXPECT_LT(DTL_ERROR_COLLECTIVE_FAILED, 200);
}

TEST(CBindingsStatus, MemoryErrorsInRange) {
    EXPECT_GE(DTL_ERROR_MEMORY, 200);
    EXPECT_LT(DTL_ERROR_MEMORY, 300);
    EXPECT_GE(DTL_ERROR_DEVICE_MEMORY, 200);
    EXPECT_LT(DTL_ERROR_DEVICE_MEMORY, 300);
}

TEST(CBindingsStatus, BoundsErrorsInRange) {
    EXPECT_GE(DTL_ERROR_BOUNDS, 400);
    EXPECT_LT(DTL_ERROR_BOUNDS, 500);
    EXPECT_GE(DTL_ERROR_NOT_SUPPORTED, 400);
    EXPECT_LT(DTL_ERROR_NOT_SUPPORTED, 500);
}

TEST(CBindingsStatus, BackendErrorsInRange) {
    EXPECT_GE(DTL_ERROR_BACKEND, 500);
    EXPECT_LT(DTL_ERROR_BACKEND, 600);
    EXPECT_GE(DTL_ERROR_SHMEM, 500);
    EXPECT_LT(DTL_ERROR_SHMEM, 600);
}

TEST(CBindingsStatus, InternalErrorsInRange) {
    EXPECT_GE(DTL_ERROR_INTERNAL, 900);
    EXPECT_LE(DTL_ERROR_INTERNAL, 999);
    EXPECT_GE(DTL_ERROR_UNKNOWN, 900);
    EXPECT_LE(DTL_ERROR_UNKNOWN, 999);
}
