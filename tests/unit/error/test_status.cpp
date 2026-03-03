// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_status.cpp
/// @brief Unit tests for dtl/error/status.hpp
/// @details Tests status codes, status class, and related utilities.

#include <dtl/error/status.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Status Code Tests
// =============================================================================

TEST(StatusCodeTest, SuccessCodeIsZero) {
    EXPECT_EQ(static_cast<int>(status_code::ok), 0);
}

TEST(StatusCodeTest, CommunicationCodesInRange) {
    EXPECT_GE(static_cast<int>(status_code::communication_error), 100);
    EXPECT_LT(static_cast<int>(status_code::communication_error), 200);
    EXPECT_GE(static_cast<int>(status_code::send_failed), 100);
    EXPECT_GE(static_cast<int>(status_code::recv_failed), 100);
    EXPECT_GE(static_cast<int>(status_code::timeout), 100);
    EXPECT_GE(static_cast<int>(status_code::canceled), 100);
    EXPECT_GE(static_cast<int>(status_code::collective_failure), 100);
}

TEST(StatusCodeTest, MemoryCodesInRange) {
    EXPECT_GE(static_cast<int>(status_code::memory_error), 200);
    EXPECT_LT(static_cast<int>(status_code::memory_error), 300);
    EXPECT_GE(static_cast<int>(status_code::allocation_failed), 200);
    EXPECT_GE(static_cast<int>(status_code::out_of_memory), 200);
}

TEST(StatusCodeTest, BoundsCodesInRange) {
    EXPECT_GE(static_cast<int>(status_code::bounds_error), 400);
    EXPECT_LT(static_cast<int>(status_code::bounds_error), 500);
    EXPECT_GE(static_cast<int>(status_code::invalid_argument), 400);
    EXPECT_GE(static_cast<int>(status_code::not_supported), 400);
}

TEST(StatusCodeTest, ConsistencyCodesInRange) {
    EXPECT_GE(static_cast<int>(status_code::consistency_error), 700);
    EXPECT_LT(static_cast<int>(status_code::consistency_error), 800);
    EXPECT_GE(static_cast<int>(status_code::consistency_violation), 700);
    EXPECT_GE(static_cast<int>(status_code::structural_invalidation), 700);
}

// =============================================================================
// Status Category Tests
// =============================================================================

TEST(StatusCategoryTest, SuccessCategory) {
    EXPECT_EQ(status_category(status_code::ok), "success");
}

TEST(StatusCategoryTest, CommunicationCategory) {
    EXPECT_EQ(status_category(status_code::communication_error), "communication");
    EXPECT_EQ(status_category(status_code::send_failed), "communication");
    EXPECT_EQ(status_category(status_code::timeout), "communication");
    EXPECT_EQ(status_category(status_code::canceled), "communication");
}

TEST(StatusCategoryTest, MemoryCategory) {
    EXPECT_EQ(status_category(status_code::memory_error), "memory");
    EXPECT_EQ(status_category(status_code::allocation_failed), "memory");
}

TEST(StatusCategoryTest, SerializationCategory) {
    EXPECT_EQ(status_category(status_code::serialization_error), "serialization");
    EXPECT_EQ(status_category(status_code::buffer_too_small), "serialization");
}

TEST(StatusCategoryTest, BoundsCategory) {
    EXPECT_EQ(status_category(status_code::bounds_error), "bounds");
    EXPECT_EQ(status_category(status_code::out_of_bounds), "bounds");
    EXPECT_EQ(status_category(status_code::invalid_argument), "bounds");
    EXPECT_EQ(status_category(status_code::not_supported), "bounds");
}

TEST(StatusCategoryTest, BackendCategory) {
    EXPECT_EQ(status_category(status_code::backend_error), "backend");
    EXPECT_EQ(status_category(status_code::cuda_error), "backend");
}

TEST(StatusCategoryTest, AlgorithmCategory) {
    EXPECT_EQ(status_category(status_code::algorithm_error), "algorithm");
    EXPECT_EQ(status_category(status_code::precondition_failed), "algorithm");
}

TEST(StatusCategoryTest, ConsistencyCategory) {
    EXPECT_EQ(status_category(status_code::consistency_error), "consistency");
    EXPECT_EQ(status_category(status_code::consistency_violation), "consistency");
    EXPECT_EQ(status_category(status_code::structural_invalidation), "consistency");
}

TEST(StatusCategoryTest, InternalCategory) {
    EXPECT_EQ(status_category(status_code::internal_error), "internal");
    EXPECT_EQ(status_category(status_code::unknown_error), "internal");
}

// =============================================================================
// Status Code Name Tests
// =============================================================================

TEST(StatusCodeNameTest, AllCodesHaveNames) {
    // Sample of codes that should have names
    EXPECT_EQ(status_code_name(status_code::ok), "ok");
    EXPECT_EQ(status_code_name(status_code::send_failed), "send_failed");
    EXPECT_EQ(status_code_name(status_code::canceled), "canceled");
    EXPECT_EQ(status_code_name(status_code::collective_failure), "collective_failure");
    EXPECT_EQ(status_code_name(status_code::allocation_failed), "allocation_failed");
    EXPECT_EQ(status_code_name(status_code::invalid_argument), "invalid_argument");
    EXPECT_EQ(status_code_name(status_code::not_supported), "not_supported");
    EXPECT_EQ(status_code_name(status_code::consistency_violation), "consistency_violation");
    EXPECT_EQ(status_code_name(status_code::structural_invalidation), "structural_invalidation");
    EXPECT_EQ(status_code_name(status_code::unknown_error), "unknown_error");
}

TEST(StatusCodeNameTest, ConstexprUsable) {
    constexpr auto name = status_code_name(status_code::ok);
    static_assert(name == "ok");
}

// =============================================================================
// Status Class Tests
// =============================================================================

TEST(StatusTest, DefaultConstruction) {
    status s;
    EXPECT_TRUE(s.ok());
    EXPECT_FALSE(s.is_error());
    EXPECT_EQ(s.code(), status_code::ok);
}

TEST(StatusTest, ConstructFromCode) {
    status s(status_code::send_failed);
    EXPECT_FALSE(s.ok());
    EXPECT_TRUE(s.is_error());
    EXPECT_EQ(s.code(), status_code::send_failed);
}

TEST(StatusTest, ConstructWithRank) {
    status s(status_code::rank_failure, 5);
    EXPECT_EQ(s.code(), status_code::rank_failure);
    EXPECT_EQ(s.rank(), 5);
}

TEST(StatusTest, ConstructWithMessage) {
    status s(status_code::backend_error, 3, "CUDA error 701");
    EXPECT_EQ(s.code(), status_code::backend_error);
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s.message(), "CUDA error 701");
}

TEST(StatusTest, BoolConversion) {
    status ok;
    status err(status_code::timeout);

    EXPECT_TRUE(static_cast<bool>(ok));
    EXPECT_FALSE(static_cast<bool>(err));
}

TEST(StatusTest, CategoryMethod) {
    status s(status_code::consistency_violation);
    EXPECT_EQ(s.category(), "consistency");
}

TEST(StatusTest, ToStringSuccess) {
    status s;
    EXPECT_EQ(s.to_string(), "ok");
}

TEST(StatusTest, ToStringError) {
    status s(status_code::send_failed, 2, "Connection refused");
    std::string str = s.to_string();

    EXPECT_NE(str.find("communication"), std::string::npos);
    EXPECT_NE(str.find("rank 2"), std::string::npos);
    EXPECT_NE(str.find("Connection refused"), std::string::npos);
}

TEST(StatusTest, EqualityComparison) {
    status s1(status_code::ok);
    status s2(status_code::ok);
    status s3(status_code::timeout);

    EXPECT_EQ(s1, s2);
    EXPECT_NE(s1, s3);
}

TEST(StatusTest, CompareWithCode) {
    status s(status_code::timeout);

    EXPECT_TRUE(s == status_code::timeout);
    EXPECT_FALSE(s == status_code::ok);
    EXPECT_FALSE(s != status_code::timeout);
    EXPECT_TRUE(s != status_code::ok);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(StatusFactoryTest, OkStatus) {
    auto s = ok_status();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code(), status_code::ok);
}

TEST(StatusFactoryTest, ErrorStatus) {
    auto s = error_status(status_code::allocation_failed, 0, "Out of GPU memory");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), status_code::allocation_failed);
    EXPECT_EQ(s.rank(), 0);
    EXPECT_EQ(s.message(), "Out of GPU memory");
}

TEST(StatusFactoryTest, ErrorStatusWithDefaults) {
    auto s = error_status(status_code::not_implemented);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), status_code::not_implemented);
    EXPECT_EQ(s.rank(), no_rank);
    EXPECT_TRUE(s.message().empty());
}

}  // namespace dtl::test
