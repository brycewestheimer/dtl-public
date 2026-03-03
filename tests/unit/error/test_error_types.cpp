// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_error_types.cpp
/// @brief Unit tests for error types and status codes
/// @details Tests for Phase 11.5: error type basics

#include <dtl/error/status.hpp>
#include <dtl/core/types.hpp>

#include <gtest/gtest.h>

#include <string>

namespace dtl::test {

// =============================================================================
// Status Code Tests
// =============================================================================

TEST(ErrorTypesTest, StatusCodeOk) {
    EXPECT_EQ(static_cast<int>(status_code::ok), 0);
}

TEST(ErrorTypesTest, StatusCodeValues) {
    // Verify error codes are non-zero
    EXPECT_NE(static_cast<int>(status_code::invalid_argument), 0);
    EXPECT_NE(static_cast<int>(status_code::out_of_range), 0);
    EXPECT_NE(static_cast<int>(status_code::not_implemented), 0);
    EXPECT_NE(static_cast<int>(status_code::internal_error), 0);
    EXPECT_NE(static_cast<int>(status_code::communication_error), 0);
}

TEST(ErrorTypesTest, StatusCodeDistinct) {
    // Each error code should be distinct
    EXPECT_NE(status_code::invalid_argument, status_code::out_of_range);
    EXPECT_NE(status_code::out_of_range, status_code::not_implemented);
    EXPECT_NE(status_code::not_implemented, status_code::internal_error);
}

TEST(ErrorTypesTest, StatusCodeCategories) {
    // Verify category organization
    EXPECT_GE(static_cast<int>(status_code::communication_error), 100);
    EXPECT_LT(static_cast<int>(status_code::communication_error), 200);

    EXPECT_GE(static_cast<int>(status_code::memory_error), 200);
    EXPECT_LT(static_cast<int>(status_code::memory_error), 300);

    EXPECT_GE(static_cast<int>(status_code::bounds_error), 400);
    EXPECT_LT(static_cast<int>(status_code::bounds_error), 500);
}

// =============================================================================
// Status Construction Tests
// =============================================================================

TEST(ErrorTypesTest, StatusDefaultConstruction) {
    status s;

    EXPECT_EQ(s.code(), status_code::ok);
    EXPECT_TRUE(s.ok());
}

TEST(ErrorTypesTest, StatusFromCode) {
    status s{status_code::invalid_argument};

    EXPECT_EQ(s.code(), status_code::invalid_argument);
    EXPECT_FALSE(s.ok());
}

TEST(ErrorTypesTest, StatusFromCodeAndRank) {
    status s{status_code::out_of_range, 3};

    EXPECT_EQ(s.code(), status_code::out_of_range);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.rank(), 3);
}

TEST(ErrorTypesTest, StatusFromCodeRankAndMessage) {
    status s{status_code::out_of_range, 2, "index too large"};

    EXPECT_EQ(s.code(), status_code::out_of_range);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.rank(), 2);
    EXPECT_EQ(s.message(), "index too large");
}

TEST(ErrorTypesTest, StatusOkCode) {
    status s{status_code::ok};

    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code(), status_code::ok);
}

// =============================================================================
// Status Query Tests
// =============================================================================

TEST(ErrorTypesTest, StatusOk) {
    status ok{status_code::ok};
    status err{status_code::invalid_argument};

    EXPECT_TRUE(ok.ok());
    EXPECT_FALSE(err.ok());
}

TEST(ErrorTypesTest, StatusIsError) {
    status ok{status_code::ok};
    status err{status_code::invalid_argument};

    EXPECT_FALSE(ok.is_error());
    EXPECT_TRUE(err.is_error());
}

TEST(ErrorTypesTest, StatusMessage) {
    status s{status_code::internal_error, no_rank, "something went wrong"};

    EXPECT_EQ(s.message(), "something went wrong");
}

TEST(ErrorTypesTest, StatusEmptyMessage) {
    status s{status_code::invalid_argument};

    // Empty message is valid
    EXPECT_TRUE(s.message().empty());
}

TEST(ErrorTypesTest, StatusRankNoRank) {
    status s{status_code::invalid_argument};

    EXPECT_EQ(s.rank(), no_rank);
}

TEST(ErrorTypesTest, StatusRankValue) {
    status s{status_code::invalid_argument, 5};

    EXPECT_EQ(s.rank(), 5);
}

// =============================================================================
// Status Comparison Tests
// =============================================================================

TEST(ErrorTypesTest, StatusEquality) {
    status s1{status_code::invalid_argument};
    status s2{status_code::invalid_argument};
    status s3{status_code::out_of_range};

    EXPECT_EQ(s1, s2);
    EXPECT_NE(s1, s3);
}

TEST(ErrorTypesTest, StatusEqualityWithCode) {
    status s{status_code::invalid_argument};

    EXPECT_EQ(s, status_code::invalid_argument);
    EXPECT_NE(s, status_code::out_of_range);
}

TEST(ErrorTypesTest, StatusOkEquality) {
    status s1{status_code::ok};
    status s2;

    EXPECT_EQ(s1, s2);
}

// =============================================================================
// Status Factory Tests
// =============================================================================

TEST(ErrorTypesTest, OkStatusFactory) {
    auto s = ok_status();

    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code(), status_code::ok);
}

TEST(ErrorTypesTest, ErrorStatusFactory) {
    auto s = error_status(status_code::invalid_argument, 1, "bad input");

    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), status_code::invalid_argument);
    EXPECT_EQ(s.rank(), 1);
    EXPECT_EQ(s.message(), "bad input");
}

TEST(ErrorTypesTest, ErrorStatusFactoryDefaults) {
    auto s = error_status(status_code::internal_error);

    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), status_code::internal_error);
    EXPECT_EQ(s.rank(), no_rank);
    EXPECT_TRUE(s.message().empty());
}

// =============================================================================
// Status Code Name Tests
// =============================================================================

TEST(ErrorTypesTest, StatusCodeNames) {
    EXPECT_EQ(status_code_name(status_code::ok), "ok");
    EXPECT_EQ(status_code_name(status_code::invalid_argument), "invalid_argument");
    EXPECT_EQ(status_code_name(status_code::out_of_bounds), "out_of_bounds");
    EXPECT_EQ(status_code_name(status_code::communication_error), "communication_error");
}

TEST(ErrorTypesTest, StatusCategory) {
    EXPECT_EQ(status_category(status_code::ok), "success");
    EXPECT_EQ(status_category(status_code::communication_error), "communication");
    EXPECT_EQ(status_category(status_code::memory_error), "memory");
    EXPECT_EQ(status_category(status_code::bounds_error), "bounds");
}

// =============================================================================
// Common Error Codes Tests
// =============================================================================

TEST(ErrorTypesTest, KeyNotFound) {
    status s{status_code::key_not_found};

    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), status_code::key_not_found);
}

TEST(ErrorTypesTest, Timeout) {
    status s{status_code::timeout};

    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), status_code::timeout);
}

TEST(ErrorTypesTest, InvalidState) {
    status s{status_code::invalid_state};

    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), status_code::invalid_state);
}

// =============================================================================
// Status to_string Tests
// =============================================================================

TEST(ErrorTypesTest, ToStringOk) {
    status s;

    EXPECT_EQ(s.to_string(), "ok");
}

TEST(ErrorTypesTest, ToStringError) {
    status s{status_code::invalid_argument};

    std::string str = s.to_string();
    EXPECT_TRUE(str.find("error") != std::string::npos);
}

TEST(ErrorTypesTest, ToStringWithRank) {
    status s{status_code::internal_error, 3};

    std::string str = s.to_string();
    EXPECT_TRUE(str.find("rank") != std::string::npos);
    EXPECT_TRUE(str.find("3") != std::string::npos);
}

TEST(ErrorTypesTest, ToStringWithMessage) {
    status s{status_code::internal_error, no_rank, "custom message"};

    std::string str = s.to_string();
    EXPECT_TRUE(str.find("custom message") != std::string::npos);
}

// =============================================================================
// Boolean Conversion Tests
// =============================================================================

TEST(ErrorTypesTest, BoolConversionOk) {
    status s{status_code::ok};

    EXPECT_TRUE(static_cast<bool>(s));
}

TEST(ErrorTypesTest, BoolConversionError) {
    status s{status_code::invalid_argument};

    EXPECT_FALSE(static_cast<bool>(s));
}

}  // namespace dtl::test
