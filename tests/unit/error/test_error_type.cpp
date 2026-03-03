// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_error_type.cpp
/// @brief Unit tests for dtl/error/error_type.hpp
/// @details Tests the error class and make_error factory functions.

#include <dtl/error/error_type.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <string>

namespace dtl::test {

// =============================================================================
// Error Construction Tests
// =============================================================================

TEST(ErrorTypeTest, ConstructFromStatus) {
    status s{status_code::timeout, 3, "connection lost"};
    error e{s};

    EXPECT_EQ(e.code(), status_code::timeout);
    EXPECT_EQ(e.rank(), 3);
    EXPECT_EQ(e.message(), "connection lost");
}

TEST(ErrorTypeTest, ConstructFromStatusCode) {
    error e{status_code::invalid_argument, 1, "bad input"};

    EXPECT_EQ(e.code(), status_code::invalid_argument);
    EXPECT_EQ(e.rank(), 1);
    EXPECT_EQ(e.message(), "bad input");
}

TEST(ErrorTypeTest, ConstructWithDefaultRank) {
    error e{status_code::internal_error};

    EXPECT_EQ(e.code(), status_code::internal_error);
    EXPECT_EQ(e.rank(), no_rank);
    EXPECT_TRUE(e.message().empty());
}

TEST(ErrorTypeTest, ConstructWithEmptyMessage) {
    error e{status_code::not_implemented, 0, ""};

    EXPECT_EQ(e.code(), status_code::not_implemented);
    EXPECT_EQ(e.rank(), 0);
    EXPECT_TRUE(e.message().empty());
}

// =============================================================================
// Observer Tests
// =============================================================================

TEST(ErrorTypeTest, GetStatus) {
    error e{status_code::allocation_failed, 2, "out of GPU memory"};
    const auto& s = e.get_status();

    EXPECT_EQ(s.code(), status_code::allocation_failed);
    EXPECT_EQ(s.rank(), 2);
    EXPECT_EQ(s.message(), "out of GPU memory");
}

TEST(ErrorTypeTest, Code) {
    error e{status_code::send_failed};
    EXPECT_EQ(e.code(), status_code::send_failed);
}

TEST(ErrorTypeTest, Rank) {
    error e{status_code::timeout, 5};
    EXPECT_EQ(e.rank(), 5);
}

TEST(ErrorTypeTest, RankDefaultNoRank) {
    error e{status_code::timeout};
    EXPECT_EQ(e.rank(), no_rank);
}

TEST(ErrorTypeTest, Message) {
    error e{status_code::internal_error, no_rank, "something failed"};
    EXPECT_EQ(e.message(), "something failed");
}

// =============================================================================
// Source Location Tests
// =============================================================================

TEST(ErrorTypeTest, SourceLocationCaptured) {
    error e{status_code::internal_error};

    // File name should contain this test file name
    std::string file_name = e.file_name();
    EXPECT_FALSE(file_name.empty());
    EXPECT_NE(file_name.find("test_error_type"), std::string::npos);
}

TEST(ErrorTypeTest, LineNumberCaptured) {
    error e{status_code::internal_error};

    // Line should be a positive number (non-zero)
    EXPECT_GT(e.line(), 0u);
}

TEST(ErrorTypeTest, FunctionNameCaptured) {
    error e{status_code::internal_error};

    std::string func_name = e.function_name();
    EXPECT_FALSE(func_name.empty());
}

TEST(ErrorTypeTest, LocationAccessor) {
    error e{status_code::invalid_state};
    const auto& loc = e.location();

    EXPECT_GT(loc.line(), 0u);
    EXPECT_NE(std::string(loc.file_name()).size(), 0u);
}

// =============================================================================
// to_string Tests
// =============================================================================

TEST(ErrorTypeTest, ToStringContainsStatusInfo) {
    error e{status_code::timeout, 3, "network issue"};
    std::string str = e.to_string();

    EXPECT_NE(str.find("network issue"), std::string::npos);
    EXPECT_NE(str.find("rank 3"), std::string::npos);
}

TEST(ErrorTypeTest, ToStringContainsLocation) {
    error e{status_code::internal_error};
    std::string str = e.to_string();

    // Should contain "at" and file info
    EXPECT_NE(str.find("at"), std::string::npos);
    EXPECT_NE(str.find("test_error_type"), std::string::npos);
}

TEST(ErrorTypeTest, ToStringMinimalError) {
    error e{status_code::unknown_error};
    std::string str = e.to_string();

    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("error"), std::string::npos);
}

// =============================================================================
// Implicit Conversion Tests
// =============================================================================

TEST(ErrorTypeTest, ImplicitConversionToStatus) {
    error e{status_code::send_failed, 1, "send error"};
    const status& s = e;  // Implicit conversion

    EXPECT_EQ(s.code(), status_code::send_failed);
    EXPECT_EQ(s.rank(), 1);
    EXPECT_EQ(s.message(), "send error");
}

// =============================================================================
// make_error Factory Tests
// =============================================================================

TEST(MakeErrorTest, WithCodeAndMessage) {
    auto e = make_error(status_code::invalid_argument, "bad param");

    EXPECT_EQ(e.code(), status_code::invalid_argument);
    EXPECT_EQ(e.rank(), no_rank);
    EXPECT_EQ(e.message(), "bad param");
}

TEST(MakeErrorTest, WithCodeRankAndMessage) {
    auto e = make_error(status_code::allocation_failed, 2, "out of memory");

    EXPECT_EQ(e.code(), status_code::allocation_failed);
    EXPECT_EQ(e.rank(), 2);
    EXPECT_EQ(e.message(), "out of memory");
}

TEST(MakeErrorTest, WithCodeOnly) {
    auto e = make_error(status_code::not_implemented);

    EXPECT_EQ(e.code(), status_code::not_implemented);
    EXPECT_EQ(e.rank(), no_rank);
    EXPECT_TRUE(e.message().empty());
}

TEST(MakeErrorTest, SourceLocationCaptured) {
    auto e = make_error(status_code::internal_error, "test");
    std::string fname = e.file_name();
    EXPECT_NE(fname.find("test_error_type"), std::string::npos);
}

// =============================================================================
// Integration with result<T>
// =============================================================================

TEST(ErrorTypeTest, ConstructResultFromError) {
    error e{status_code::timeout, 0, "timed out"};
    result<int> r(e);

    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::timeout);
}

TEST(ErrorTypeTest, ConstructResultFromMoveError) {
    error e{status_code::send_failed, 1, "fail"};
    result<int> r(std::move(e));

    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::send_failed);
}

TEST(ErrorTypeTest, ConstructVoidResultFromError) {
    error e{status_code::barrier_failed};
    result<void> r(e);

    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::barrier_failed);
}

// =============================================================================
// Various Status Codes with Error
// =============================================================================

TEST(ErrorTypeTest, CommunicationError) {
    error e{status_code::communication_error};
    EXPECT_EQ(e.code(), status_code::communication_error);
}

TEST(ErrorTypeTest, MemoryError) {
    error e{status_code::memory_error};
    EXPECT_EQ(e.code(), status_code::memory_error);
}

TEST(ErrorTypeTest, BackendError) {
    error e{status_code::backend_error};
    EXPECT_EQ(e.code(), status_code::backend_error);
}

TEST(ErrorTypeTest, SerializationError) {
    error e{status_code::serialization_error};
    EXPECT_EQ(e.code(), status_code::serialization_error);
}

TEST(ErrorTypeTest, BoundsError) {
    error e{status_code::bounds_error};
    EXPECT_EQ(e.code(), status_code::bounds_error);
}

TEST(ErrorTypeTest, AlgorithmError) {
    error e{status_code::algorithm_error};
    EXPECT_EQ(e.code(), status_code::algorithm_error);
}

TEST(ErrorTypeTest, ConsistencyError) {
    error e{status_code::consistency_error};
    EXPECT_EQ(e.code(), status_code::consistency_error);
}

}  // namespace dtl::test
