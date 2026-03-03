// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_error_coverage.cpp
/// @brief Comprehensive unit tests for the DTL error module
/// @details Phase 14 T09: error construction, status, result monadic ops,
///          make_error overloads, collective_error aggregation, factory fns.

#include <dtl/error/status.hpp>
#include <dtl/error/result.hpp>
#include <dtl/error/error_type.hpp>
#include <dtl/error/error_code.hpp>
#include <dtl/error/collective_error.hpp>
#include <dtl/error/failure_handler.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

namespace dtl::test {

// =============================================================================
// Status Code Tests
// =============================================================================

TEST(StatusCodeTest, CategorySuccess) {
    EXPECT_EQ(status_category(status_code::ok), "success");
}

TEST(StatusCodeTest, CategoryCommunication) {
    EXPECT_EQ(status_category(status_code::send_failed), "communication");
    EXPECT_EQ(status_category(status_code::timeout), "communication");
}

TEST(StatusCodeTest, CategoryMemory) {
    EXPECT_EQ(status_category(status_code::allocation_failed), "memory");
    EXPECT_EQ(status_category(status_code::out_of_memory), "memory");
}

TEST(StatusCodeTest, CategorySerialization) {
    EXPECT_EQ(status_category(status_code::serialization_error), "serialization");
}

TEST(StatusCodeTest, CategoryBounds) {
    EXPECT_EQ(status_category(status_code::out_of_bounds), "bounds");
    EXPECT_EQ(status_category(status_code::invalid_rank), "bounds");
}

TEST(StatusCodeTest, CategoryBackend) {
    EXPECT_EQ(status_category(status_code::backend_error), "backend");
    EXPECT_EQ(status_category(status_code::cuda_error), "backend");
}

TEST(StatusCodeTest, CategoryAlgorithm) {
    EXPECT_EQ(status_category(status_code::algorithm_error), "algorithm");
}

TEST(StatusCodeTest, CategoryInternal) {
    EXPECT_EQ(status_category(status_code::internal_error), "internal");
    EXPECT_EQ(status_category(status_code::not_implemented), "internal");
}

TEST(StatusCodeTest, StatusCodeName) {
    EXPECT_EQ(status_code_name(status_code::ok), "ok");
    EXPECT_EQ(status_code_name(status_code::timeout), "timeout");
    EXPECT_EQ(status_code_name(status_code::allocation_failed), "allocation_failed");
    EXPECT_EQ(status_code_name(status_code::out_of_bounds), "out_of_bounds");
    EXPECT_EQ(status_code_name(status_code::not_implemented), "not_implemented");
}

// =============================================================================
// Status Class Tests
// =============================================================================

TEST(StatusCovTest, DefaultIsOk) {
    status s;
    EXPECT_TRUE(s.ok());
    EXPECT_FALSE(s.is_error());
    EXPECT_EQ(s.code(), status_code::ok);
}

TEST(StatusCovTest, ConstructFromCode) {
    status s(status_code::timeout);
    EXPECT_FALSE(s.ok());
    EXPECT_TRUE(s.is_error());
    EXPECT_EQ(s.code(), status_code::timeout);
}

TEST(StatusCovTest, ConstructWithRank) {
    status s(status_code::send_failed, 3);
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s.code(), status_code::send_failed);
}

TEST(StatusCovTest, ConstructWithMessage) {
    status s(status_code::allocation_failed, no_rank, "out of memory");
    EXPECT_EQ(s.message(), "out of memory");
    EXPECT_EQ(s.rank(), no_rank);
}

TEST(StatusCovTest, BoolConversion) {
    status ok_s;
    EXPECT_TRUE(static_cast<bool>(ok_s));

    status err_s(status_code::internal_error);
    EXPECT_FALSE(static_cast<bool>(err_s));
}

TEST(StatusCovTest, Equality) {
    status s1(status_code::timeout);
    status s2(status_code::timeout);
    status s3(status_code::internal_error);

    EXPECT_EQ(s1, s2);
    EXPECT_NE(s1, s3);
    EXPECT_EQ(s1, status_code::timeout);
    EXPECT_NE(s1, status_code::ok);
}

TEST(StatusCovTest, ToStringOk) {
    status s;
    EXPECT_EQ(s.to_string(), "ok");
}

TEST(StatusCovTest, ToStringError) {
    status s(status_code::timeout, 2, "deadline exceeded");
    auto str = s.to_string();
    EXPECT_NE(str.find("communication"), std::string::npos);
    EXPECT_NE(str.find("rank 2"), std::string::npos);
    EXPECT_NE(str.find("deadline exceeded"), std::string::npos);
}

TEST(StatusCovTest, CategoryMethod) {
    status s(status_code::allocation_failed);
    EXPECT_EQ(s.category(), "memory");
}

TEST(StatusCovTest, OkStatusFactory) {
    auto s = ok_status();
    EXPECT_TRUE(s.ok());
}

TEST(StatusCovTest, ErrorStatusFactory) {
    auto s = error_status(status_code::internal_error, 0, "bad");
    EXPECT_TRUE(s.is_error());
    EXPECT_EQ(s.rank(), 0);
    EXPECT_EQ(s.message(), "bad");
}

// =============================================================================
// Result<T> Tests
// =============================================================================

TEST(ResultCovTest, DefaultConstructHasValue) {
    result<int> r;
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 0);
}

TEST(ResultCovTest, ConstructWithValue) {
    result<int> r(42);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(ResultCovTest, ConstructWithError) {
    result<int> r(status_code::internal_error);
    EXPECT_TRUE(r.has_error());
    EXPECT_FALSE(r.has_value());
}

TEST(ResultCovTest, ConstructFromStatus) {
    status err(status_code::timeout);
    result<int> r(err);
    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::timeout);
}

TEST(ResultCovTest, BoolConversion) {
    result<int> ok(42);
    EXPECT_TRUE(static_cast<bool>(ok));

    result<int> err(status_code::internal_error);
    EXPECT_FALSE(static_cast<bool>(err));
}

TEST(ResultCovTest, ValueOr) {
    result<int> ok(42);
    EXPECT_EQ(ok.value_or(0), 42);

    result<int> err(status_code::internal_error);
    EXPECT_EQ(err.value_or(99), 99);
}

TEST(ResultCovTest, DereferenceOperator) {
    result<int> r(42);
    EXPECT_EQ(*r, 42);
}

TEST(ResultCovTest, ArrowOperator) {
    result<std::string> r(std::string("hello"));
    EXPECT_EQ(r->size(), 5u);
}

TEST(ResultCovTest, InPlaceConstructor) {
    result<std::string> r(std::in_place, 5, 'x');
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), "xxxxx");
}

// =============================================================================
// Result<T> Monadic Operations
// =============================================================================

TEST(ResultMonadicCovTest, MapSuccess) {
    result<int> r(10);
    auto mapped = r.map([](int v) { return v * 2; });
    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 20);
}

TEST(ResultMonadicCovTest, MapError) {
    result<int> r(status_code::internal_error);
    auto mapped = r.map([](int v) { return v * 2; });
    EXPECT_TRUE(mapped.has_error());
}

TEST(ResultMonadicCovTest, MapTypeChange) {
    result<int> r(42);
    auto mapped = r.map([](int v) { return std::to_string(v); });
    static_assert(std::is_same_v<decltype(mapped), result<std::string>>);
    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), "42");
}

TEST(ResultMonadicCovTest, AndThenSuccess) {
    result<int> r(5);
    auto chained = r.and_then([](int v) -> result<int> {
        return result<int>(v + 10);
    });
    ASSERT_TRUE(chained.has_value());
    EXPECT_EQ(chained.value(), 15);
}

TEST(ResultMonadicCovTest, AndThenError) {
    result<int> r(status_code::internal_error);
    auto chained = r.and_then([](int v) -> result<int> {
        return result<int>(v + 10);
    });
    EXPECT_TRUE(chained.has_error());
}

TEST(ResultMonadicCovTest, AndThenChainedError) {
    result<int> r(5);
    auto chained = r.and_then([](int) -> result<int> {
        return status_code::timeout;
    });
    EXPECT_TRUE(chained.has_error());
    EXPECT_EQ(chained.error().code(), status_code::timeout);
}

TEST(ResultMonadicCovTest, TransformErrorSuccess) {
    result<int> r(42);
    auto transformed = r.transform_error([](status s) {
        return status(status_code::unknown_error, s.rank(), "wrapped: " + s.message());
    });
    EXPECT_TRUE(transformed.has_value());
    EXPECT_EQ(transformed.value(), 42);
}

TEST(ResultMonadicCovTest, TransformErrorOnError) {
    result<int> r(status(status_code::timeout, no_rank, "orig"));
    auto transformed = r.transform_error([](status s) {
        return status(status_code::unknown_error, s.rank(), "wrapped: " + s.message());
    });
    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::unknown_error);
}

TEST(ResultMonadicCovTest, OrElseSuccess) {
    result<int> r(42);
    int val = r.or_else([](const status&) { return 0; });
    EXPECT_EQ(val, 42);
}

TEST(ResultMonadicCovTest, OrElseError) {
    result<int> r(status_code::internal_error);
    int val = r.or_else([](const status&) { return 99; });
    EXPECT_EQ(val, 99);
}

// =============================================================================
// Result<T> Factory Methods
// =============================================================================

TEST(ResultFactoryCovTest, SuccessFactory) {
    auto r = result<int>::success(42);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(ResultFactoryCovTest, SuccessInPlaceFactory) {
    auto r = result<std::string>::success_in_place(3, 'a');
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), "aaa");
}

TEST(ResultFactoryCovTest, FailureFactory) {
    auto r = result<int>::failure(status(status_code::timeout));
    EXPECT_TRUE(r.has_error());
}

// =============================================================================
// Result<void> Tests
// =============================================================================

TEST(ResultVoidCovTest, DefaultIsSuccess) {
    result<void> r;
    EXPECT_TRUE(r.has_value());
    EXPECT_FALSE(r.has_error());
}

TEST(ResultVoidCovTest, ConstructWithError) {
    result<void> r(status_code::internal_error);
    EXPECT_TRUE(r.has_error());
}

TEST(ResultVoidCovTest, BoolConversion) {
    result<void> ok;
    EXPECT_TRUE(static_cast<bool>(ok));
    result<void> err(status_code::timeout);
    EXPECT_FALSE(static_cast<bool>(err));
}

TEST(ResultVoidCovTest, MapSuccess) {
    result<void> r;
    auto mapped = r.map([] { return 42; });
    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 42);
}

TEST(ResultVoidCovTest, MapError) {
    result<void> r(status_code::internal_error);
    auto mapped = r.map([] { return 42; });
    EXPECT_TRUE(mapped.has_error());
}

TEST(ResultVoidCovTest, AndThenSuccess) {
    result<void> r;
    auto chained = r.and_then([] { return result<int>(42); });
    ASSERT_TRUE(chained.has_value());
    EXPECT_EQ(chained.value(), 42);
}

TEST(ResultVoidCovTest, AndThenError) {
    result<void> r(status_code::internal_error);
    auto chained = r.and_then([] { return result<int>(42); });
    EXPECT_TRUE(chained.has_error());
}

TEST(ResultVoidCovTest, TransformErrorOnError) {
    result<void> r(status(status_code::timeout));
    auto transformed = r.transform_error([](status) {
        return status(status_code::unknown_error);
    });
    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::unknown_error);
}

TEST(ResultVoidCovTest, OrElseOnError) {
    result<void> r(status(status_code::timeout));
    bool called = false;
    r.or_else([&](const status& s) {
        called = true;
        EXPECT_EQ(s.code(), status_code::timeout);
    });
    EXPECT_TRUE(called);
}

TEST(ResultVoidCovTest, OrElseNotCalledOnSuccess) {
    result<void> r;
    bool called = false;
    r.or_else([&](const status&) { called = true; });
    EXPECT_FALSE(called);
}

TEST(ResultVoidCovTest, SuccessFactory) {
    auto r = result<void>::success();
    EXPECT_TRUE(r.has_value());
}

TEST(ResultVoidCovTest, FailureFactory) {
    auto r = result<void>::failure(status(status_code::timeout));
    EXPECT_TRUE(r.has_error());
}

// =============================================================================
// Free Factory Functions
// =============================================================================

TEST(ResultFreeFunctionsTest, MakeResult) {
    auto r = make_result(42);
    static_assert(std::is_same_v<decltype(r), result<int>>);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(ResultFreeFunctionsTest, MakeOkResult) {
    auto r = make_ok_result();
    EXPECT_TRUE(r.has_value());
}

TEST(ResultFreeFunctionsTest, MakeErrorResultFromStatus) {
    auto r = make_error_result<int>(status(status_code::timeout));
    EXPECT_TRUE(r.has_error());
}

TEST(ResultFreeFunctionsTest, MakeErrorResultFromCode) {
    auto r = make_error_result<int>(status_code::timeout);
    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::timeout);
}

TEST(ResultFreeFunctionsTest, MakeErrorWithMessage) {
    auto r = make_error<int>(status_code::timeout, "deadline exceeded");
    EXPECT_TRUE(r.has_error());
}

TEST(ResultFreeFunctionsTest, MakeErrorWithRankAndMessage) {
    auto r = make_error<int>(status_code::timeout, 3, "on rank 3");
    EXPECT_TRUE(r.has_error());
}

// =============================================================================
// Error Type Tests
// =============================================================================

TEST(ErrorTypeCovTest, ConstructFromStatus) {
    status s(status_code::timeout, 2, "test");
    error err(s);
    EXPECT_EQ(err.code(), status_code::timeout);
    EXPECT_EQ(err.rank(), 2);
    EXPECT_EQ(err.message(), "test");
}

TEST(ErrorTypeCovTest, ConstructFromComponents) {
    error err(status_code::allocation_failed, 1, "oom");
    EXPECT_EQ(err.code(), status_code::allocation_failed);
    EXPECT_EQ(err.rank(), 1);
    EXPECT_EQ(err.message(), "oom");
}

TEST(ErrorTypeCovTest, SourceLocation) {
    error err(status_code::internal_error);
    // Should capture file and line of this call site
    EXPECT_NE(err.file_name(), nullptr);
    EXPECT_GT(err.line(), 0u);
    EXPECT_NE(err.function_name(), nullptr);
}

TEST(ErrorTypeCovTest, ToStringContainsContext) {
    error err(status_code::timeout, no_rank, "msg");
    auto str = err.to_string();
    EXPECT_NE(str.find("at"), std::string::npos);
    EXPECT_NE(str.find("in"), std::string::npos);
}

TEST(ErrorTypeCovTest, GetStatus) {
    error err(status_code::timeout);
    const status& s = err.get_status();
    EXPECT_EQ(s.code(), status_code::timeout);
}

TEST(ErrorTypeCovTest, ImplicitConversionToStatus) {
    error err(status_code::timeout);
    const status& s = err;  // implicit conversion
    EXPECT_EQ(s.code(), status_code::timeout);
}

// =============================================================================
// make_error Free Function Tests
// =============================================================================

TEST(MakeErrorFreeTest, WithCodeAndMessage) {
    auto err = make_error(status_code::timeout, "deadline");
    EXPECT_EQ(err.code(), status_code::timeout);
    EXPECT_EQ(err.message(), "deadline");
}

TEST(MakeErrorFreeTest, WithCodeRankMessage) {
    auto err = make_error(status_code::send_failed, 5, "failed on rank 5");
    EXPECT_EQ(err.code(), status_code::send_failed);
    EXPECT_EQ(err.rank(), 5);
}

TEST(MakeErrorFreeTest, WithCodeOnly) {
    auto err = make_error(status_code::internal_error);
    EXPECT_EQ(err.code(), status_code::internal_error);
    EXPECT_TRUE(err.message().empty());
}

// =============================================================================
// Error Code Alias Test
// =============================================================================

TEST(ErrorCodeAliasTest, ErrorCodeIsStatusCode) {
    static_assert(std::is_same_v<error_code, status_code>);
    SUCCEED();
}

// =============================================================================
// Collective Error Tests
// =============================================================================

TEST(CollectiveErrorCovTest, DefaultNoErrors) {
    collective_error ce;
    EXPECT_FALSE(ce.has_errors());
    EXPECT_TRUE(ce.all_succeeded());
    EXPECT_EQ(ce.error_count(), 0u);
    EXPECT_EQ(ce.num_ranks(), 0);
}

TEST(CollectiveErrorCovTest, ConstructWithNumRanks) {
    collective_error ce(4);
    EXPECT_EQ(ce.num_ranks(), 4);
    EXPECT_TRUE(ce.all_succeeded());
}

TEST(CollectiveErrorCovTest, AddError) {
    collective_error ce(4);
    ce.add_error(1, status(status_code::timeout));
    EXPECT_TRUE(ce.has_errors());
    EXPECT_FALSE(ce.all_succeeded());
    EXPECT_EQ(ce.error_count(), 1u);
}

TEST(CollectiveErrorCovTest, AddMultipleErrors) {
    collective_error ce(4);
    ce.add_error(1, status(status_code::timeout));
    ce.add_error(3, status(status_code::send_failed));
    EXPECT_EQ(ce.error_count(), 2u);
}

TEST(CollectiveErrorCovTest, AddOkStatusDoesNotError) {
    collective_error ce(4);
    ce.add_error(0, ok_status());
    EXPECT_TRUE(ce.all_succeeded());
    EXPECT_EQ(ce.error_count(), 0u);
}

TEST(CollectiveErrorCovTest, FirstError) {
    collective_error ce(4);
    ce.add_error(1, status(status_code::timeout));
    ce.add_error(2, status(status_code::send_failed));
    auto first = ce.first_error();
    EXPECT_EQ(first.code(), status_code::timeout);
}

TEST(CollectiveErrorCovTest, FirstErrorWhenNoErrors) {
    collective_error ce(4);
    auto first = ce.first_error();
    EXPECT_TRUE(first.ok());
}

TEST(CollectiveErrorCovTest, MostCommonError) {
    collective_error ce(4);
    ce.add_error(0, status(status_code::timeout));
    ce.add_error(1, status(status_code::timeout));
    ce.add_error(2, status(status_code::send_failed));
    EXPECT_EQ(ce.most_common_error(), status_code::timeout);
}

TEST(CollectiveErrorCovTest, MostCommonErrorNoErrors) {
    collective_error ce(4);
    EXPECT_EQ(ce.most_common_error(), status_code::ok);
}

TEST(CollectiveErrorCovTest, SummaryAllSucceeded) {
    collective_error ce(4);
    auto summary = ce.summary();
    EXPECT_TRUE(summary.ok());
}

TEST(CollectiveErrorCovTest, SummarySomeFailed) {
    collective_error ce(4);
    ce.add_error(1, status(status_code::timeout));
    auto summary = ce.summary();
    EXPECT_TRUE(summary.is_error());
    auto msg = summary.message();
    EXPECT_NE(msg.find("1 of 4"), std::string::npos);
}

TEST(CollectiveErrorCovTest, ToStringAllSucceeded) {
    collective_error ce(4);
    auto str = ce.to_string();
    EXPECT_NE(str.find("succeeded"), std::string::npos);
}

TEST(CollectiveErrorCovTest, ToStringSomeFailed) {
    collective_error ce(4);
    ce.add_error(2, status(status_code::timeout, 2, "timed out"));
    auto str = ce.to_string();
    EXPECT_NE(str.find("failed"), std::string::npos);
    EXPECT_NE(str.find("rank 2"), std::string::npos);
}

TEST(CollectiveErrorCovTest, ErrorsAccessor) {
    collective_error ce(3);
    ce.add_error(0, status(status_code::timeout));
    ce.add_error(2, status(status_code::send_failed));

    const auto& errors = ce.errors();
    ASSERT_EQ(errors.size(), 2u);
    EXPECT_EQ(errors[0].rank, 0);
    EXPECT_EQ(errors[1].rank, 2);
    EXPECT_TRUE(errors[0].has_error());
}

// =============================================================================
// Failure Handler Tests
// =============================================================================

TEST(FailureHandlerTest, CategorizeTimeout) {
    EXPECT_EQ(categorize_failure(status_code::timeout), failure_category::transient);
}

TEST(FailureHandlerTest, CategorizeConnectionLost) {
    EXPECT_EQ(categorize_failure(status_code::connection_lost), failure_category::transient);
}

TEST(FailureHandlerTest, CategorizeCommunication) {
    EXPECT_EQ(categorize_failure(status_code::send_failed), failure_category::communication);
}

TEST(FailureHandlerTest, CategorizeMemory) {
    EXPECT_EQ(categorize_failure(status_code::allocation_failed), failure_category::resource);
}

TEST(FailureHandlerTest, CategorizeInvalidFormat) {
    EXPECT_EQ(categorize_failure(status_code::invalid_format), failure_category::corruption);
}

TEST(FailureHandlerTest, CategorizeSerializationNonRecoverable) {
    EXPECT_EQ(categorize_failure(status_code::serialize_failed),
              failure_category::non_recoverable);
}

}  // namespace dtl::test
