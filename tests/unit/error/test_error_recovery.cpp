// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_error_recovery.cpp
/// @brief R10.2: Error recovery and propagation tests
/// @details Tests that error states propagate correctly through the
///          result<T> monadic API and that error information survives
///          copy/move operations and aggregation patterns.

#include <dtl/error/result.hpp>
#include <dtl/error/status.hpp>
#include <dtl/error/error_type.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Error Propagation Through Function Calls
// =============================================================================

namespace {

// Simulates a chain of operations that propagate errors
result<int> step1_succeeds(int value) {
    return result<int>{value * 2};
}

result<int> step1_fails() {
    return make_error<int>(status_code::communication_error, "step1 failed");
}

result<int> step2_succeeds(int value) {
    return result<int>{value + 10};
}

result<int> step3_succeeds(int value) {
    return result<int>{value * 3};
}

}  // namespace

TEST(ErrorRecoveryTest, ErrorPropagatesThroughMultipleFunctionCalls) {
    // Create an error and propagate it through multiple calls using and_then
    auto r = step1_fails();
    ASSERT_TRUE(r.has_error());

    auto r2 = r.and_then([](int v) { return step2_succeeds(v); });
    ASSERT_TRUE(r2.has_error());
    EXPECT_EQ(r2.error().code(), status_code::communication_error);

    auto r3 = r2.and_then([](int v) { return step3_succeeds(v); });
    ASSERT_TRUE(r3.has_error());
    EXPECT_EQ(r3.error().code(), status_code::communication_error);
}

TEST(ErrorRecoveryTest, SuccessChainsThroughAndThen) {
    auto r = step1_succeeds(5)
        .and_then([](int v) { return step2_succeeds(v); })
        .and_then([](int v) { return step3_succeeds(v); });

    ASSERT_TRUE(r.has_value());
    // 5 * 2 = 10, + 10 = 20, * 3 = 60
    EXPECT_EQ(r.value(), 60);
}

TEST(ErrorRecoveryTest, ErrorStopsChainAtFailurePoint) {
    bool step2_called = false;
    bool step3_called = false;

    auto r = step1_fails()
        .and_then([&](int v) { step2_called = true; return step2_succeeds(v); })
        .and_then([&](int v) { step3_called = true; return step3_succeeds(v); });

    EXPECT_FALSE(step2_called);
    EXPECT_FALSE(step3_called);
    ASSERT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::communication_error);
}

// =============================================================================
// Error Code Preservation Through Monadic Operations
// =============================================================================

TEST(ErrorRecoveryTest, MapPreservesErrorCode) {
    auto r = make_error<int>(status_code::allocation_failed, "no memory");
    auto mapped = r.map([](int v) { return v * 2; });

    ASSERT_TRUE(mapped.has_error());
    EXPECT_EQ(mapped.error().code(), status_code::allocation_failed);
}

TEST(ErrorRecoveryTest, MapPreservesErrorMessage) {
    auto r = make_error<int>(status_code::out_of_bounds, "index 99 out of range");
    auto mapped = r.map([](int v) { return static_cast<double>(v); });

    ASSERT_TRUE(mapped.has_error());
    EXPECT_EQ(mapped.error().code(), status_code::out_of_bounds);
    EXPECT_NE(mapped.error().message().find("index 99"), std::string::npos);
}

TEST(ErrorRecoveryTest, AndThenPreservesErrorCode) {
    auto r = make_error<int>(status_code::timeout, "operation timed out");
    auto chained = r.and_then([](int v) -> result<std::string> {
        return result<std::string>{std::to_string(v)};
    });

    ASSERT_TRUE(chained.has_error());
    EXPECT_EQ(chained.error().code(), status_code::timeout);
}

TEST(ErrorRecoveryTest, AndThenPreservesErrorMessage) {
    auto r = make_error<int>(status_code::timeout, "custom timeout msg");
    auto chained = r.and_then([](int v) { return step2_succeeds(v); });

    ASSERT_TRUE(chained.has_error());
    EXPECT_NE(chained.error().message().find("custom timeout msg"), std::string::npos);
}

TEST(ErrorRecoveryTest, TransformErrorChangesErrorCode) {
    auto r = make_error<int>(status_code::communication_error, "comm fail");

    auto transformed = r.transform_error([](const status& s) {
        return status{status_code::internal_error, s.rank(), "wrapped: " + s.message()};
    });

    ASSERT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::internal_error);
    EXPECT_NE(transformed.error().message().find("wrapped:"), std::string::npos);
    EXPECT_NE(transformed.error().message().find("comm fail"), std::string::npos);
}

TEST(ErrorRecoveryTest, MapChangesValueType) {
    auto r = step1_succeeds(5);
    auto mapped = r.map([](int v) { return std::to_string(v); });

    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), "10");  // 5 * 2 = 10
}

// =============================================================================
// Error Status Survives Copy/Move Operations
// =============================================================================

TEST(ErrorRecoveryTest, ErrorSurvivesCopyConstruction) {
    auto original = make_error<int>(status_code::serialization_error, "bad format");
    auto copy = original;  // NOLINT(performance-unnecessary-copy-initialization)

    ASSERT_TRUE(copy.has_error());
    EXPECT_EQ(copy.error().code(), status_code::serialization_error);
    EXPECT_EQ(copy.error().message(), "bad format");

    // Original should be unchanged
    ASSERT_TRUE(original.has_error());
    EXPECT_EQ(original.error().code(), status_code::serialization_error);
}

TEST(ErrorRecoveryTest, ErrorSurvivesMoveConstruction) {
    auto original = make_error<int>(status_code::backend_not_available, "no GPU");
    auto moved = std::move(original);

    ASSERT_TRUE(moved.has_error());
    EXPECT_EQ(moved.error().code(), status_code::backend_not_available);
    EXPECT_EQ(moved.error().message(), "no GPU");
}

TEST(ErrorRecoveryTest, ErrorSurvivesCopyAssignment) {
    auto r1 = result<int>{42};
    auto r2 = make_error<int>(status_code::precondition_failed, "pre fail");

    r1 = r2;
    ASSERT_TRUE(r1.has_error());
    EXPECT_EQ(r1.error().code(), status_code::precondition_failed);
    EXPECT_EQ(r1.error().message(), "pre fail");
}

TEST(ErrorRecoveryTest, ErrorSurvivesMoveAssignment) {
    auto r1 = result<int>{42};
    auto r2 = make_error<int>(status_code::convergence_failed, "no converge");

    r1 = std::move(r2);
    ASSERT_TRUE(r1.has_error());
    EXPECT_EQ(r1.error().code(), status_code::convergence_failed);
    EXPECT_EQ(r1.error().message(), "no converge");
}

TEST(ErrorRecoveryTest, StatusSurvivesCopyConstruction) {
    status original{status_code::mpi_error, 3, "MPI_Allreduce failed"};
    status copy = original;  // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(copy.code(), status_code::mpi_error);
    EXPECT_EQ(copy.rank(), 3);
    EXPECT_EQ(copy.message(), "MPI_Allreduce failed");
}

TEST(ErrorRecoveryTest, StatusSurvivesMoveConstruction) {
    status original{status_code::cuda_error, 0, "cudaMalloc failed"};
    status moved = std::move(original);

    EXPECT_EQ(moved.code(), status_code::cuda_error);
    EXPECT_EQ(moved.rank(), 0);
    EXPECT_EQ(moved.message(), "cudaMalloc failed");
}

TEST(ErrorRecoveryTest, ErrorObjectSurvivesCopy) {
    error original{status_code::nccl_error, 2, "NCCL allreduce failed"};
    error copy = original;  // NOLINT(performance-unnecessary-copy-initialization)

    EXPECT_EQ(copy.code(), status_code::nccl_error);
    EXPECT_EQ(copy.rank(), 2);
    EXPECT_EQ(copy.message(), "NCCL allreduce failed");
}

TEST(ErrorRecoveryTest, ErrorObjectSurvivesMove) {
    error original{status_code::shmem_error, 1, "shmem_put failed"};
    error moved = std::move(original);

    EXPECT_EQ(moved.code(), status_code::shmem_error);
    EXPECT_EQ(moved.rank(), 1);
    EXPECT_EQ(moved.message(), "shmem_put failed");
}

// =============================================================================
// Error Aggregation Patterns
// =============================================================================

TEST(ErrorRecoveryTest, CollectErrorsFromMultipleOperations) {
    // Simulate collecting errors from multiple operations
    std::vector<result<int>> results;
    results.push_back(result<int>{10});
    results.push_back(make_error<int>(status_code::send_failed, "rank 1 send failed"));
    results.push_back(result<int>{30});
    results.push_back(make_error<int>(status_code::recv_failed, "rank 3 recv failed"));
    results.push_back(result<int>{50});

    // Count errors
    int error_count = 0;
    int success_count = 0;
    for (const auto& r : results) {
        if (r.has_error()) {
            ++error_count;
        } else {
            ++success_count;
        }
    }

    EXPECT_EQ(error_count, 2);
    EXPECT_EQ(success_count, 3);
}

TEST(ErrorRecoveryTest, FirstErrorFromMultipleOperations) {
    // Pattern: find the first error from a batch of operations
    std::vector<result<int>> results;
    results.push_back(result<int>{10});
    results.push_back(result<int>{20});
    results.push_back(make_error<int>(status_code::barrier_failed, "barrier timeout"));
    results.push_back(make_error<int>(status_code::reduce_failed, "reduce failed"));

    // Find first error
    const status* first_error = nullptr;
    for (const auto& r : results) {
        if (r.has_error()) {
            first_error = &r.error();
            break;
        }
    }

    ASSERT_NE(first_error, nullptr);
    EXPECT_EQ(first_error->code(), status_code::barrier_failed);
    EXPECT_EQ(first_error->message(), "barrier timeout");
}

TEST(ErrorRecoveryTest, AggregateSuccessValuesSkippingErrors) {
    // Pattern: accumulate only successful values, track errors separately
    std::vector<result<int>> results;
    results.push_back(result<int>{10});
    results.push_back(make_error<int>(status_code::send_failed, "err"));
    results.push_back(result<int>{20});
    results.push_back(result<int>{30});
    results.push_back(make_error<int>(status_code::recv_failed, "err"));

    int sum = 0;
    std::vector<status_code> error_codes;
    for (const auto& r : results) {
        if (r.has_value()) {
            sum += r.value();
        } else {
            error_codes.push_back(r.error().code());
        }
    }

    EXPECT_EQ(sum, 60);
    ASSERT_EQ(error_codes.size(), 2u);
    EXPECT_EQ(error_codes[0], status_code::send_failed);
    EXPECT_EQ(error_codes[1], status_code::recv_failed);
}

TEST(ErrorRecoveryTest, OrElseRecoveryPattern) {
    // Use or_else to provide fallback values on error
    auto r = make_error<int>(status_code::timeout, "timed out");

    int recovered = r.or_else([](const status&) { return -1; });
    EXPECT_EQ(recovered, -1);
}

TEST(ErrorRecoveryTest, OrElsePassesThroughSuccess) {
    auto r = result<int>{42};

    int value = r.or_else([](const status&) { return -1; });
    EXPECT_EQ(value, 42);
}

TEST(ErrorRecoveryTest, ValueOrDefaultOnError) {
    auto r = make_error<int>(status_code::communication_error, "err");
    EXPECT_EQ(r.value_or(999), 999);
}

TEST(ErrorRecoveryTest, ValueOrReturnsValueOnSuccess) {
    auto r = result<int>{42};
    EXPECT_EQ(r.value_or(999), 42);
}

// =============================================================================
// result<void> Error Propagation
// =============================================================================

TEST(ErrorRecoveryTest, VoidResultErrorPropagation) {
    result<void> r{status_code::invalid_state};
    ASSERT_TRUE(r.has_error());

    // Chain to value-producing result via map
    auto mapped = r.map([]() { return 42; });
    ASSERT_TRUE(mapped.has_error());
    EXPECT_EQ(mapped.error().code(), status_code::invalid_state);
}

TEST(ErrorRecoveryTest, VoidResultSuccessChaining) {
    result<void> r{};  // success
    ASSERT_TRUE(r.has_value());

    auto mapped = r.map([]() { return 42; });
    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 42);
}

TEST(ErrorRecoveryTest, VoidResultAndThenPropagatesError) {
    result<void> r{status_code::memory_error};

    auto chained = r.and_then([]() -> result<int> {
        return result<int>{42};
    });

    ASSERT_TRUE(chained.has_error());
    EXPECT_EQ(chained.error().code(), status_code::memory_error);
}

TEST(ErrorRecoveryTest, VoidResultTransformError) {
    result<void> r{status_code::communication_error};

    auto transformed = r.transform_error([](const status& s) {
        return status{status_code::internal_error, s.rank(),
                      "wrapped: " + s.message()};
    });

    ASSERT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::internal_error);
}

// =============================================================================
// Status Code Category Preservation
// =============================================================================

TEST(ErrorRecoveryTest, CategoryPreservedThroughChain) {
    auto r = make_error<int>(status_code::cuda_error, "device error");

    // Propagate through and_then
    auto r2 = r.and_then([](int v) { return step2_succeeds(v); });

    EXPECT_EQ(r2.error().category(), "backend");
    EXPECT_EQ(r2.error().code(), status_code::cuda_error);
}

TEST(ErrorRecoveryTest, AllCategoriesPreserved) {
    // Test a representative error from each category
    struct CategoryTest {
        status_code code;
        std::string_view expected_category;
    };

    std::vector<CategoryTest> tests = {
        {status_code::communication_error, "communication"},
        {status_code::allocation_failed, "memory"},
        {status_code::serialization_error, "serialization"},
        {status_code::out_of_bounds, "bounds"},
        {status_code::backend_error, "backend"},
        {status_code::algorithm_error, "algorithm"},
        {status_code::consistency_error, "consistency"},
        {status_code::internal_error, "internal"},
    };

    for (const auto& test : tests) {
        auto r = make_error<int>(test.code, "test");
        auto chained = r.and_then([](int v) { return step2_succeeds(v); });
        EXPECT_EQ(chained.error().category(), test.expected_category)
            << "Category mismatch for code " << static_cast<int>(test.code);
    }
}

}  // namespace dtl::test
