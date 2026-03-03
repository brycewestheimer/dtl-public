// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_phase05_status_codes.cpp
/// @brief Phase 05 tests: status_code_name completeness + C/C++ parity
/// @details Tests for T01 (missing name entries) and T02 (C-only codes).

#include <dtl/error/status.hpp>

#include <gtest/gtest.h>

#include <string_view>

namespace dtl::test {

// =============================================================================
// T01: All status_code enum values have non-empty names
// =============================================================================

TEST(Phase05StatusCodeNameTest, KeyNotFoundHasName) {
    EXPECT_EQ(status_code_name(status_code::key_not_found), "key_not_found");
}

TEST(Phase05StatusCodeNameTest, OutOfRangeHasName) {
    EXPECT_EQ(status_code_name(status_code::out_of_range), "out_of_range");
}

TEST(Phase05StatusCodeNameTest, AllEnumValuesHaveNonEmptyNames) {
    // Exhaustive check of every status_code enum value
    auto check = [](status_code code, std::string_view expected_substring) {
        auto name = status_code_name(code);
        EXPECT_NE(name, "unknown")
            << "status_code " << static_cast<int>(code)
            << " has no name entry (returned 'unknown')";
        EXPECT_FALSE(name.empty())
            << "status_code " << static_cast<int>(code) << " returned empty name";
        if (!expected_substring.empty()) {
            EXPECT_EQ(name, expected_substring)
                << "Mismatch for code " << static_cast<int>(code);
        }
    };

    // Success
    check(status_code::ok, "ok");

    // Non-error sentinels
    check(status_code::not_found, "not_found");
    check(status_code::end_iterator, "end_iterator");

    // Communication
    check(status_code::communication_error, "communication_error");
    check(status_code::send_failed, "send_failed");
    check(status_code::recv_failed, "recv_failed");
    check(status_code::broadcast_failed, "broadcast_failed");
    check(status_code::reduce_failed, "reduce_failed");
    check(status_code::barrier_failed, "barrier_failed");
    check(status_code::timeout, "timeout");
    check(status_code::canceled, "canceled");
    check(status_code::connection_lost, "connection_lost");
    check(status_code::rank_failure, "rank_failure");
    check(status_code::collective_failure, "collective_failure");

    // Memory
    check(status_code::memory_error, "memory_error");
    check(status_code::allocation_failed, "allocation_failed");
    check(status_code::out_of_memory, "out_of_memory");
    check(status_code::invalid_pointer, "invalid_pointer");
    check(status_code::memory_transfer_failed, "memory_transfer_failed");
    check(status_code::device_memory_error, "device_memory_error");

    // Serialization
    check(status_code::serialization_error, "serialization_error");
    check(status_code::serialize_failed, "serialize_failed");
    check(status_code::deserialize_failed, "deserialize_failed");
    check(status_code::buffer_too_small, "buffer_too_small");
    check(status_code::invalid_format, "invalid_format");

    // Bounds
    check(status_code::bounds_error, "bounds_error");
    check(status_code::out_of_bounds, "out_of_bounds");
    check(status_code::invalid_index, "invalid_index");
    check(status_code::invalid_rank, "invalid_rank");
    check(status_code::dimension_mismatch, "dimension_mismatch");
    check(status_code::extent_mismatch, "extent_mismatch");
    check(status_code::key_not_found, "key_not_found");
    check(status_code::out_of_range, "out_of_range");
    check(status_code::invalid_argument, "invalid_argument");
    check(status_code::null_pointer, "null_pointer");
    check(status_code::not_supported, "not_supported");

    // Backend
    check(status_code::backend_error, "backend_error");
    check(status_code::backend_not_available, "backend_not_available");
    check(status_code::backend_init_failed, "backend_init_failed");
    check(status_code::cuda_error, "cuda_error");
    check(status_code::hip_error, "hip_error");
    check(status_code::mpi_error, "mpi_error");
    check(status_code::nccl_error, "nccl_error");
    check(status_code::shmem_error, "shmem_error");

    // Algorithm (algorithm_error and operation_failed share value 600)
    check(status_code::algorithm_error, "algorithm_error");
    check(status_code::precondition_failed, "precondition_failed");
    check(status_code::postcondition_failed, "postcondition_failed");
    check(status_code::convergence_failed, "convergence_failed");

    // Consistency
    check(status_code::consistency_error, "consistency_error");
    check(status_code::consistency_violation, "consistency_violation");
    check(status_code::structural_invalidation, "structural_invalidation");

    // Internal
    check(status_code::internal_error, "internal_error");
    check(status_code::not_implemented, "not_implemented");
    check(status_code::invalid_state, "invalid_state");
    check(status_code::unknown_error, "unknown_error");
}

// =============================================================================
// T02: C++ enum values match C defines numerically
// =============================================================================

TEST(Phase05StatusCodeParityTest, NotFoundMatchesCDefine) {
    // DTL_NOT_FOUND = 1
    EXPECT_EQ(static_cast<int>(status_code::not_found), 1);
}

TEST(Phase05StatusCodeParityTest, EndIteratorMatchesCDefine) {
    // DTL_END = 2
    EXPECT_EQ(static_cast<int>(status_code::end_iterator), 2);
}

TEST(Phase05StatusCodeParityTest, NullPointerMatchesCDefine) {
    // DTL_ERROR_NULL_POINTER = 411
    EXPECT_EQ(static_cast<int>(status_code::null_pointer), 411);
}

TEST(Phase05StatusCodeParityTest, NcclErrorMatchesCDefine) {
    // DTL_ERROR_NCCL = 540
    EXPECT_EQ(static_cast<int>(status_code::nccl_error), 540);
}

TEST(Phase05StatusCodeParityTest, ShmemErrorMatchesCDefine) {
    // DTL_ERROR_SHMEM = 550
    EXPECT_EQ(static_cast<int>(status_code::shmem_error), 550);
}

TEST(Phase05StatusCodeParityTest, ExistingCodesStillMatch) {
    // Verify a sample of previously existing codes haven't shifted
    EXPECT_EQ(static_cast<int>(status_code::ok), 0);
    EXPECT_EQ(static_cast<int>(status_code::communication_error), 100);
    EXPECT_EQ(static_cast<int>(status_code::memory_error), 200);
    EXPECT_EQ(static_cast<int>(status_code::serialization_error), 300);
    EXPECT_EQ(static_cast<int>(status_code::bounds_error), 400);
    EXPECT_EQ(static_cast<int>(status_code::key_not_found), 406);
    EXPECT_EQ(static_cast<int>(status_code::out_of_range), 407);
    EXPECT_EQ(static_cast<int>(status_code::invalid_argument), 410);
    EXPECT_EQ(static_cast<int>(status_code::backend_error), 500);
    EXPECT_EQ(static_cast<int>(status_code::cuda_error), 510);
    EXPECT_EQ(static_cast<int>(status_code::hip_error), 520);
    EXPECT_EQ(static_cast<int>(status_code::mpi_error), 530);
    EXPECT_EQ(static_cast<int>(status_code::algorithm_error), 600);
    EXPECT_EQ(static_cast<int>(status_code::internal_error), 900);
    EXPECT_EQ(static_cast<int>(status_code::unknown_error), 999);
}

TEST(Phase05StatusCodeParityTest, NewCodesHaveCorrectNames) {
    EXPECT_EQ(status_code_name(status_code::not_found), "not_found");
    EXPECT_EQ(status_code_name(status_code::end_iterator), "end_iterator");
    EXPECT_EQ(status_code_name(status_code::null_pointer), "null_pointer");
    EXPECT_EQ(status_code_name(status_code::nccl_error), "nccl_error");
    EXPECT_EQ(status_code_name(status_code::shmem_error), "shmem_error");
}

TEST(Phase05StatusCodeParityTest, NewCodesHaveCorrectCategories) {
    EXPECT_EQ(status_category(status_code::not_found), "success");
    EXPECT_EQ(status_category(status_code::end_iterator), "success");
    EXPECT_EQ(status_category(status_code::null_pointer), "bounds");
    EXPECT_EQ(status_category(status_code::nccl_error), "backend");
    EXPECT_EQ(status_category(status_code::shmem_error), "backend");
}

}  // namespace dtl::test
