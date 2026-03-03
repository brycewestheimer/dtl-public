// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_request.cpp
/// @brief Unit tests for MPI request RAII wrapper
/// @details Tests mpi_request ownership semantics, move operations,
///          wait/test/cancel/release methods, and destructor cleanup.
///          These tests run without MPI (DTL_ENABLE_MPI=0) and verify
///          the structural behavior of the RAII wrapper.

#include <backends/mpi/mpi_request.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <gtest/gtest.h>
#include <utility>

namespace dtl::test {

// =============================================================================
// Basic Construction Tests
// =============================================================================

TEST(MpiRequestTest, DefaultConstructorCreatesInvalidRequest) {
    mpi::mpi_request req;
    EXPECT_FALSE(req.valid());
    EXPECT_FALSE(static_cast<bool>(req));
}

TEST(MpiRequestTest, ConstructFromNullRequestHandle) {
    request_handle handle;
    ASSERT_FALSE(handle.valid());

    mpi::mpi_request req(handle);
    EXPECT_FALSE(req.valid());
    EXPECT_EQ(handle.handle, nullptr);
}

TEST(MpiRequestTest, ConstructFromNullRawPointer) {
    mpi::mpi_request req(nullptr);
    EXPECT_FALSE(req.valid());
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(MpiRequestTest, MoveConstructorTransfersOwnership) {
    mpi::mpi_request req1;
    mpi::mpi_request req2(std::move(req1));

    // Both should be invalid since source was empty
    EXPECT_FALSE(req1.valid());
    EXPECT_FALSE(req2.valid());
}

TEST(MpiRequestTest, MoveAssignmentTransfersOwnership) {
    mpi::mpi_request req1;
    mpi::mpi_request req2;

    req2 = std::move(req1);

    EXPECT_FALSE(req1.valid());
    EXPECT_FALSE(req2.valid());
}

TEST(MpiRequestTest, SelfMoveAssignmentIsSafe) {
    mpi::mpi_request req;
    auto* addr = &req;
    // Suppress self-move warning; this is intentional
    req = std::move(*addr);
    EXPECT_FALSE(req.valid());
}

// =============================================================================
// Non-Copyable Verification (compile-time)
// =============================================================================

static_assert(!std::is_copy_constructible_v<mpi::mpi_request>,
              "mpi_request must not be copy-constructible");
static_assert(!std::is_copy_assignable_v<mpi::mpi_request>,
              "mpi_request must not be copy-assignable");

// =============================================================================
// Move-Only Verification (compile-time)
// =============================================================================

static_assert(std::is_move_constructible_v<mpi::mpi_request>,
              "mpi_request must be move-constructible");
static_assert(std::is_move_assignable_v<mpi::mpi_request>,
              "mpi_request must be move-assignable");

// =============================================================================
// Nothrow Verification (compile-time)
// =============================================================================

static_assert(std::is_nothrow_default_constructible_v<mpi::mpi_request>,
              "mpi_request default constructor must be noexcept");
static_assert(std::is_nothrow_move_constructible_v<mpi::mpi_request>,
              "mpi_request move constructor must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<mpi::mpi_request>,
              "mpi_request move assignment must be noexcept");

// =============================================================================
// Wait/Test/Cancel on Empty Request
// =============================================================================

TEST(MpiRequestTest, WaitOnEmptyRequestReturnsTrue) {
    mpi::mpi_request req;
    EXPECT_TRUE(req.wait());
    EXPECT_FALSE(req.valid());
}

TEST(MpiRequestTest, TestOnEmptyRequestReturnsTrue) {
    mpi::mpi_request req;
    EXPECT_TRUE(req.test());
    EXPECT_FALSE(req.valid());
}

TEST(MpiRequestTest, CancelOnEmptyRequestReturnsTrue) {
    mpi::mpi_request req;
    EXPECT_TRUE(req.cancel());
    EXPECT_FALSE(req.valid());
}

// =============================================================================
// Release Methods
// =============================================================================

TEST(MpiRequestTest, ReleaseOnEmptyReturnsNullptr) {
    mpi::mpi_request req;
    auto* ptr = req.release();
    EXPECT_EQ(ptr, nullptr);
    EXPECT_FALSE(req.valid());
}

TEST(MpiRequestTest, ReleaseToHandleOnEmptyReturnsInvalidHandle) {
    mpi::mpi_request req;
    auto handle = req.release_to_handle();
    EXPECT_FALSE(handle.valid());
    EXPECT_FALSE(req.valid());
}

// =============================================================================
// Ownership Transfer via request_handle
// =============================================================================

TEST(MpiRequestTest, ConstructFromRequestHandleNullifiesSource) {
    // Simulate a request handle with a non-null pointer.
    // In real usage this would be a heap-allocated MPI_Request.
    // We use a void* sentinel to avoid type mismatches between
    // native_handle_type (MPI_Request*) and arbitrary pointer types.
    request_handle handle;
    int dummy = 42;
    void* sentinel = &dummy;
    handle.handle = sentinel;
    ASSERT_TRUE(handle.valid());

    // Construction should take ownership and null out the source handle.
    // We immediately release to avoid the destructor trying to interpret
    // dummy as an MPI_Request (which would be UB with MPI enabled).
    mpi::mpi_request req(handle);
    EXPECT_FALSE(handle.valid());
    EXPECT_TRUE(req.valid());

    // Release to avoid destructor calling MPI functions on a non-MPI pointer.
    // Compare via void* to avoid type mismatch when MPI is enabled
    // (native_handle_type is MPI_Request*, not int*).
    auto* released = req.release();
    EXPECT_EQ(static_cast<void*>(released), sentinel);
    EXPECT_FALSE(req.valid());
}

// =============================================================================
// Destructor Safety (no crash on empty)
// =============================================================================

TEST(MpiRequestTest, DestructorOnEmptyRequestDoesNotCrash) {
    // Destructor is called at scope exit -- no assertion needed,
    // just verify no crash/hang
    { mpi::mpi_request req; }
    SUCCEED();
}

TEST(MpiRequestTest, MultipleEmptyRequestsDestructSafely) {
    {
        mpi::mpi_request req1;
        mpi::mpi_request req2;
        mpi::mpi_request req3;
    }
    SUCCEED();
}

// =============================================================================
// Move Chain Tests
// =============================================================================

TEST(MpiRequestTest, MoveChainPreservesInvariant) {
    mpi::mpi_request req1;
    mpi::mpi_request req2(std::move(req1));
    mpi::mpi_request req3(std::move(req2));

    EXPECT_FALSE(req1.valid());
    EXPECT_FALSE(req2.valid());
    EXPECT_FALSE(req3.valid());
}

TEST(MpiRequestTest, MoveAssignmentChainsWork) {
    mpi::mpi_request req1;
    mpi::mpi_request req2;
    mpi::mpi_request req3;

    req3 = std::move(req2);
    req2 = std::move(req1);

    EXPECT_FALSE(req1.valid());
    EXPECT_FALSE(req2.valid());
    EXPECT_FALSE(req3.valid());
}

// =============================================================================
// Release After Wait/Test/Cancel
// =============================================================================

TEST(MpiRequestTest, ReleaseAfterWaitReturnsNullptr) {
    mpi::mpi_request req;
    req.wait();
    auto* ptr = req.release();
    EXPECT_EQ(ptr, nullptr);
}

TEST(MpiRequestTest, ReleaseToHandleAfterCancelReturnsInvalid) {
    mpi::mpi_request req;
    req.cancel();
    auto handle = req.release_to_handle();
    EXPECT_FALSE(handle.valid());
}

}  // namespace dtl::test
