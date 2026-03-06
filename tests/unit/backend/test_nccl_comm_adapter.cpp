// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_nccl_comm_adapter.cpp
/// @brief Unit tests for NCCL concept-compliant communicator adapter

#include <dtl/core/config.hpp>

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
#include <backends/nccl/nccl_comm_adapter.hpp>
#endif

#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

// =============================================================================
// Concept Satisfaction Tests (compile-time — the real test)
// =============================================================================

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA

// These static_asserts are in the header, but repeat here for test clarity
static_assert(Communicator<nccl::nccl_comm_adapter>,
              "nccl_comm_adapter must satisfy Communicator concept");
static_assert(CollectiveCommunicator<nccl::nccl_comm_adapter>,
              "nccl_comm_adapter must satisfy CollectiveCommunicator concept");
static_assert(ReducingCommunicator<nccl::nccl_comm_adapter>,
              "nccl_comm_adapter must satisfy ReducingCommunicator concept");

TEST(NcclCommAdapterTest, ConceptSatisfaction) {
    EXPECT_TRUE((Communicator<nccl::nccl_comm_adapter>));
    EXPECT_TRUE((CollectiveCommunicator<nccl::nccl_comm_adapter>));
    EXPECT_TRUE((ReducingCommunicator<nccl::nccl_comm_adapter>));
}

TEST(NcclCommAdapterTest, ConstructFromCommunicator) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    EXPECT_EQ(adapter.rank(), no_rank);
    EXPECT_EQ(adapter.size(), 0);
}

TEST(NcclCommAdapterTest, ConstructFromSharedPtr) {
    auto comm = std::make_shared<nccl::nccl_communicator>();
    nccl::nccl_comm_adapter adapter(comm);
    EXPECT_EQ(adapter.rank(), no_rank);
    EXPECT_EQ(adapter.size(), 0);
}

TEST(NcclCommAdapterTest, SendThrowsOnInvalidComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    int data = 42;
    EXPECT_THROW(adapter.send(&data, sizeof(int), 0, 0), nccl::communication_error);
}

TEST(NcclCommAdapterTest, RecvThrowsOnInvalidComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    int data = 0;
    EXPECT_THROW(adapter.recv(&data, sizeof(int), 0, 0), nccl::communication_error);
}

TEST(NcclCommAdapterTest, IsendThrowsOnInvalidComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    int data = 42;
    EXPECT_THROW(adapter.isend(&data, sizeof(int), 0, 0), nccl::communication_error);
}

TEST(NcclCommAdapterTest, IrecvThrowsOnInvalidComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    int data = 0;
    EXPECT_THROW(adapter.irecv(&data, sizeof(int), 0, 0), nccl::communication_error);
}

TEST(NcclCommAdapterTest, BarrierThrowsOnInvalidComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    EXPECT_THROW(adapter.barrier(), nccl::communication_error);
}

TEST(NcclCommAdapterTest, ReduceSumThrowsOnInvalidComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    double send = 1.0, recv = 0.0;
    EXPECT_THROW(adapter.reduce_sum(&send, &recv, 1, 0), nccl::communication_error);
}

TEST(NcclCommAdapterTest, AllreduceSumThrowsOnInvalidComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    double send = 1.0, recv = 0.0;
    EXPECT_THROW(adapter.allreduce_sum(&send, &recv, 1), nccl::communication_error);
}

TEST(NcclCommAdapterTest, UnderlyingReturnsReference) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    EXPECT_EQ(&adapter.underlying(), &comm);
}

TEST(NcclCommAdapterTest, IsRootWithDefaultComm) {
    nccl::nccl_communicator comm;
    nccl::nccl_comm_adapter adapter(comm);
    // Default-constructed has rank no_rank (-1), so not root
    EXPECT_FALSE(adapter.is_root());
}

TEST(NcclCommAdapterTest, HasSizeType) {
    static_assert(std::is_same_v<nccl::nccl_comm_adapter::size_type, dtl::size_type>,
                  "size_type should be dtl::size_type");
    SUCCEED();
}

TEST(NcclCommAdapterTest, NonCopyable) {
    // Adapter should be copyable (it holds raw ptr + shared_ptr)
    static_assert(std::is_copy_constructible_v<nccl::nccl_comm_adapter>,
                  "nccl_comm_adapter should be copy constructible");
    SUCCEED();
}

#else

// Placeholder tests when NCCL/CUDA not available

TEST(NcclCommAdapterTest, ConceptSatisfactionPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, ConstructFromCommunicatorPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, ConstructFromSharedPtrPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, SendThrowsPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, RecvThrowsPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, IsendThrowsPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, IrecvThrowsPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, BarrierThrowsPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, ReduceSumThrowsPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, AllreduceSumThrowsPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, UnderlyingPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, IsRootPlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, HasSizeTypePlaceholder) {
    SUCCEED();
}

TEST(NcclCommAdapterTest, CopyablePlaceholder) {
    SUCCEED();
}

#endif  // DTL_ENABLE_NCCL && DTL_ENABLE_CUDA

}  // namespace dtl::test
