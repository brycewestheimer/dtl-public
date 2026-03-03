// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_communicator_split.cpp
/// @brief Unit tests for communicator split ownership model
/// @details Verifies that mpi_comm_adapter correctly handles ownership
///          when split() creates new communicators. Full split testing
///          with MPI is in integration/test_mpi_communicator.cpp

#include <backends/mpi/mpi_comm_adapter.hpp>
#include <backends/mpi/mpi_lifecycle.hpp>
#include <gtest/gtest.h>

namespace dtl::tests {

// ============================================================================
// Ownership Model Tests
// ============================================================================

/// @test Verify default adapter construction still works with ownership model
TEST(MpiSplitTest, DefaultAdapterWorks) {
    // Default adapter wraps the world communicator (non-owning)
    dtl::mpi::mpi_comm_adapter adapter;

    // Should be able to query rank and size
#if DTL_ENABLE_MPI
    if (dtl::mpi::is_initialized()) {
        EXPECT_GE(adapter.rank(), 0);
        EXPECT_GT(adapter.size(), 0);
    } else {
        // Pre-init access should be safe but yields sentinel values
        EXPECT_EQ(adapter.rank(), dtl::no_rank);
        EXPECT_EQ(adapter.size(), dtl::rank_t{0});
    }
#else
    EXPECT_EQ(adapter.rank(), dtl::no_rank);
    EXPECT_EQ(adapter.size(), dtl::rank_t{0});
#endif
}

/// @test Verify adapter is copyable (shared ownership via shared_ptr)
TEST(MpiSplitTest, AdapterIsCopyable) {
    dtl::mpi::mpi_comm_adapter adapter1;

    // Copy construction
    dtl::mpi::mpi_comm_adapter adapter2 = adapter1;
    EXPECT_EQ(adapter1.rank(), adapter2.rank());
    EXPECT_EQ(adapter1.size(), adapter2.size());

    // Copy assignment
    dtl::mpi::mpi_comm_adapter adapter3;
    adapter3 = adapter1;
    EXPECT_EQ(adapter1.rank(), adapter3.rank());
    EXPECT_EQ(adapter1.size(), adapter3.size());
}

/// @test Verify adapter is movable
TEST(MpiSplitTest, AdapterIsMovable) {
    dtl::mpi::mpi_comm_adapter adapter1;
    auto rank1 = adapter1.rank();
    auto size1 = adapter1.size();

    // Move construction
    dtl::mpi::mpi_comm_adapter adapter2 = std::move(adapter1);
    EXPECT_EQ(adapter2.rank(), rank1);
    EXPECT_EQ(adapter2.size(), size1);

    // Move assignment
    dtl::mpi::mpi_comm_adapter adapter3;
    adapter3 = std::move(adapter2);
    EXPECT_EQ(adapter3.rank(), rank1);
    EXPECT_EQ(adapter3.size(), size1);
}

/// @test Verify world_adapter() factory works
TEST(MpiSplitTest, WorldAdapterFactory) {
    auto adapter = dtl::mpi::world_adapter();
#if DTL_ENABLE_MPI
    if (dtl::mpi::is_initialized()) {
        EXPECT_GE(adapter.rank(), 0);
        EXPECT_GT(adapter.size(), 0);
    } else {
        EXPECT_EQ(adapter.rank(), dtl::no_rank);
        EXPECT_EQ(adapter.size(), dtl::rank_t{0});
    }
#else
    EXPECT_EQ(adapter.rank(), dtl::no_rank);
    EXPECT_EQ(adapter.size(), dtl::rank_t{0});
#endif
}

// ============================================================================
// Concept Compliance
// ============================================================================

/// @test Verify mpi_comm_adapter still satisfies Communicator concept
TEST(MpiSplitTest, SatisfiesCommunicatorConcept) {
    // Compile-time check - if this compiles, the concept is satisfied
    static_assert(dtl::Communicator<dtl::mpi::mpi_comm_adapter>,
                  "mpi_comm_adapter must satisfy Communicator concept");
}

/// @test Verify mpi_comm_adapter still satisfies CollectiveCommunicator concept
TEST(MpiSplitTest, SatisfiesCollectiveCommunicatorConcept) {
    static_assert(dtl::CollectiveCommunicator<dtl::mpi::mpi_comm_adapter>,
                  "mpi_comm_adapter must satisfy CollectiveCommunicator concept");
}

/// @test Verify mpi_comm_adapter still satisfies ReducingCommunicator concept
TEST(MpiSplitTest, SatisfiesReducingCommunicatorConcept) {
    static_assert(dtl::ReducingCommunicator<dtl::mpi::mpi_comm_adapter>,
                  "mpi_comm_adapter must satisfy ReducingCommunicator concept");
}

// ============================================================================
// Note on MPI Split Testing
// ============================================================================
//
// Actual split() functionality testing requires MPI runtime and multiple ranks.
// Those tests are in:
//   tests/integration/mpi/test_mpi_communicator.cpp
//
// This unit test file only verifies:
//   1. The ownership model doesn't break existing functionality
//   2. Adapters remain copyable/movable
//   3. Concept compliance is maintained

}  // namespace dtl::tests
