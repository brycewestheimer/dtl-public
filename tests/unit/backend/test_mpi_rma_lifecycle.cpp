// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_rma_lifecycle.cpp
/// @brief Verify mpi_rma_adapter Rule-of-5 (destructor, move, no copy)

#include <backends/mpi/mpi_rma_adapter.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using dtl::mpi::mpi_rma_adapter;

TEST(MpiRmaLifecycle, NotCopyConstructible) {
    static_assert(!std::is_copy_constructible_v<mpi_rma_adapter>,
                  "mpi_rma_adapter must not be copy constructible");
}

TEST(MpiRmaLifecycle, NotCopyAssignable) {
    static_assert(!std::is_copy_assignable_v<mpi_rma_adapter>,
                  "mpi_rma_adapter must not be copy assignable");
}

TEST(MpiRmaLifecycle, MoveConstructible) {
    static_assert(std::is_move_constructible_v<mpi_rma_adapter>,
                  "mpi_rma_adapter must be move constructible");
}

TEST(MpiRmaLifecycle, MoveAssignable) {
    static_assert(std::is_move_assignable_v<mpi_rma_adapter>,
                  "mpi_rma_adapter must be move assignable");
}

TEST(MpiRmaLifecycle, DefaultConstructAndDestroy) {
    // Should not leak — destructor cleans up empty map
    mpi_rma_adapter adapter;
    EXPECT_EQ(adapter.rank(), dtl::no_rank);
}
