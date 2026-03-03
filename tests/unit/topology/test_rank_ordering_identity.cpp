// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rank_ordering_identity.cpp
/// @brief Verify optimize_rank_ordering returns identity ordering

#include <backends/mpi/mpi_topology.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <utility>

// optimize_rank_ordering takes (const mpi_topology&, const vector<pair<rank_t,rank_t>>&)
// and returns a vector<rank_t> with identity permutation (stub).
// Since constructing a real mpi_topology requires an MPI communicator,
// we use a default-constructed mpi_topology which has world_size()==0.
// The returned vector should be empty (identity of size 0).

TEST(RankOrderingIdentity, DefaultTopologyReturnsEmptyOrdering) {
    dtl::mpi::mpi_topology topo;  // default: world_size == 0
    std::vector<std::pair<dtl::rank_t, dtl::rank_t>> pattern;
    auto ordering = dtl::mpi::optimize_rank_ordering(topo, pattern);
    // Default topology has world_size 0, so identity ordering is empty
    EXPECT_TRUE(ordering.empty());
}

TEST(RankOrderingIdentity, IdentityPermutationProperty) {
    // Verify that the result satisfies the identity property:
    // for all i in [0, size), ordering[i] == i
    dtl::mpi::mpi_topology topo;
    std::vector<std::pair<dtl::rank_t, dtl::rank_t>> pattern;
    auto ordering = dtl::mpi::optimize_rank_ordering(topo, pattern);
    std::vector<dtl::rank_t> expected(static_cast<std::size_t>(topo.world_size()));
    std::iota(expected.begin(), expected.end(), dtl::rank_t{0});
    EXPECT_EQ(ordering, expected);
}
