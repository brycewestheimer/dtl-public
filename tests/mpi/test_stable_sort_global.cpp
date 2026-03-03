// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_stable_sort_global.cpp
/// @brief Integration tests for globally stable distributed sort
/// @details Tests that stable_sort_global preserves original global ordering
///          for equal-key elements across multiple ranks.
/// @note Run with: mpirun -np 2 ./dtl_mpi_tests --gtest_filter="*StableSortGlobal*"
///       or:       mpirun -np 4 ./dtl_mpi_tests --gtest_filter="*StableSortGlobal*"

#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/sorting/stable_sort_global.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Element with Origin Tracking (for test verification)
// =============================================================================

/// @brief Test element that tracks its original position
struct TrackedElement {
    int key;                ///< Sort key (may have duplicates)
    rank_t origin_rank;     ///< Rank where element originated
    size_type origin_index; ///< Local index on originating rank

    bool operator==(const TrackedElement& other) const = default;
};

/// @brief Comparator that only compares keys (ignores origin)
struct CompareByKey {
    bool operator()(const TrackedElement& a, const TrackedElement& b) const {
        return a.key < b.key;
    }
};

// =============================================================================
// Test Fixture
// =============================================================================

class StableSortGlobalTest : public ::testing::Test {
protected:
    void SetUp() override {
        mpi_domain_ = std::make_unique<mpi_domain>();
        adapter_ = &mpi_domain_->communicator();
        my_rank_ = mpi_domain_->rank();
        num_ranks_ = mpi_domain_->size();
    }

    /// @brief Verify that equal-key elements are in origin order
    /// @param local_view View of the locally sorted data
    /// @return true if stability is maintained within local partition
    bool verify_local_stability(const auto& local_view) {
        for (size_type i = 1; i < local_view.size(); ++i) {
            const auto& prev = local_view[i - 1];
            const auto& curr = local_view[i];

            if (prev.key == curr.key) {
                // Equal keys: verify origin order
                if (prev.origin_rank > curr.origin_rank) {
                    return false;  // Wrong rank order
                }
                if (prev.origin_rank == curr.origin_rank &&
                    prev.origin_index >= curr.origin_index) {
                    return false;  // Wrong index order within same rank
                }
            }
        }
        return true;
    }

    /// @brief Gather all elements to rank 0 for global verification
    std::vector<TrackedElement> gather_all(
        const distributed_vector<TrackedElement>& vec) {

        auto local_v = vec.local_view();
        int local_count = static_cast<int>(local_v.size());

        // Gather counts
        std::vector<int> counts(static_cast<size_type>(num_ranks_));
        adapter_->allgather(&local_count, counts.data(), sizeof(int));

        // Compute displacements
        std::vector<int> displs(static_cast<size_type>(num_ranks_));
        std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);

        size_type total = static_cast<size_type>(displs.back() + counts.back());

        // Gather all elements
        std::vector<TrackedElement> all_elements(total);
        adapter_->gatherv(local_v.data(), static_cast<size_type>(local_count),
                          all_elements.data(), counts.data(), displs.data(),
                          sizeof(TrackedElement), 0);

        return all_elements;
    }

    /// @brief Verify global stability on rank 0
    bool verify_global_stability(const std::vector<TrackedElement>& all) {
        if (my_rank_ != 0) return true;  // Only rank 0 verifies

        for (size_type i = 1; i < all.size(); ++i) {
            const auto& prev = all[i - 1];
            const auto& curr = all[i];

            if (prev.key == curr.key) {
                // Equal keys: verify origin order
                if (prev.origin_rank > curr.origin_rank) {
                    return false;
                }
                if (prev.origin_rank == curr.origin_rank &&
                    prev.origin_index >= curr.origin_index) {
                    return false;
                }
            }
        }
        return true;
    }

    std::unique_ptr<mpi_domain> mpi_domain_;
    mpi::mpi_comm_adapter* adapter_ = nullptr;
    rank_t my_rank_ = 0;
    rank_t num_ranks_ = 1;
};

// =============================================================================
// Basic Stability Tests
// =============================================================================

TEST_F(StableSortGlobalTest, AllEqualKeys) {
    // All elements have the same key - stability should preserve original order
    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    // Fill: all keys are 42, but each element tracks its origin
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = TrackedElement{42, my_rank_, i};
    }

    // Sort globally with stability
    auto result = stable_sort_global(par{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);

    // Verify local stability
    EXPECT_TRUE(verify_local_stability(vec.local_view()))
        << "Local stability violated on rank " << my_rank_;

    // Gather and verify global stability on rank 0
    auto all = gather_all(vec);
    bool global_ok = verify_global_stability(all);

    // Broadcast verification result
    bool global_result = adapter_->allreduce_land_value(global_ok);
    EXPECT_TRUE(global_result) << "Global stability violated";
}

TEST_F(StableSortGlobalTest, TwoDistinctKeys) {
    // Two keys (0 and 1), many duplicates
    size_type elements_per_rank = 20;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    // Fill: alternating keys 0 and 1
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        int key = static_cast<int>(i % 2);  // 0 or 1
        local_v[i] = TrackedElement{key, my_rank_, i};
    }

    // Sort globally
    auto result = stable_sort_global(par{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);

    // Verify local view is sorted by key
    auto sorted_local = vec.local_view();
    bool is_key_sorted = std::is_sorted(
        sorted_local.begin(), sorted_local.end(),
        [](const TrackedElement& a, const TrackedElement& b) {
            return a.key < b.key;
        });
    EXPECT_TRUE(is_key_sorted) << "Data not sorted by key on rank " << my_rank_;

    // Verify stability
    EXPECT_TRUE(verify_local_stability(sorted_local))
        << "Local stability violated on rank " << my_rank_;

    auto all = gather_all(vec);
    bool global_ok = verify_global_stability(all);
    bool global_result = adapter_->allreduce_land_value(global_ok);
    EXPECT_TRUE(global_result) << "Global stability violated";
}

TEST_F(StableSortGlobalTest, MultipleKeyGroups) {
    // Multiple key values with many duplicates per key
    size_type elements_per_rank = 30;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    // Fill: keys 0-9 with repetition
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        int key = static_cast<int>(i % 10);
        local_v[i] = TrackedElement{key, my_rank_, i};
    }

    // Sort globally
    auto result = stable_sort_global(par{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);

    // Verify sorting
    auto sorted_local = vec.local_view();
    bool is_key_sorted = std::is_sorted(
        sorted_local.begin(), sorted_local.end(),
        [](const TrackedElement& a, const TrackedElement& b) {
            return a.key < b.key;
        });
    EXPECT_TRUE(is_key_sorted);

    // Verify stability
    EXPECT_TRUE(verify_local_stability(sorted_local));

    auto all = gather_all(vec);
    bool global_ok = verify_global_stability(all);
    bool global_result = adapter_->allreduce_land_value(global_ok);
    EXPECT_TRUE(global_result) << "Global stability violated with multiple keys";
}

TEST_F(StableSortGlobalTest, SequentialExecution) {
    // Test with sequential execution policy
    size_type elements_per_rank = 15;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        int key = static_cast<int>(i % 5);
        local_v[i] = TrackedElement{key, my_rank_, i};
    }

    // Sort with sequential policy
    auto result = stable_sort_global(seq{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);

    EXPECT_TRUE(verify_local_stability(vec.local_view()));

    auto all = gather_all(vec);
    bool global_ok = verify_global_stability(all);
    bool global_result = adapter_->allreduce_land_value(global_ok);
    EXPECT_TRUE(global_result);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(StableSortGlobalTest, EmptyContainer) {
    distributed_vector<TrackedElement> vec(0, *adapter_);

    auto result = stable_sort_global(par{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_sent, 0u);
    EXPECT_EQ(result.elements_received, 0u);
}

TEST_F(StableSortGlobalTest, SingleElementPerRank) {
    size_type global_size = static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    // Each rank has one element with same key
    auto local_v = vec.local_view();
    if (local_v.size() > 0) {
        local_v[0] = TrackedElement{100, my_rank_, 0};
    }

    auto result = stable_sort_global(par{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);

    // All elements have same key - should be in rank order
    auto all = gather_all(vec);
    if (my_rank_ == 0) {
        for (size_type i = 1; i < all.size(); ++i) {
            EXPECT_LT(all[i - 1].origin_rank, all[i].origin_rank)
                << "Elements not in rank order at index " << i;
        }
    }
}

TEST_F(StableSortGlobalTest, AlreadySorted) {
    // Data is already globally sorted - should remain stable
    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    // Fill with ascending keys (already sorted)
    auto local_v = vec.local_view();
    size_type base = my_rank_ * elements_per_rank;
    for (size_type i = 0; i < local_v.size(); ++i) {
        int key = static_cast<int>(base + i);
        local_v[i] = TrackedElement{key, my_rank_, i};
    }

    auto result = stable_sort_global(par{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);

    // Should still be sorted and stable
    auto sorted_local = vec.local_view();
    EXPECT_TRUE(std::is_sorted(
        sorted_local.begin(), sorted_local.end(),
        [](const TrackedElement& a, const TrackedElement& b) {
            return a.key < b.key;
        }));
}

TEST_F(StableSortGlobalTest, ReverseSorted) {
    // Data is reverse sorted - should become sorted while maintaining stability
    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    // Fill with descending keys
    auto local_v = vec.local_view();
    size_type base = (num_ranks_ - my_rank_ - 1) * elements_per_rank;
    for (size_type i = 0; i < local_v.size(); ++i) {
        int key = static_cast<int>(base + elements_per_rank - i - 1);
        local_v[i] = TrackedElement{key, my_rank_, i};
    }

    auto result = stable_sort_global(par{}, vec, CompareByKey{}, *adapter_);
    EXPECT_TRUE(result.success);

    // Should be sorted
    auto sorted_local = vec.local_view();
    EXPECT_TRUE(std::is_sorted(
        sorted_local.begin(), sorted_local.end(),
        [](const TrackedElement& a, const TrackedElement& b) {
            return a.key < b.key;
        }));
}

// =============================================================================
// Descending Order Test
// =============================================================================

TEST_F(StableSortGlobalTest, DescendingOrder) {
    size_type elements_per_rank = 20;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    distributed_vector<TrackedElement> vec(global_size, *adapter_);

    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        int key = static_cast<int>(i % 5);
        local_v[i] = TrackedElement{key, my_rank_, i};
    }

    // Sort in descending order
    auto desc_comp = [](const TrackedElement& a, const TrackedElement& b) {
        return a.key > b.key;  // Greater-than for descending
    };

    auto result = stable_sort_global(par{}, vec, desc_comp, *adapter_);
    EXPECT_TRUE(result.success);

    // Verify descending order
    auto sorted_local = vec.local_view();
    bool is_desc_sorted = std::is_sorted(
        sorted_local.begin(), sorted_local.end(),
        [](const TrackedElement& a, const TrackedElement& b) {
            return a.key > b.key;
        });
    EXPECT_TRUE(is_desc_sorted);

    // Stability for descending: among equal keys, still origin order
    // (lower rank/index first, even in descending key order)
    for (size_type i = 1; i < sorted_local.size(); ++i) {
        const auto& prev = sorted_local[i - 1];
        const auto& curr = sorted_local[i];
        if (prev.key == curr.key) {
            // Equal keys should maintain origin order
            bool origin_ok = (prev.origin_rank < curr.origin_rank) ||
                             (prev.origin_rank == curr.origin_rank &&
                              prev.origin_index < curr.origin_index);
            EXPECT_TRUE(origin_ok)
                << "Stability violated at local index " << i;
        }
    }
}

// =============================================================================
// Determinism Test
// =============================================================================

TEST_F(StableSortGlobalTest, Deterministic) {
    // Run sort twice, verify identical results
    size_type elements_per_rank = 25;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks_);

    // First sort
    distributed_vector<TrackedElement> vec1(global_size, *adapter_);
    auto local1 = vec1.local_view();
    for (size_type i = 0; i < local1.size(); ++i) {
        int key = static_cast<int>((i * 7 + my_rank_ * 3) % 8);  // Pseudo-random
        local1[i] = TrackedElement{key, my_rank_, i};
    }

    stable_sort_global(par{}, vec1, CompareByKey{}, *adapter_);
    auto result1 = gather_all(vec1);

    // Second sort with identical input
    distributed_vector<TrackedElement> vec2(global_size, *adapter_);
    auto local2 = vec2.local_view();
    for (size_type i = 0; i < local2.size(); ++i) {
        int key = static_cast<int>((i * 7 + my_rank_ * 3) % 8);
        local2[i] = TrackedElement{key, my_rank_, i};
    }

    stable_sort_global(par{}, vec2, CompareByKey{}, *adapter_);
    auto result2 = gather_all(vec2);

    // Verify identical on rank 0
    if (my_rank_ == 0) {
        ASSERT_EQ(result1.size(), result2.size());
        for (size_type i = 0; i < result1.size(); ++i) {
            EXPECT_EQ(result1[i], result2[i])
                << "Results differ at index " << i;
        }
    }
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test
