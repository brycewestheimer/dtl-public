// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_distributed_tensor_impl.cpp
/// @brief Unit tests for distributed_tensor implementation
/// @details Tests for Task 2.6: distributed_tensor (MVP-CRITICAL)

#include <dtl/containers/distributed_tensor.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace dtl::test {

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

// =============================================================================
// Layout Tests
// =============================================================================

TEST(LayoutTest, RowMajorLinearize2D) {
    // 3x4 matrix, row-major: last dimension varies fastest
    nd_extent<2> extents = {3, 4};

    EXPECT_EQ(row_major::linearize(nd_index<2>{0, 0}, extents), 0);
    EXPECT_EQ(row_major::linearize(nd_index<2>{0, 1}, extents), 1);
    EXPECT_EQ(row_major::linearize(nd_index<2>{0, 3}, extents), 3);
    EXPECT_EQ(row_major::linearize(nd_index<2>{1, 0}, extents), 4);
    EXPECT_EQ(row_major::linearize(nd_index<2>{2, 3}, extents), 11);
}

TEST(LayoutTest, RowMajorDelinearize2D) {
    nd_extent<2> extents = {3, 4};

    EXPECT_EQ(row_major::delinearize<2>(0, extents), (nd_index<2>{0, 0}));
    EXPECT_EQ(row_major::delinearize<2>(1, extents), (nd_index<2>{0, 1}));
    EXPECT_EQ(row_major::delinearize<2>(4, extents), (nd_index<2>{1, 0}));
    EXPECT_EQ(row_major::delinearize<2>(11, extents), (nd_index<2>{2, 3}));
}

TEST(LayoutTest, RowMajorRoundtrip) {
    nd_extent<3> extents = {4, 5, 6};
    size_type total = row_major::size(extents);

    for (index_t linear = 0; linear < static_cast<index_t>(total); ++linear) {
        auto idx = row_major::delinearize<3>(linear, extents);
        EXPECT_EQ(row_major::linearize(idx, extents), linear);
    }
}

TEST(LayoutTest, ColumnMajorLinearize2D) {
    // 3x4 matrix, column-major: first dimension varies fastest
    nd_extent<2> extents = {3, 4};

    EXPECT_EQ(column_major::linearize(nd_index<2>{0, 0}, extents), 0);
    EXPECT_EQ(column_major::linearize(nd_index<2>{1, 0}, extents), 1);
    EXPECT_EQ(column_major::linearize(nd_index<2>{2, 0}, extents), 2);
    EXPECT_EQ(column_major::linearize(nd_index<2>{0, 1}, extents), 3);
}

TEST(LayoutTest, LayoutSize) {
    nd_extent<3> extents = {2, 3, 4};
    EXPECT_EQ(row_major::size(extents), 24);
    EXPECT_EQ(column_major::size(extents), 24);
}

// =============================================================================
// ND Partition Map Tests
// =============================================================================

TEST(NDPartitionMapTest, Construction) {
    nd_extent<2> extents = {100, 50};
    nd_partition_map<block_partition<>, 2> map(extents, 0, 4, 1);

    EXPECT_EQ(map.global_extents()[0], 100);
    EXPECT_EQ(map.global_extents()[1], 50);
    EXPECT_EQ(map.num_ranks(), 4);
    EXPECT_EQ(map.my_rank(), 1);
    EXPECT_EQ(map.partition_dim(), 0);
}

TEST(NDPartitionMapTest, LocalExtents) {
    nd_extent<2> extents = {100, 50};  // Partition along dim 0
    nd_partition_map<block_partition<>, 2> map(extents, 0, 4, 1);

    // Each rank gets 25 rows, all 50 columns
    EXPECT_EQ(map.local_extents()[0], 25);
    EXPECT_EQ(map.local_extents()[1], 50);
}

TEST(NDPartitionMapTest, LocalSize) {
    nd_extent<2> extents = {100, 50};
    nd_partition_map<block_partition<>, 2> map(extents, 0, 4, 1);

    EXPECT_EQ(map.global_size(), 5000);  // 100 * 50
    EXPECT_EQ(map.local_size(), 1250);   // 25 * 50
}

TEST(NDPartitionMapTest, Ownership) {
    nd_extent<2> extents = {10, 5};
    nd_partition_map<block_partition<>, 2> map(extents, 0, 4, 1);

    // Partition along dim 0: rank 1 owns rows [3, 6)
    EXPECT_FALSE(map.is_local(nd_index<2>{0, 0}));
    EXPECT_FALSE(map.is_local(nd_index<2>{2, 0}));
    EXPECT_TRUE(map.is_local(nd_index<2>{3, 0}));
    EXPECT_TRUE(map.is_local(nd_index<2>{5, 4}));
    EXPECT_FALSE(map.is_local(nd_index<2>{6, 0}));
}

TEST(NDPartitionMapTest, IndexTranslation) {
    nd_extent<2> extents = {10, 5};
    nd_partition_map<block_partition<>, 2> map(extents, 0, 4, 1);

    // Global (3, 2) -> Local (0, 2) on rank 1
    auto local_idx = map.to_local(nd_index<2>{3, 2});
    EXPECT_EQ(local_idx[0], 0);
    EXPECT_EQ(local_idx[1], 2);

    // Local (0, 2) -> Global (3, 2) on rank 1
    auto global_idx = map.to_global(nd_index<2>{0, 2});
    EXPECT_EQ(global_idx[0], 3);
    EXPECT_EQ(global_idx[1], 2);
}

// =============================================================================
// Distributed Tensor Construction Tests
// =============================================================================

TEST(DistributedTensorTest, DefaultConstruction) {
    distributed_tensor<int, 2> tensor;

    EXPECT_EQ(tensor.size(), 0);
    EXPECT_TRUE(tensor.empty());
    EXPECT_EQ(tensor.rank(), 2);
}

TEST(DistributedTensorTest, ConstructWithExtents) {
    nd_extent<2> extents = {100, 50};
    distributed_tensor<double, 2> tensor(extents, test_context{1, 4});

    EXPECT_EQ(tensor.size(), 5000);
    EXPECT_EQ(tensor.local_size(), 1250);
    EXPECT_EQ(tensor.extent(0), 100);
    EXPECT_EQ(tensor.extent(1), 50);
    EXPECT_EQ(tensor.local_extent(0), 25);
    EXPECT_EQ(tensor.local_extent(1), 50);
}

TEST(DistributedTensorTest, ConstructWithInitialValue) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, 42, test_context{0, 1}, size_type{0});

    // All local elements should be 42
    for (size_type i = 0; i < tensor.local_size(); ++i) {
        EXPECT_EQ(tensor.local_linear(i), 42);
    }
}

TEST(DistributedTensorTest, Tensor3D) {
    nd_extent<3> extents = {10, 20, 30};
    distributed_tensor<float, 3> tensor(extents, test_context{0, 4});

    EXPECT_EQ(tensor.rank(), 3);
    EXPECT_EQ(tensor.size(), 6000);
    EXPECT_EQ(tensor.extent(0), 10);
    EXPECT_EQ(tensor.extent(1), 20);
    EXPECT_EQ(tensor.extent(2), 30);
}

// =============================================================================
// Local Access Tests
// =============================================================================

TEST(DistributedTensorTest, LocalNDAccess) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});  // Single rank

    // Write using ND indices
    tensor.local(nd_index<2>{0, 0}) = 100;
    tensor.local(nd_index<2>{5, 5}) = 200;
    tensor.local(nd_index<2>{9, 9}) = 300;

    // Read back
    EXPECT_EQ(tensor.local(nd_index<2>{0, 0}), 100);
    EXPECT_EQ(tensor.local(nd_index<2>{5, 5}), 200);
    EXPECT_EQ(tensor.local(nd_index<2>{9, 9}), 300);
}

TEST(DistributedTensorTest, LocalVariadicAccess) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    // Write using variadic indices
    tensor.local(0, 0) = 100;
    tensor.local(5, 5) = 200;

    // Read back
    EXPECT_EQ(tensor.local(0, 0), 100);
    EXPECT_EQ(tensor.local(5, 5), 200);
}

TEST(DistributedTensorTest, OperatorParens) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    // Write using operator()
    tensor(0, 0) = 100;
    tensor(5, 5) = 200;

    // Read back
    EXPECT_EQ(tensor(0, 0), 100);
    EXPECT_EQ(tensor(5, 5), 200);
}

TEST(DistributedTensorTest, LocalLinearAccess) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    // Linear access
    for (size_type i = 0; i < tensor.local_size(); ++i) {
        tensor.local_linear(i) = static_cast<int>(i);
    }

    EXPECT_EQ(tensor.local_linear(0), 0);
    EXPECT_EQ(tensor.local_linear(50), 50);
    EXPECT_EQ(tensor.local_linear(99), 99);
}

// =============================================================================
// Global Access Tests
// =============================================================================

TEST(DistributedTensorTest, GlobalAccessReturnsRemoteRef) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{1, 4});

    auto ref = tensor.global(nd_index<2>{5, 5});

    // Should return remote_ref
    static_assert(std::is_same_v<decltype(ref), remote_ref<int>>);
}

TEST(DistributedTensorTest, GlobalLocalAccess) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, 42, test_context{0, 1}, size_type{0});

    auto ref = tensor.global(nd_index<2>{5, 5});
    EXPECT_TRUE(ref.is_local());

    auto result = ref.get();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

// =============================================================================
// Distribution Tests
// =============================================================================

TEST(DistributedTensorTest, IsLocal) {
    nd_extent<2> extents = {10, 5};
    distributed_tensor<int, 2> tensor(extents, test_context{1, 4}, size_type{0});  // Rank 1 of 4

    // Rank 1 owns rows [3, 6)
    EXPECT_FALSE(tensor.is_local(nd_index<2>{0, 0}));
    EXPECT_FALSE(tensor.is_local(nd_index<2>{2, 4}));
    EXPECT_TRUE(tensor.is_local(nd_index<2>{3, 0}));
    EXPECT_TRUE(tensor.is_local(nd_index<2>{5, 4}));
    EXPECT_FALSE(tensor.is_local(nd_index<2>{6, 0}));
}

TEST(DistributedTensorTest, Owner) {
    nd_extent<2> extents = {10, 5};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 4}, size_type{0});

    EXPECT_EQ(tensor.owner(nd_index<2>{0, 0}), 0);
    EXPECT_EQ(tensor.owner(nd_index<2>{2, 0}), 0);
    EXPECT_EQ(tensor.owner(nd_index<2>{3, 0}), 1);
    EXPECT_EQ(tensor.owner(nd_index<2>{6, 0}), 2);
    EXPECT_EQ(tensor.owner(nd_index<2>{8, 0}), 3);
}

TEST(DistributedTensorTest, IndexTranslation) {
    nd_extent<2> extents = {10, 5};
    distributed_tensor<int, 2> tensor(extents, test_context{1, 4}, size_type{0});

    // Global (3, 2) -> Local (0, 2)
    auto local = tensor.to_local(nd_index<2>{3, 2});
    EXPECT_EQ(local[0], 0);
    EXPECT_EQ(local[1], 2);

    // Local (0, 2) -> Global (3, 2)
    auto global = tensor.to_global(nd_index<2>{0, 2});
    EXPECT_EQ(global[0], 3);
    EXPECT_EQ(global[1], 2);
}

TEST(DistributedTensorTest, GlobalOffset) {
    nd_extent<2> extents = {100, 50};
    distributed_tensor<int, 2> tensor(extents, test_context{2, 4}, size_type{0});

    auto offset = tensor.global_offset();
    EXPECT_EQ(offset[0], 50);  // Row offset for rank 2
    EXPECT_EQ(offset[1], 0);   // No column offset
}

// =============================================================================
// View Tests
// =============================================================================

TEST(DistributedTensorTest, LocalView) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, 42, test_context{0, 1}, size_type{0});

    auto view = tensor.local_view();
    EXPECT_EQ(view.size(), 100);

    // Verify content
    for (size_type i = 0; i < view.size(); ++i) {
        EXPECT_EQ(view[i], 42);
    }
}

TEST(DistributedTensorTest, LocalSpan) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    auto span = tensor.local_span();
    EXPECT_EQ(span.size(), 100);

    // Can use STL on span
    std::fill(span.begin(), span.end(), 99);
    EXPECT_EQ(tensor.local_linear(0), 99);
    EXPECT_EQ(tensor.local_linear(99), 99);
}

// =============================================================================
// Synchronization Tests
// =============================================================================

TEST(DistributedTensorTest, Barrier) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    auto result = tensor.barrier();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedTensorTest, Fence) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    auto result = tensor.fence();
    EXPECT_TRUE(result.has_value());
}

TEST(DistributedTensorTest, StructuralMetadataConsistentAfterConstruction) {
    nd_extent<2> extents = {8, 4};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});
    EXPECT_TRUE(tensor.structural_metadata_consistent());
}

TEST(DistributedTensorTest, ReplaceLocalPartitionPreservesExtents) {
    nd_extent<2> extents = {4, 4};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});
    distributed_tensor<int, 2>::storage_type replacement(tensor.local_size(), 5);

    auto result = tensor.replace_local_partition(std::move(replacement));
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(tensor.structural_metadata_consistent());
    EXPECT_EQ(tensor.size(), 16u);
}

TEST(DistributedTensorTest, ReplaceLocalPartitionRejectsSizeMismatch) {
    nd_extent<2> extents = {4, 4};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});
    distributed_tensor<int, 2>::storage_type replacement(tensor.local_size() + 1, 7);

    auto result = tensor.replace_local_partition(std::move(replacement));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), status_code::invalid_argument);
}

TEST(DistributedTensorTest, ReplaceLocalPartitionWithNewExtentsUpdatesMetadata) {
    nd_extent<2> old_extents = {4, 4};
    distributed_tensor<int, 2> tensor(old_extents, test_context{0, 1});
    nd_extent<2> new_extents = {3, 3};
    distributed_tensor<int, 2>::storage_type replacement(9, 11);

    auto result = tensor.replace_local_partition(std::move(replacement), new_extents);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(tensor.size(), 9u);
    EXPECT_EQ(tensor.local_size(), 9u);
    EXPECT_TRUE(tensor.structural_metadata_consistent());
}

// =============================================================================
// Structural Operation Tests
// =============================================================================

TEST(DistributedTensorTest, Resize) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    EXPECT_EQ(tensor.size(), 100);

    nd_extent<2> new_extents = {20, 20};
    auto result = tensor.resize(new_extents);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(tensor.size(), 400);
}

// =============================================================================
// Type Alias Tests
// =============================================================================

TEST(DistributedTensorTest, Tensor1DAlias) {
    tensor1d<int> t({100}, test_context{0, 4});
    EXPECT_EQ(t.rank(), 1);
    EXPECT_EQ(t.size(), 100);
}

TEST(DistributedTensorTest, Tensor2DAlias) {
    tensor2d<int> t({10, 20}, test_context{0, 4});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 200);
}

TEST(DistributedTensorTest, Tensor3DAlias) {
    tensor3d<int> t({5, 6, 7}, test_context{0, 4});
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 210);
}

TEST(DistributedTensorTest, DistributedMatrixAlias) {
    distributed_matrix<double> m({100, 100}, test_context{0, 4});
    EXPECT_EQ(m.rank(), 2);
    EXPECT_EQ(m.size(), 10000);
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(DistributedTensorTest, IsDistributedContainer) {
    static_assert(is_distributed_container_v<distributed_tensor<int, 2>>);
    static_assert(is_distributed_container_v<distributed_tensor<double, 3>>);
    static_assert(is_distributed_container_v<tensor1d<int>>);
    static_assert(is_distributed_container_v<tensor2d<double>>);

    static_assert(!is_distributed_container_v<std::vector<int>>);
}

// =============================================================================
// Layout Verification Tests
// =============================================================================

TEST(DistributedTensorTest, RowMajorLayout) {
    nd_extent<2> extents = {3, 4};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 1});

    // Fill with linear indices
    for (size_type i = 0; i < tensor.local_size(); ++i) {
        tensor.local_linear(i) = static_cast<int>(i);
    }

    // Verify row-major layout: element (i,j) is at linear index i*4 + j
    EXPECT_EQ(tensor(0, 0), 0);
    EXPECT_EQ(tensor(0, 1), 1);
    EXPECT_EQ(tensor(0, 3), 3);
    EXPECT_EQ(tensor(1, 0), 4);
    EXPECT_EQ(tensor(2, 3), 11);
}

// =============================================================================
// Constexpr Tests
// =============================================================================

TEST(DistributedTensorTest, ConstexprRank) {
    static_assert(distributed_tensor<int, 2>::tensor_rank == 2);
    static_assert(distributed_tensor<int, 3>::tensor_rank == 3);
    static_assert(tensor1d<int>::tensor_rank == 1);
    static_assert(tensor2d<int>::tensor_rank == 2);
    static_assert(tensor3d<int>::tensor_rank == 3);
}

// =============================================================================
// Global View Tests (Phase 1.2.2)
// =============================================================================

TEST(DistributedTensorTest, GlobalViewTypeExists) {
    using tensor_type = distributed_tensor<int, 2>;
    using global_view_type = typename tensor_type::global_view_type;

    // Verify the type alias exists
    static_assert(!std::is_same_v<global_view_type, void>,
                  "global_view_type must be defined");

    SUCCEED();
}

TEST(DistributedTensorTest, GlobalViewAccess) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, 42, test_context{0, 1}, size_type{0});

    auto gview = tensor.global_view();
    EXPECT_EQ(gview.size(), 100);

    // Local element access via linearized index
    auto ref = gview[0];
    EXPECT_TRUE(ref.is_local());
}

// =============================================================================
// Segmented View Tests (Phase 1.2.2)
// =============================================================================

TEST(DistributedTensorTest, SegmentedViewTypeExists) {
    using tensor_type = distributed_tensor<int, 2>;
    using segmented_view_type = typename tensor_type::segmented_view_type;

    static_assert(!std::is_same_v<segmented_view_type, void>,
                  "segmented_view_type must be defined");

    SUCCEED();
}

TEST(DistributedTensorTest, SegmentedViewAccess) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, 42, test_context{1, 4}, size_type{0});

    auto sview = tensor.segmented_view();
    EXPECT_EQ(sview.num_segments(), 4);  // One per rank

    // Check local segment
    auto local_seg = sview.local_segment();
    EXPECT_TRUE(local_seg.is_local());
}

// =============================================================================
// Process Rank Method Test (Phase 1.2.2)
// =============================================================================

TEST(DistributedTensorTest, MyRankMethod) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{2, 4}, size_type{0});

    // my_rank() returns the process rank
    EXPECT_EQ(tensor.my_rank(), 2);       // Process rank
    EXPECT_EQ(tensor.num_ranks(), 4);     // Number of processes

    // Static rank() returns tensor dimensionality (different concept)
    EXPECT_EQ(tensor.rank(), 2);          // Tensor dimensionality (2D)
    EXPECT_EQ(tensor.tensor_rank, 2);     // Same via constexpr member
}

// =============================================================================
// DistributedContainer Concept Compliance (Phase 1.2.2)
// =============================================================================

TEST(DistributedTensorTest, ConceptCompliance) {
    // distributed_tensor should satisfy DistributedContainer after adding views
    using tensor_type = distributed_tensor<int, 2>;

    // Verify required type aliases exist
    static_assert(std::is_same_v<typename tensor_type::value_type, int>,
                  "value_type must be defined");
    static_assert(!std::is_same_v<typename tensor_type::size_type, void>,
                  "size_type must be defined");

    // Create instance and verify methods exist
    nd_extent<2> extents = {10, 10};
    tensor_type tensor(extents, test_context{0, 1}, size_type{0});

    // These methods are required by DistributedContainer
    [[maybe_unused]] auto s = tensor.size();
    [[maybe_unused]] auto ls = tensor.local_size();
    [[maybe_unused]] auto lv = tensor.local_view();
    [[maybe_unused]] auto gv = tensor.global_view();
    [[maybe_unused]] auto sv = tensor.segmented_view();

    SUCCEED();
}

TEST(DistributedTensorTest, DistributedTensorConceptCompliance) {
    // distributed_tensor should satisfy DistributedTensor concept
    using tensor_type = distributed_tensor<int, 2>;

    // Verify extents_type exists
    static_assert(!std::is_same_v<typename tensor_type::extents_type, void>,
                  "extents_type must be defined");

    nd_extent<2> extents = {10, 10};
    tensor_type tensor(extents, test_context{0, 1}, size_type{0});

    // These are required by DistributedTensor concept
    [[maybe_unused]] auto ge = tensor.global_extents();
    [[maybe_unused]] auto le = tensor.local_extents();
    [[maybe_unused]] auto e0 = tensor.extent(0);

    SUCCEED();
}

// =============================================================================
// Linearized Index Translation Tests (Phase 1.2.2)
// =============================================================================

TEST(DistributedTensorTest, LinearizedIsLocal) {
    nd_extent<2> extents = {10, 10};  // 10x10 = 100 elements
    distributed_tensor<int, 2> tensor(extents, test_context{1, 4}, size_type{0});

    // With 10 rows and 4 ranks using block partition:
    // Rank 0: rows 0-2 (3 rows) -> linear indices 0-29
    // Rank 1: rows 3-5 (3 rows) -> linear indices 30-59
    // Rank 2: rows 6-7 (2 rows) -> linear indices 60-79
    // Rank 3: rows 8-9 (2 rows) -> linear indices 80-99

    EXPECT_FALSE(tensor.is_local(index_t{0}));   // Row 0 (rank 0)
    EXPECT_FALSE(tensor.is_local(index_t{29}));  // Row 2 (rank 0)
    EXPECT_TRUE(tensor.is_local(index_t{30}));   // Row 3 (first of rank 1)
    EXPECT_TRUE(tensor.is_local(index_t{59}));   // Row 5 (last of rank 1)
    EXPECT_FALSE(tensor.is_local(index_t{60}));  // Row 6 (rank 2)
}

TEST(DistributedTensorTest, LinearizedOwner) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{0, 4}, size_type{0});

    // With block partition: ranks 0,1 get 3 rows each, ranks 2,3 get 2 rows each
    EXPECT_EQ(tensor.owner(index_t{0}), 0);   // Row 0 (rank 0)
    EXPECT_EQ(tensor.owner(index_t{29}), 0);  // Row 2 (rank 0)
    EXPECT_EQ(tensor.owner(index_t{30}), 1);  // Row 3 (rank 1)
    EXPECT_EQ(tensor.owner(index_t{60}), 2);  // Row 6 (rank 2)
    EXPECT_EQ(tensor.owner(index_t{80}), 3);  // Row 8 (rank 3)
}

TEST(DistributedTensorTest, LinearizedToLocal) {
    nd_extent<2> extents = {10, 10};
    distributed_tensor<int, 2> tensor(extents, test_context{1, 4}, size_type{0});

    // Rank 1 owns rows 3-5 (global linear indices 30-59)
    // Global linear 30 (row 3, col 0) -> local linear 0 (row 0 of local, col 0)
    EXPECT_EQ(tensor.to_local(index_t{30}), 0);  // First element of rank 1
    EXPECT_EQ(tensor.to_local(index_t{31}), 1);  // Second element of rank 1
    EXPECT_EQ(tensor.to_local(index_t{40}), 10); // First element of local row 1
}

}  // namespace dtl::test
