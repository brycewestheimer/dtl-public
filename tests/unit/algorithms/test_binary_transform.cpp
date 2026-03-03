// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_binary_transform.cpp
/// @brief Phase 06 tests for binary transform dispatch consistency (T06)

#include <dtl/algorithms/modifying/transform.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>

#include <gtest/gtest.h>

#include <functional>

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
// T06 verification — seq and par must produce identical results for binary transform
// =============================================================================

TEST(BinaryTransformDispatch, SeqAndParIdenticalAdd) {
    distributed_vector<int> a(6, test_context{0, 1});
    distributed_vector<int> b(6, test_context{0, 1});
    distributed_vector<int> out_seq(6, test_context{0, 1});
    distributed_vector<int> out_par(6, test_context{0, 1});

    auto la = a.local_view();
    auto lb = b.local_view();
    for (size_t i = 0; i < 6; ++i) {
        la[i] = static_cast<int>(i) + 1;
        lb[i] = (static_cast<int>(i) + 1) * 10;
    }

    dtl::transform(dtl::seq{}, a, b, out_seq, std::plus<>{});
    dtl::transform(dtl::par{}, a, b, out_par, std::plus<>{});

    auto ls = out_seq.local_view();
    auto lp = out_par.local_view();
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(ls[i], lp[i]) << "Mismatch at index " << i;
    }
}

TEST(BinaryTransformDispatch, SeqAndParIdenticalMultiply) {
    distributed_vector<int> a(5, test_context{0, 1});
    distributed_vector<int> b(5, test_context{0, 1});
    distributed_vector<int> out_seq(5, test_context{0, 1});
    distributed_vector<int> out_par(5, test_context{0, 1});

    auto la = a.local_view();
    auto lb = b.local_view();
    for (size_t i = 0; i < 5; ++i) {
        la[i] = static_cast<int>(i) + 1;
        lb[i] = static_cast<int>(i) + 2;
    }

    dtl::transform(dtl::seq{}, a, b, out_seq, std::multiplies<>{});
    dtl::transform(dtl::par{}, a, b, out_par, std::multiplies<>{});

    auto ls = out_seq.local_view();
    auto lp = out_par.local_view();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(ls[i], lp[i]) << "Mismatch at index " << i;
    }
}

TEST(BinaryTransformDispatch, SeqAndParIdenticalCustomOp) {
    distributed_vector<int> a(4, test_context{0, 1});
    distributed_vector<int> b(4, test_context{0, 1});
    distributed_vector<int> out_seq(4, test_context{0, 1});
    distributed_vector<int> out_par(4, test_context{0, 1});

    auto la = a.local_view();
    auto lb = b.local_view();
    la[0] = 10; la[1] = 20; la[2] = 30; la[3] = 40;
    lb[0] = 3;  lb[1] = 7;  lb[2] = 5;  lb[3] = 2;

    auto custom_op = [](int x, int y) { return x - y; };

    dtl::transform(dtl::seq{}, a, b, out_seq, custom_op);
    dtl::transform(dtl::par{}, a, b, out_par, custom_op);

    auto ls = out_seq.local_view();
    auto lp = out_par.local_view();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ls[i], lp[i]) << "Mismatch at index " << i;
    }
}

TEST(BinaryTransformDispatch, BinaryTransformCorrectValues) {
    distributed_vector<int> a(4, test_context{0, 1});
    distributed_vector<int> b(4, test_context{0, 1});
    distributed_vector<int> out(4, test_context{0, 1});

    auto la = a.local_view();
    auto lb = b.local_view();
    la[0] = 1; la[1] = 2; la[2] = 3; la[3] = 4;
    lb[0] = 5; lb[1] = 6; lb[2] = 7; lb[3] = 8;

    dtl::transform(dtl::seq{}, a, b, out, std::plus<>{});

    auto lo = out.local_view();
    EXPECT_EQ(lo[0], 6);
    EXPECT_EQ(lo[1], 8);
    EXPECT_EQ(lo[2], 10);
    EXPECT_EQ(lo[3], 12);
}

TEST(BinaryTransformDispatch, EmptyContainers) {
    distributed_vector<int> a(0, test_context{0, 1});
    distributed_vector<int> b(0, test_context{0, 1});
    distributed_vector<int> out(0, test_context{0, 1});

    // Should not crash on empty containers
    dtl::transform(dtl::seq{}, a, b, out, std::plus<>{});
    dtl::transform(dtl::par{}, a, b, out, std::plus<>{});
    EXPECT_EQ(out.global_size(), 0u);
}

}  // namespace dtl::test
