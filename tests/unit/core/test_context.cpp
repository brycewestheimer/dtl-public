// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_context.cpp
/// @brief Unit tests for multi-domain context
/// @details Tests for V1.3.0 context<Domains...> template

#include <dtl/core/context.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// CPU Context Tests
// =============================================================================

TEST(ContextTest, CpuContextBasics) {
    cpu_context ctx;

    EXPECT_EQ(ctx.rank(), 0);
    EXPECT_EQ(ctx.size(), 1);
    EXPECT_TRUE(ctx.is_root());
    EXPECT_TRUE(ctx.valid());
}

TEST(ContextTest, CpuContextProducesValidHandle) {
    cpu_context ctx;
    auto h = ctx.handle();

    EXPECT_TRUE(h.valid());
    EXPECT_EQ(h.communicator().rank(), 0);
    EXPECT_EQ(h.communicator().size(), 1);
    EXPECT_TRUE(h.communicator().barrier().has_value());
}

TEST(ContextTest, CpuContextHasQueries) {
    static_assert(cpu_context::has<cpu_domain>());
    static_assert(!cpu_context::has<mpi_domain>());
    static_assert(!cpu_context::has<cuda_domain>());
    static_assert(!cpu_context::has<nccl_domain>());

    EXPECT_TRUE(cpu_context::has_cpu());
    EXPECT_FALSE(cpu_context::has_mpi());
    EXPECT_FALSE(cpu_context::has_cuda());
}

TEST(ContextTest, CpuContextDomainAccess) {
    cpu_context ctx;

    auto& cpu = ctx.get<cpu_domain>();
    EXPECT_EQ(cpu.rank(), 0);
    EXPECT_EQ(cpu.size(), 1);
}

TEST(ContextTest, CpuContextDomainCount) {
    static_assert(cpu_context::domain_count == 1);
    EXPECT_EQ(cpu_context::domain_count, 1);
}

TEST(ContextTest, CpuContextSatisfiesConcept) {
    static_assert(Context<cpu_context>);
    EXPECT_TRUE(Context<cpu_context>);
}

// =============================================================================
// MPI Context Tests (Type-Level)
// =============================================================================

TEST(ContextTest, MpiContextHasQueries) {
    static_assert(mpi_context::has<mpi_domain>());
    static_assert(mpi_context::has<cpu_domain>());
    static_assert(!mpi_context::has<cuda_domain>());
    static_assert(!mpi_context::has<nccl_domain>());

    EXPECT_TRUE(mpi_context::has_mpi());
    EXPECT_TRUE(mpi_context::has_cpu());
    EXPECT_FALSE(mpi_context::has_cuda());
}

TEST(ContextTest, MpiContextDomainCount) {
    static_assert(mpi_context::domain_count == 2);
    EXPECT_EQ(mpi_context::domain_count, 2);
}

TEST(ContextTest, MpiContextHasCommunicationDomain) {
    static_assert(mpi_context::has_communication_domain());
    EXPECT_TRUE(mpi_context::has_communication_domain());
}

// =============================================================================
// CUDA Context Tests (Type-Level)
// =============================================================================

TEST(ContextTest, MpiCudaContextHasQueries) {
    static_assert(mpi_cuda_context::has<mpi_domain>());
    static_assert(mpi_cuda_context::has<cpu_domain>());
    static_assert(mpi_cuda_context::has<cuda_domain>());
    static_assert(!mpi_cuda_context::has<nccl_domain>());

    EXPECT_TRUE(mpi_cuda_context::has_mpi());
    EXPECT_TRUE(mpi_cuda_context::has_cpu());
    EXPECT_TRUE(mpi_cuda_context::has_cuda());
    EXPECT_FALSE(mpi_cuda_context::has_nccl());
}

TEST(ContextTest, MpiCudaContextDomainCount) {
    static_assert(mpi_cuda_context::domain_count == 3);
    EXPECT_EQ(mpi_cuda_context::domain_count, 3);
}

// =============================================================================
// NCCL Context Tests (Type-Level)
// =============================================================================

TEST(ContextTest, MpiNcclContextHasQueries) {
    static_assert(mpi_nccl_context::has<mpi_domain>());
    static_assert(mpi_nccl_context::has<cpu_domain>());
    static_assert(mpi_nccl_context::has<cuda_domain>());
    static_assert(mpi_nccl_context::has<nccl_domain>());

    EXPECT_TRUE(mpi_nccl_context::has_mpi());
    EXPECT_TRUE(mpi_nccl_context::has_cpu());
    EXPECT_TRUE(mpi_nccl_context::has_cuda());
    EXPECT_TRUE(mpi_nccl_context::has_nccl());
}

TEST(ContextTest, MpiNcclContextDomainCount) {
    static_assert(mpi_nccl_context::domain_count == 4);
    EXPECT_EQ(mpi_nccl_context::domain_count, 4);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(ContextTest, MakeCpuContext) {
    auto ctx = make_cpu_context();

    EXPECT_EQ(ctx.rank(), 0);
    EXPECT_EQ(ctx.size(), 1);
    EXPECT_TRUE(ctx.is_root());
}

TEST(ContextTest, WithCpuFactory) {
    cpu_context ctx;
    auto ctx2 = ctx.with_cpu();

    // with_cpu on cpu_context should return same type (already has cpu)
    static_assert(std::is_same_v<decltype(ctx2), cpu_context>);
    EXPECT_EQ(ctx2.size(), 1);
}

// =============================================================================
// Context Construction Tests
// =============================================================================

TEST(ContextTest, ExplicitDomainConstruction) {
    cpu_context ctx{cpu_domain{}};

    EXPECT_EQ(ctx.rank(), 0);
    EXPECT_EQ(ctx.size(), 1);
}

TEST(ContextTest, TupleConstruction) {
    auto domains = std::make_tuple(cpu_domain{});
    cpu_context ctx{std::move(domains)};

    EXPECT_EQ(ctx.rank(), 0);
    EXPECT_EQ(ctx.size(), 1);
}

// =============================================================================
// Detail Helper Tests
// =============================================================================

TEST(ContextDetailTest, ContainsType) {
    static_assert(detail::contains_v<int, int, double, float>);
    static_assert(detail::contains_v<double, int, double, float>);
    static_assert(detail::contains_v<float, int, double, float>);
    static_assert(!detail::contains_v<char, int, double, float>);
    static_assert(!detail::contains_v<int>);  // Empty pack

    constexpr bool contains_int = detail::contains_v<int, int, double>;
    EXPECT_TRUE(contains_int);
}

TEST(ContextDetailTest, IndexOf) {
    static_assert(detail::index_of_v<int, int, double, float> == 0);
    static_assert(detail::index_of_v<double, int, double, float> == 1);
    static_assert(detail::index_of_v<float, int, double, float> == 2);

    constexpr size_type idx = detail::index_of_v<int, int, double>;
    EXPECT_EQ(idx, 0);
}

TEST(ContextDetailTest, AppendDomain) {
    using original = context<mpi_domain, cpu_domain>;
    using appended = detail::append_domain_t<original, cuda_domain>;

    static_assert(appended::has<mpi_domain>());
    static_assert(appended::has<cpu_domain>());
    static_assert(appended::has<cuda_domain>());
    static_assert(appended::domain_count == 3);

    EXPECT_TRUE(true);
}

// =============================================================================
// Legacy Compatibility Tests
// =============================================================================

TEST(ContextTest, NumRanksDeprecatedAlias) {
    cpu_context ctx;

    // num_ranks() is deprecated alias for size()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    EXPECT_EQ(ctx.num_ranks(), ctx.size());
#pragma GCC diagnostic pop
}

// =============================================================================
// Barrier/Sync Tests (CPU context only)
// =============================================================================

TEST(ContextTest, CpuContextBarrier) {
    cpu_context ctx;

    // CPU context barrier is a no-op
    ctx.barrier();
    EXPECT_TRUE(true);
}

TEST(ContextTest, UnboundMultiRankCommHandleReturnsExplicitError) {
    struct pseudo_ctx {
        [[nodiscard]] rank_t rank() const noexcept { return 1; }
        [[nodiscard]] rank_t size() const noexcept { return 4; }
    };

    auto comm = handle::make_comm_handle(pseudo_ctx{});
    auto barrier = comm.barrier();

    EXPECT_FALSE(barrier.has_value());
    EXPECT_EQ(barrier.error().code(), status_code::invalid_state);
}

TEST(ContextTest, CpuContextSynchronizeDevice) {
    cpu_context ctx;

    // CPU context doesn't have device, this is a no-op
    ctx.synchronize_device();
    EXPECT_TRUE(true);
}

}  // namespace dtl::test
