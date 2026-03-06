// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_domain.cpp
/// @brief Unit tests for domain types
/// @details Tests for V1.3.0 domain abstractions (mpi_domain, cpu_domain, etc.)

#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Domain Tag Tests
// =============================================================================

TEST(DomainTest, TagTypesExist) {
    // Verify tag types are distinct
    static_assert(!std::is_same_v<mpi_domain_tag, cpu_domain_tag>);
    static_assert(!std::is_same_v<mpi_domain_tag, cuda_domain_tag>);
    static_assert(!std::is_same_v<mpi_domain_tag, nccl_domain_tag>);
    static_assert(!std::is_same_v<mpi_domain_tag, shmem_domain_tag>);
    static_assert(!std::is_same_v<cpu_domain_tag, cuda_domain_tag>);

    EXPECT_TRUE(true);  // Make GTest happy
}

// =============================================================================
// CPU Domain Tests
// =============================================================================

TEST(DomainTest, CpuDomainBasics) {
    cpu_domain cpu;

    // Always single-rank
    EXPECT_EQ(cpu.rank(), 0);
    EXPECT_EQ(cpu.size(), 1);

    // Always valid
    EXPECT_TRUE(cpu.valid());
    EXPECT_TRUE(cpu.is_root());
}

TEST(DomainTest, CpuDomainConstexpr) {
    constexpr cpu_domain cpu;

    static_assert(cpu.rank() == 0);
    static_assert(cpu.size() == 1);
    static_assert(cpu.valid());
    static_assert(cpu.is_root());

    EXPECT_TRUE(true);  // Make GTest happy
}

TEST(DomainTest, CpuDomainSatisfiesConcepts) {
    static_assert(CommunicationDomain<cpu_domain>);
    static_assert(ExecutionDomain<cpu_domain>);

    EXPECT_TRUE(CommunicationDomain<cpu_domain>);
}

TEST(DomainTest, CpuDomainTagType) {
    static_assert(std::is_same_v<cpu_domain::tag_type, cpu_domain_tag>);
    EXPECT_TRUE(true);
}

// =============================================================================
// CUDA Domain Tests (Stub)
// =============================================================================

TEST(DomainTest, CudaDomainDefaultConstruction) {
    cuda_domain cuda;

#if DTL_ENABLE_CUDA
    // May or may not be valid depending on GPU availability
    EXPECT_GE(cuda.device_id(), -1);
#else
    // Stub: always invalid
    EXPECT_FALSE(cuda.valid());
    EXPECT_EQ(cuda.device_id(), -1);
#endif
}

TEST(DomainTest, CudaDomainTagType) {
    static_assert(std::is_same_v<cuda_domain::tag_type, cuda_domain_tag>);
    EXPECT_TRUE(true);
}

// =============================================================================
// NCCL Domain Tests (Stub)
// =============================================================================

TEST(DomainTest, NcclDomainDefaultConstruction) {
    nccl_domain nccl;

    // Default construction creates invalid domain
    EXPECT_FALSE(nccl.valid());
    EXPECT_EQ(nccl.rank(), 0);
    EXPECT_EQ(nccl.size(), 1);
}

TEST(DomainTest, NcclDomainModeAndCapabilityQueries) {
    nccl_domain nccl;
    EXPECT_EQ(nccl.mode(), nccl_operation_mode::hybrid_parity);
    EXPECT_FALSE(nccl.supports_native(nccl_operation::scan));
#if DTL_ENABLE_NCCL
    EXPECT_TRUE(nccl.supports_hybrid(nccl_operation::scan));
#else
    EXPECT_FALSE(nccl.supports_hybrid(nccl_operation::scan));
#endif
}

TEST(DomainTest, NcclDomainTagType) {
    static_assert(std::is_same_v<nccl_domain::tag_type, nccl_domain_tag>);
    EXPECT_TRUE(true);
}

// =============================================================================
// SHMEM Domain Tests (Stub)
// =============================================================================

TEST(DomainTest, ShmemDomainDefaultConstruction) {
    shmem_domain shmem;

#if DTL_ENABLE_SHMEM
    // May be valid if SHMEM is initialized
    EXPECT_GE(shmem.rank(), 0);
    EXPECT_GE(shmem.size(), 1);
#else
    // Stub: always invalid
    EXPECT_FALSE(shmem.valid());
    EXPECT_EQ(shmem.rank(), 0);
    EXPECT_EQ(shmem.size(), 1);
#endif
}

TEST(DomainTest, ShmemDomainTagType) {
    static_assert(std::is_same_v<shmem_domain::tag_type, shmem_domain_tag>);
    EXPECT_TRUE(true);
}

// =============================================================================
// MPI Domain Tests (requires MPI)
// =============================================================================

#if DTL_ENABLE_MPI

TEST(DomainTest, MpiDomainBasics) {
    // Note: This test requires MPI to be initialized
    // In a proper test setup, environment would handle this
    mpi_domain mpi;

    if (mpi.valid()) {
        EXPECT_GE(mpi.rank(), 0);
        EXPECT_GE(mpi.size(), 1);
        EXPECT_LE(mpi.rank(), mpi.size() - 1);
        EXPECT_EQ(mpi.is_root(), (mpi.rank() == 0));
    }
}

TEST(DomainTest, MpiDomainSatisfiesConcepts) {
    static_assert(CommunicationDomain<mpi_domain>);
    EXPECT_TRUE(CommunicationDomain<mpi_domain>);
}

#endif  // DTL_ENABLE_MPI

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(DomainTest, CommunicationDomainTrait) {
    static_assert(is_communication_domain_v<cpu_domain>);
    static_assert(is_communication_domain_v<mpi_domain>);
    static_assert(is_communication_domain_v<nccl_domain>);
    static_assert(is_communication_domain_v<shmem_domain>);

    // cpu_domain also satisfies communication domain
    EXPECT_TRUE(is_communication_domain_v<cpu_domain>);
}

TEST(DomainTest, ExecutionDomainTrait) {
    static_assert(is_execution_domain_v<cpu_domain>);
    static_assert(is_execution_domain_v<cuda_domain>);
    static_assert(is_execution_domain_v<hip_domain>);

    EXPECT_TRUE(is_execution_domain_v<cpu_domain>);
}

TEST(DomainTest, PrimaryCommunicationDomainEmpty) {
    using primary = primary_communication_domain_t<>;
    static_assert(std::is_same_v<primary, void>);
    EXPECT_TRUE(true);
}

TEST(DomainTest, PrimaryCommunicationDomainSingle) {
    using primary = primary_communication_domain_t<mpi_domain>;
    static_assert(std::is_same_v<primary, mpi_domain>);
    EXPECT_TRUE(true);
}

TEST(DomainTest, PrimaryCommunicationDomainMixed) {
    // MPI comes first, so it's the primary
    using primary1 = primary_communication_domain_t<mpi_domain, cpu_domain>;
    static_assert(std::is_same_v<primary1, mpi_domain>);

    // CPU comes first here
    using primary2 = primary_communication_domain_t<cpu_domain, mpi_domain>;
    static_assert(std::is_same_v<primary2, cpu_domain>);

    EXPECT_TRUE(true);
}

}  // namespace dtl::test
