// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_memory_space.cpp
/// @brief Unit tests for CUDA memory space semantics
/// @details Covers typed allocation/deallocation, allocation tracking, and the
///          explicit trivial-type contract for device memory.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA
#include <backends/cuda/cuda_memory_space.hpp>
#endif

#include <dtl/backend/concepts/memory_space.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace dtl::test {

#if DTL_ENABLE_CUDA

TEST(CudaMemorySpaceUnitTest, SatisfiesMemorySpaceConcepts) {
    static_assert(MemorySpace<dtl::cuda::cuda_memory_space>);
    static_assert(TypedMemorySpace<dtl::cuda::cuda_memory_space, int>);
    static_assert(TypedMemorySpace<dtl::cuda::cuda_memory_space, double>);
    SUCCEED();
}

TEST(CudaMemorySpaceUnitTest, ReportsExpectedProperties) {
    dtl::cuda::cuda_memory_space space;
    auto props = space.properties();

    EXPECT_FALSE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_EQ(props.alignment, 256u);
}

TEST(CudaMemorySpaceUnitTest, ContainsRejectsHostPointer) {
    dtl::cuda::cuda_memory_space space;
    int host_value = 7;
    EXPECT_FALSE(space.contains(&host_value));
}

TEST(CudaMemorySpaceUnitTest, TypedAllocationTracking) {
    if (dtl::cuda::device_count() <= 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    dtl::cuda::cuda_memory_space space;
    constexpr dtl::size_type count = 16;
    constexpr dtl::size_type bytes = count * sizeof(int);

    auto* ptr = space.allocate_typed<int>(count);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(space.contains(ptr));
    EXPECT_EQ(space.total_allocated(), bytes);
    EXPECT_GE(space.peak_allocated(), bytes);

    space.construct(ptr);
    space.destroy(ptr);
    space.deallocate_typed(ptr, count);

    EXPECT_EQ(space.total_allocated(), 0u);
}

TEST(CudaMemorySpaceUnitTest, AlignedAllocationUsesAlignedDeallocation) {
    if (dtl::cuda::device_count() <= 0) {
        GTEST_SKIP() << "No CUDA devices available";
    }

    dtl::cuda::cuda_memory_space space;
    constexpr dtl::size_type size = 128;
    constexpr dtl::size_type alignment = 512;

    void* ptr = space.allocate(size, alignment);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % alignment, 0u);

    space.deallocate_aligned(ptr, size, alignment);
    EXPECT_EQ(space.total_allocated(), 0u);
}

#else

TEST(CudaMemorySpaceUnitTest, ConceptsPlaceholder) { SUCCEED(); }
TEST(CudaMemorySpaceUnitTest, PropertiesPlaceholder) { SUCCEED(); }
TEST(CudaMemorySpaceUnitTest, ContainsPlaceholder) { SUCCEED(); }
TEST(CudaMemorySpaceUnitTest, TypedAllocationPlaceholder) { SUCCEED(); }
TEST(CudaMemorySpaceUnitTest, AlignedAllocationPlaceholder) { SUCCEED(); }

#endif

}  // namespace dtl::test
