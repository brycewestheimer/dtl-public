// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_pinned_memory.cpp
/// @brief Unit tests for pinned memory space
/// @details Tests allocation and deallocation using the fallback path
///          (std::malloc/std::free) when no GPU backend is available.
///          Updated for static interface and thread-safe statistics.

#include <dtl/memory/pinned_memory_space.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>

namespace dtl::test {

// Helper fixture to reset statistics between tests
class PinnedMemorySpaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        dtl::pinned_memory_space::reset_statistics();
    }
    void TearDown() override {
        dtl::pinned_memory_space::reset_statistics();
    }
};

// =============================================================================
// Name and Properties Tests
// =============================================================================

TEST_F(PinnedMemorySpaceTest, NameIsPinned) {
    EXPECT_STREQ(dtl::pinned_memory_space::name(), "pinned");
}

TEST_F(PinnedMemorySpaceTest, PropertiesCorrect) {
    auto props = dtl::pinned_memory_space::properties();
    EXPECT_TRUE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_FALSE(props.pageable);
}

// =============================================================================
// Allocation Tests (Fallback Path)
// =============================================================================

TEST_F(PinnedMemorySpaceTest, FallbackAllocation) {
    void* ptr = dtl::pinned_memory_space::allocate(256);
    EXPECT_NE(ptr, nullptr);
    dtl::pinned_memory_space::deallocate(ptr, 256);
}

TEST_F(PinnedMemorySpaceTest, FallbackDeallocation) {
    void* ptr = dtl::pinned_memory_space::allocate(128);
    EXPECT_NE(ptr, nullptr);
    dtl::pinned_memory_space::deallocate(ptr, 128);
}

TEST_F(PinnedMemorySpaceTest, AllocateZero) {
    void* ptr = dtl::pinned_memory_space::allocate(0);
    // Zero-size allocation returns nullptr
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(PinnedMemorySpaceTest, NullDeallocate) {
    // Deallocating nullptr should be a no-op
    dtl::pinned_memory_space::deallocate(nullptr, 0);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(PinnedMemorySpaceTest, TotalAllocated) {
    EXPECT_EQ(dtl::pinned_memory_space::total_allocated(), 0u);

    void* ptr = dtl::pinned_memory_space::allocate(512);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(dtl::pinned_memory_space::total_allocated(), 512u);
    dtl::pinned_memory_space::deallocate(ptr, 512);
    EXPECT_EQ(dtl::pinned_memory_space::total_allocated(), 0u);
}

TEST_F(PinnedMemorySpaceTest, PeakAllocated) {
    EXPECT_EQ(dtl::pinned_memory_space::peak_allocated(), 0u);

    void* r1 = dtl::pinned_memory_space::allocate(256);
    EXPECT_NE(r1, nullptr);
    EXPECT_EQ(dtl::pinned_memory_space::peak_allocated(), 256u);

    void* r2 = dtl::pinned_memory_space::allocate(512);
    EXPECT_NE(r2, nullptr);
    EXPECT_EQ(dtl::pinned_memory_space::peak_allocated(), 768u);
    dtl::pinned_memory_space::deallocate(r2, 512);

    // Peak should remain at high-water mark
    EXPECT_EQ(dtl::pinned_memory_space::peak_allocated(), 768u);
    dtl::pinned_memory_space::deallocate(r1, 256);
}

TEST_F(PinnedMemorySpaceTest, MultipleAllocations) {
    constexpr int num_allocs = 5;
    void* ptrs[num_allocs] = {};
    constexpr dtl::size_type alloc_size = 64;

    for (int i = 0; i < num_allocs; ++i) {
        ptrs[i] = dtl::pinned_memory_space::allocate(alloc_size);
        EXPECT_NE(ptrs[i], nullptr);
    }

    for (int i = 0; i < num_allocs; ++i) {
        dtl::pinned_memory_space::deallocate(ptrs[i], alloc_size);
    }

    EXPECT_EQ(dtl::pinned_memory_space::total_allocated(), 0u);
}

TEST_F(PinnedMemorySpaceTest, LargeAllocation) {
    constexpr dtl::size_type large_size = 1024 * 1024;  // 1 MB
    void* ptr = dtl::pinned_memory_space::allocate(large_size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(dtl::pinned_memory_space::total_allocated(), large_size);
    dtl::pinned_memory_space::deallocate(ptr, large_size);
}

TEST_F(PinnedMemorySpaceTest, AllocateDeallocateRoundtrip) {
    void* ptr = dtl::pinned_memory_space::allocate(1024);
    EXPECT_NE(ptr, nullptr);
    dtl::pinned_memory_space::deallocate(ptr, 1024);
}

TEST_F(PinnedMemorySpaceTest, MemsetAfterAlloc) {
    void* ptr = dtl::pinned_memory_space::allocate(256);
    EXPECT_NE(ptr, nullptr);
    // Should be able to write to allocated memory
    std::memset(ptr, 0xAB, 256);
    auto* bytes = static_cast<unsigned char*>(ptr);
    EXPECT_EQ(bytes[0], 0xAB);
    EXPECT_EQ(bytes[255], 0xAB);
    dtl::pinned_memory_space::deallocate(ptr, 256);
}

// =============================================================================
// Static Interface Tests
// =============================================================================

TEST_F(PinnedMemorySpaceTest, StaticInterface) {
    // Verify static methods work without an instance
    EXPECT_STREQ(dtl::pinned_memory_space::name(), "pinned");
    auto props = dtl::pinned_memory_space::properties();
    EXPECT_TRUE(props.host_accessible);
}

TEST_F(PinnedMemorySpaceTest, AlignedAllocation) {
    void* ptr = dtl::pinned_memory_space::allocate(512, 64);
    EXPECT_NE(ptr, nullptr);
    dtl::pinned_memory_space::deallocate(ptr, 512);
}

// =============================================================================
// Concept Conformance Tests
// =============================================================================

TEST_F(PinnedMemorySpaceTest, SatisfiesMemorySpaceConcept) {
    // This is also verified at compile time via static_assert in the header.
    EXPECT_TRUE((dtl::MemorySpace<dtl::pinned_memory_space>));
}

// =============================================================================
// Thread Safety Tests (T06)
// =============================================================================

TEST_F(PinnedMemorySpaceTest, ConcurrentAllocations) {
    constexpr int num_threads = 4;
    constexpr int allocs_per_thread = 10;
    constexpr dtl::size_type alloc_size = 64;

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> thread_ptrs(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t_idx = static_cast<std::size_t>(t), &thread_ptrs]() {
            for (int i = 0; i < allocs_per_thread; ++i) {
                void* ptr = dtl::pinned_memory_space::allocate(alloc_size);
                if (ptr) {
                    thread_ptrs[t_idx].push_back(ptr);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    // Deallocate all
    for (auto& ptrs : thread_ptrs) {
        for (void* p : ptrs) {
            dtl::pinned_memory_space::deallocate(p, alloc_size);
        }
    }

    EXPECT_EQ(dtl::pinned_memory_space::total_allocated(), 0u);
}

// =============================================================================
// Memory Space Traits Tests
// =============================================================================

TEST_F(PinnedMemorySpaceTest, TraitsCorrect) {
    EXPECT_TRUE(dtl::memory_space_traits<dtl::pinned_memory_space>::is_host_space);
    EXPECT_FALSE(dtl::memory_space_traits<dtl::pinned_memory_space>::is_device_space);
    EXPECT_FALSE(dtl::memory_space_traits<dtl::pinned_memory_space>::is_unified_space);
    EXPECT_TRUE(dtl::memory_space_traits<dtl::pinned_memory_space>::is_thread_safe);
}

}  // namespace dtl::test
