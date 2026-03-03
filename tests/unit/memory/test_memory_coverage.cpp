// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_memory_coverage.cpp
/// @brief Expanded unit tests for the DTL memory module
/// @details Phase 14 T01: host_memory_space, allocator, copy, pointer_utils,
///          prefetch_policy coverage.

#include <dtl/memory/host_memory_space.hpp>
#include <dtl/memory/allocator.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <dtl/memory/copy.hpp>
#pragma GCC diagnostic pop

#include <dtl/memory/pointer_utils.hpp>
#include <dtl/memory/prefetch_policy.hpp>
#include <dtl/memory/memory_space_base.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace dtl::test {

// =============================================================================
// host_memory_space Tests
// =============================================================================

TEST(HostMemorySpaceTest, AllocateNonZero) {
    void* ptr = host_memory_space::allocate(128);
    ASSERT_NE(ptr, nullptr);
    host_memory_space::deallocate(ptr, 128);
}

TEST(HostMemorySpaceTest, AllocateZeroSize) {
    // malloc(0) is implementation-defined (may return nullptr or unique ptr)
    void* ptr = host_memory_space::allocate(0);
    // Just ensure it doesn't crash
    host_memory_space::deallocate(ptr, 0);
}

TEST(HostMemorySpaceTest, AllocateAligned) {
    constexpr size_type alignment = 64;
    void* ptr = host_memory_space::allocate(256, alignment);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, alignment));
    host_memory_space::deallocate(ptr, 256, alignment);
}

TEST(HostMemorySpaceTest, AllocateAligned16) {
    constexpr size_type alignment = 16;
    void* ptr = host_memory_space::allocate(100, alignment);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, alignment));
    host_memory_space::deallocate(ptr, 100, alignment);
}

TEST(HostMemorySpaceTest, DeallocateNullptr) {
    // free(nullptr) is a no-op per the C standard
    host_memory_space::deallocate(nullptr, 0);
}

TEST(HostMemorySpaceTest, Properties) {
    auto props = host_memory_space::properties();
    EXPECT_TRUE(props.host_accessible);
    EXPECT_FALSE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_TRUE(props.pageable);
    EXPECT_GT(props.alignment, 0u);
}

TEST(HostMemorySpaceTest, Name) {
    EXPECT_STREQ(host_memory_space::name(), "host");
}

TEST(HostMemorySpaceTest, AllocateTyped) {
    double* ptr = host_memory_space::allocate_typed<double>(10);
    ASSERT_NE(ptr, nullptr);
    // Write to verify it's usable memory
    for (int i = 0; i < 10; ++i) {
        ptr[i] = static_cast<double>(i);
    }
    EXPECT_DOUBLE_EQ(ptr[5], 5.0);
    host_memory_space::deallocate_typed(ptr, 10);
}

TEST(HostMemorySpaceTest, ConstructDestroy) {
    int* ptr = host_memory_space::allocate_typed<int>(1);
    ASSERT_NE(ptr, nullptr);
    host_memory_space::construct(ptr, 42);
    EXPECT_EQ(*ptr, 42);
    host_memory_space::destroy(ptr);
    host_memory_space::deallocate_typed(ptr, 1);
}

TEST(HostMemorySpaceTest, ConstructString) {
    // Test construct/destroy with a non-trivial type
    auto* ptr = static_cast<std::string*>(
        host_memory_space::allocate(sizeof(std::string), alignof(std::string)));
    ASSERT_NE(ptr, nullptr);
    host_memory_space::construct(ptr, std::string("hello"));
    EXPECT_EQ(*ptr, "hello");
    host_memory_space::destroy(ptr);
    host_memory_space::deallocate(ptr, sizeof(std::string), alignof(std::string));
}

TEST(HostMemorySpaceTest, TraitsSpecialization) {
    EXPECT_TRUE(memory_space_traits<host_memory_space>::is_host_space);
    EXPECT_FALSE(memory_space_traits<host_memory_space>::is_device_space);
    EXPECT_FALSE(memory_space_traits<host_memory_space>::is_unified_space);
    EXPECT_TRUE(memory_space_traits<host_memory_space>::is_thread_safe);
}

// =============================================================================
// memory_space_allocator Tests
// =============================================================================

TEST(MemorySpaceAllocatorTest, DefaultConstruction) {
    memory_space_allocator<int> alloc;
    (void)alloc;
}

TEST(MemorySpaceAllocatorTest, AllocateDeallocate) {
    memory_space_allocator<int> alloc;
    int* ptr = alloc.allocate(10);
    ASSERT_NE(ptr, nullptr);
    for (int i = 0; i < 10; ++i) {
        ptr[i] = i * 2;
    }
    EXPECT_EQ(ptr[5], 10);
    alloc.deallocate(ptr, 10);
}

TEST(MemorySpaceAllocatorTest, AllocateThrowsOnOverflow) {
    memory_space_allocator<int> alloc;
    EXPECT_THROW(
        { [[maybe_unused]] auto* p = alloc.allocate(std::numeric_limits<std::size_t>::max()); },
        std::bad_alloc);
}

TEST(MemorySpaceAllocatorTest, EqualityComparison) {
    memory_space_allocator<int> a;
    memory_space_allocator<double> b;
    // Stateless allocators of the same space are always equal
    EXPECT_TRUE(a == b);
}

TEST(MemorySpaceAllocatorTest, CopyConstruction) {
    memory_space_allocator<int> a;
    memory_space_allocator<int> b(a);
    EXPECT_TRUE(a == b);
}

TEST(MemorySpaceAllocatorTest, RebindConstruction) {
    memory_space_allocator<int> a;
    memory_space_allocator<double> b(a);
    EXPECT_TRUE(a == b);
}

TEST(MemorySpaceAllocatorTest, SpaceName) {
    memory_space_allocator<int> alloc;
    EXPECT_STREQ(alloc.space_name(), "host");
}

TEST(MemorySpaceAllocatorTest, SpaceProperties) {
    memory_space_allocator<int> alloc;
    auto props = alloc.space_properties();
    EXPECT_TRUE(props.host_accessible);
}

TEST(MemorySpaceAllocatorTest, PropagationTraits) {
    using alloc_t = memory_space_allocator<int>;
    EXPECT_TRUE(alloc_t::propagate_on_container_copy_assignment::value);
    EXPECT_TRUE(alloc_t::propagate_on_container_move_assignment::value);
    EXPECT_TRUE(alloc_t::propagate_on_container_swap::value);
}

TEST(MemorySpaceAllocatorTest, UseWithVector) {
    std::vector<int, memory_space_allocator<int>> vec(10, 42);
    EXPECT_EQ(vec.size(), 10u);
    EXPECT_EQ(vec[0], 42);
    vec.push_back(99);
    EXPECT_EQ(vec.back(), 99);
}

// =============================================================================
// allocator_traits_ext Tests
// =============================================================================

TEST(AllocatorTraitsExtTest, DefaultTraits) {
    using traits = allocator_traits_ext<std::allocator<int>>;
    EXPECT_TRUE(traits::is_host_allocator);
    EXPECT_FALSE(traits::is_device_allocator);
    EXPECT_FALSE(traits::supports_streams);
}

TEST(AllocatorTraitsExtTest, MemorySpaceAllocatorTraits) {
    using alloc_t = memory_space_allocator<int, host_memory_space>;
    using traits = allocator_traits_ext<alloc_t>;
    EXPECT_TRUE(traits::is_host_allocator);
    EXPECT_FALSE(traits::is_device_allocator);
    EXPECT_FALSE(traits::supports_streams);
}

// =============================================================================
// copy.hpp Tests (non-CUDA path: memcpy fallback)
// =============================================================================

TEST(CopyTest, CopyToHostFromHostMemory) {
    // Without CUDA, copy_to_host uses memcpy
    std::vector<int> src = {1, 2, 3, 4, 5};
    auto result = copy_to_host(src.data(), size_type{5});
    ASSERT_EQ(result.size(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(result[i], src[i]);
    }
}

TEST(CopyTest, CopyToHostZeroSize) {
    int dummy = 0;
    auto result = copy_to_host(&dummy, size_type{0});
    EXPECT_TRUE(result.empty());
}

TEST(CopyTest, CopyFromHostToHost) {
    std::vector<int> src = {10, 20, 30};
    std::vector<int> dst(3, 0);
    auto res = copy_from_host(src.data(), dst.data(), size_type{3});
    EXPECT_TRUE(res.success);
    EXPECT_EQ(res.bytes_copied, 3 * sizeof(int));
    EXPECT_EQ(res.error_code, 0);
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 20);
    EXPECT_EQ(dst[2], 30);
}

TEST(CopyTest, CopyFromHostZeroCount) {
    int src = 0, dst = 0;
    auto res = copy_from_host(&src, &dst, size_type{0});
    EXPECT_TRUE(res.success);
    EXPECT_EQ(res.bytes_copied, 0u);
}

TEST(CopyTest, CopyDeviceToDevice) {
    // Without CUDA, this uses memcpy
    std::vector<double> src = {1.1, 2.2, 3.3};
    std::vector<double> dst(3, 0.0);
    auto res = copy_device_to_device(src.data(), dst.data(), size_type{3});
    EXPECT_TRUE(res.success);
    EXPECT_DOUBLE_EQ(dst[0], 1.1);
    EXPECT_DOUBLE_EQ(dst[2], 3.3);
}

TEST(CopyTest, CopyResultBoolConversion) {
    copy_result ok{.success = true, .bytes_copied = 10, .error_code = 0};
    EXPECT_TRUE(static_cast<bool>(ok));

    copy_result fail{.success = false, .bytes_copied = 0, .error_code = 1};
    EXPECT_FALSE(static_cast<bool>(fail));
}

TEST(CopyTest, CopyDirectionTags) {
    // Just verify the tag types exist and are distinct
    [[maybe_unused]] auto h2d = host_to_device;
    [[maybe_unused]] auto d2h = device_to_host;
    [[maybe_unused]] auto d2d = device_to_device;
    SUCCEED();
}

// =============================================================================
// pointer_utils Tests
// =============================================================================

TEST(PointerUtilsTest, QueryHostPointerKind) {
    // Without CUDA, query_pointer_kind returns unknown for any pointer
    int x = 42;
    auto kind = query_pointer_kind(&x);
    EXPECT_EQ(kind, pointer_kind::unknown);
}

TEST(PointerUtilsTest, QueryNullPointer) {
    auto kind = query_pointer_kind(nullptr);
    EXPECT_EQ(kind, pointer_kind::unknown);
}

TEST(PointerUtilsTest, IsHostAccessible) {
    int x = 0;
    // Without CUDA, host pointers return unknown, which is host-accessible
    EXPECT_TRUE(is_host_accessible(&x));
}

TEST(PointerUtilsTest, IsDeviceAccessible) {
    int x = 0;
    // Without CUDA, nothing is device-accessible
    EXPECT_FALSE(is_device_accessible(&x));
}

TEST(PointerUtilsTest, IsGpuAwareMpiWithoutCuda) {
    EXPECT_FALSE(is_gpu_aware_mpi());
}

TEST(PointerUtilsTest, PointerKindEnumValues) {
    EXPECT_NE(pointer_kind::host, pointer_kind::device);
    EXPECT_NE(pointer_kind::device, pointer_kind::managed);
    EXPECT_NE(pointer_kind::managed, pointer_kind::unregistered);
    EXPECT_NE(pointer_kind::unregistered, pointer_kind::unknown);
}

// =============================================================================
// prefetch_policy Tests
// =============================================================================

TEST(PrefetchPolicyCoverageTest, EnumValues) {
    EXPECT_NE(prefetch_policy::none, prefetch_policy::to_device);
    EXPECT_NE(prefetch_policy::to_device, prefetch_policy::to_host);
    EXPECT_NE(prefetch_policy::to_host, prefetch_policy::bidirectional);
}

TEST(PrefetchPolicyCoverageTest, ToStringNone) {
    EXPECT_EQ(to_string(prefetch_policy::none), "none");
}

TEST(PrefetchPolicyCoverageTest, ToStringToDevice) {
    EXPECT_EQ(to_string(prefetch_policy::to_device), "to_device");
}

TEST(PrefetchPolicyCoverageTest, ToStringToHost) {
    EXPECT_EQ(to_string(prefetch_policy::to_host), "to_host");
}

TEST(PrefetchPolicyCoverageTest, ToStringBidirectional) {
    EXPECT_EQ(to_string(prefetch_policy::bidirectional), "bidirectional");
}

TEST(PrefetchPolicyCoverageTest, PrefetchHintDefaults) {
    prefetch_hint hint{};
    EXPECT_EQ(hint.policy, prefetch_policy::none);
    EXPECT_EQ(hint.device_id, 0);
    EXPECT_EQ(hint.offset, 0u);
    EXPECT_EQ(hint.size, 0u);
}

TEST(PrefetchPolicyCoverageTest, MakeDevicePrefetch) {
    auto hint = make_device_prefetch(2);
    EXPECT_EQ(hint.policy, prefetch_policy::to_device);
    EXPECT_EQ(hint.device_id, 2);
    EXPECT_EQ(hint.offset, 0u);
    EXPECT_EQ(hint.size, 0u);
}

TEST(PrefetchPolicyCoverageTest, MakeDevicePrefetchDefault) {
    auto hint = make_device_prefetch();
    EXPECT_EQ(hint.device_id, 0);
}

TEST(PrefetchPolicyCoverageTest, MakeHostPrefetch) {
    auto hint = make_host_prefetch();
    EXPECT_EQ(hint.policy, prefetch_policy::to_host);
    EXPECT_EQ(hint.device_id, 0);
}

// =============================================================================
// memory_space_base utilities Tests
// =============================================================================

TEST(MemoryUtilsTest, IsAligned) {
    alignas(64) char buf[128] = {};
    void* p = buf;
    EXPECT_TRUE(is_aligned(p, 1));
    EXPECT_TRUE(is_aligned(p, 2));
    // buf is aligned to 64, so definitely aligned to 4, 8, 16, 32, 64
    EXPECT_TRUE(is_aligned(p, 64));
}

TEST(MemoryUtilsTest, IsAlignedUnaligned) {
    alignas(64) char buf[128] = {};
    void* p = buf + 1;
    // buf+1 won't be 64-aligned
    EXPECT_FALSE(is_aligned(p, 64));
}

TEST(MemoryUtilsTest, AlignSize) {
    EXPECT_EQ(align_size(1, 8), 8u);
    EXPECT_EQ(align_size(8, 8), 8u);
    EXPECT_EQ(align_size(9, 8), 16u);
    EXPECT_EQ(align_size(0, 8), 0u);
    EXPECT_EQ(align_size(15, 16), 16u);
    EXPECT_EQ(align_size(16, 16), 16u);
    EXPECT_EQ(align_size(17, 16), 32u);
}

TEST(MemoryUtilsTest, DefaultAlignment) {
    EXPECT_EQ(default_alignment<int>(), alignof(int));
    EXPECT_EQ(default_alignment<double>(), alignof(double));
}

TEST(MemoryUtilsTest, PlatformAlignment) {
    EXPECT_EQ(platform_alignment(), alignof(std::max_align_t));
}

TEST(MemoryUtilsTest, ZeroMemory) {
    int data[4] = {1, 2, 3, 4};
    zero_memory(data, sizeof(data));
    for (auto& d : data) {
        EXPECT_EQ(d, 0);
    }
}

TEST(MemoryUtilsTest, CopyMemory) {
    int src[3] = {10, 20, 30};
    int dst[3] = {};
    copy_memory(dst, src, sizeof(src));
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 20);
    EXPECT_EQ(dst[2], 30);
}

TEST(AllocationResultTest, SuccessCheck) {
    allocation_result res{.ptr = reinterpret_cast<void*>(0x1000), .size = 64, .alignment = 8};
    EXPECT_TRUE(res.success());
    EXPECT_TRUE(static_cast<bool>(res));
}

TEST(AllocationResultTest, FailureCheck) {
    allocation_result res{};
    EXPECT_FALSE(res.success());
    EXPECT_FALSE(static_cast<bool>(res));
}

}  // namespace dtl::test
