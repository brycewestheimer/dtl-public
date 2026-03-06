// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_placement_policies.cpp
/// @brief Integration tests for placement policies and CUDA memory integration
/// @details Tests that placement policies correctly wire to memory spaces and
///          that containers properly allocate memory based on placement.

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/memory/default_allocator.hpp>
#include <dtl/memory/host_memory_space.hpp>
#include <dtl/containers/distributed_array.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/containers/distributed_tensor.hpp>
#include <dtl/policies/placement/host_only.hpp>
#include <dtl/policies/placement/device_only.hpp>
#include <dtl/policies/placement/device_preferred.hpp>
#include <dtl/policies/placement/unified_memory.hpp>

#if DTL_ENABLE_CUDA
#include <dtl/memory/cuda_memory_space.hpp>
#include <dtl/memory/cuda_device_memory_space.hpp>
#include <dtl/cuda/device_guard.hpp>
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <type_traits>

namespace dtl::test {

namespace {

template <typename Container>
concept has_local_view = requires(Container& container) {
    container.local_view();
};

template <typename Container>
concept has_local_span = requires(Container& container) {
    container.local_span();
};

template <typename Container>
concept has_device_view = requires(Container& container) {
    container.device_view();
};

template <typename Container>
concept has_local_index_access = requires(Container& container) {
    container.local(0);
};

template <typename Container>
concept has_value_resize = requires(Container& container,
                                    typename Container::size_type size,
                                    typename Container::value_type value) {
    container.resize(size, value);
};

template <typename Container>
concept has_fill_member = requires(Container& container,
                                   typename Container::value_type value) {
    container.fill(value);
};

template <typename Container>
concept has_tensor_local_index_access = requires(Container& container) {
    container.local(typename Container::index_type{});
};

}  // namespace

// =============================================================================
// Compile-Time Allocator Selection Tests
// =============================================================================

TEST(PlacementPolicyTest, HostOnlySelectsHostAllocator) {
    using alloc_t = select_allocator_t<float, host_only>;
    using expected_t = memory_space_allocator<float, host_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>,
                  "host_only should select host_memory_space allocator");
    SUCCEED();
}

TEST(PlacementPolicyTest, DefaultPlacementSelectsHostAllocator) {
    using alloc_t = select_allocator_t<int, host_only>;
    using expected_t = memory_space_allocator<int, host_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>,
                  "Default placement should select host allocator");
    SUCCEED();
}

#if DTL_ENABLE_CUDA

TEST(PlacementPolicyTest, DeviceOnlySelectsCudaDeviceAllocator) {
    using alloc_t = select_allocator_t<float, device_only<0>>;
    using expected_t = memory_space_allocator<float, cuda::cuda_device_memory_space_for<0>>;

    static_assert(std::is_same_v<alloc_t, expected_t>,
                  "device_only<0> should select cuda_device_memory_space allocator");
    SUCCEED();
}

TEST(PlacementPolicyTest, UnifiedMemorySelectsCudaUnifiedAllocator) {
    using alloc_t = select_allocator_t<double, unified_memory>;
    using expected_t = memory_space_allocator<double, cuda::cuda_unified_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>,
                  "unified_memory should select cuda_unified_memory_space allocator");
    SUCCEED();
}

TEST(PlacementPolicyTest, DevicePreferredSelectsCudaDeviceAllocator) {
    using alloc_t = select_allocator_t<int, device_preferred>;
    using expected_t = memory_space_allocator<int, cuda::cuda_device_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>,
                  "device_preferred should select cuda_device_memory_space allocator when CUDA enabled");
    SUCCEED();
}

#else  // !DTL_ENABLE_CUDA

TEST(PlacementPolicyTest, DevicePreferredFallsBackToHost) {
    using alloc_t = select_allocator_t<float, device_preferred>;
    using expected_t = memory_space_allocator<float, host_memory_space>;

    static_assert(std::is_same_v<alloc_t, expected_t>,
                  "device_preferred should fall back to host allocator when CUDA disabled");
    SUCCEED();
}

#endif  // DTL_ENABLE_CUDA

// =============================================================================
// Memory Space Properties Tests
// =============================================================================

TEST(MemorySpacePropertiesTest, HostMemorySpaceProperties) {
    auto props = host_memory_space::properties();

    EXPECT_TRUE(props.host_accessible);
    EXPECT_FALSE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_TRUE(props.pageable);
}

#if DTL_ENABLE_CUDA

TEST(MemorySpacePropertiesTest, CudaDeviceMemorySpaceProperties) {
    auto props = cuda::cuda_device_memory_space::properties();

    EXPECT_FALSE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_FALSE(props.pageable);
    EXPECT_EQ(props.alignment, 256);
}

TEST(MemorySpacePropertiesTest, CudaUnifiedMemorySpaceProperties) {
    auto props = cuda::cuda_unified_memory_space::properties();

    EXPECT_TRUE(props.host_accessible);
    EXPECT_TRUE(props.device_accessible);
    EXPECT_TRUE(props.unified);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_FALSE(props.pageable);
    EXPECT_EQ(props.alignment, 256);
}

// =============================================================================
// Memory Space Traits Tests
// =============================================================================

TEST(MemorySpaceTraitsTest, HostSpaceTraits) {
    using traits = memory_space_traits<host_memory_space>;

    EXPECT_TRUE(traits::is_host_space);
    EXPECT_FALSE(traits::is_device_space);
    EXPECT_FALSE(traits::is_unified_space);
}

TEST(MemorySpaceTraitsTest, CudaDeviceSpaceTraits) {
    using traits = memory_space_traits<cuda::cuda_device_memory_space>;

    EXPECT_FALSE(traits::is_host_space);
    EXPECT_TRUE(traits::is_device_space);
    EXPECT_FALSE(traits::is_unified_space);
}

TEST(MemorySpaceTraitsTest, CudaUnifiedSpaceTraits) {
    using traits = memory_space_traits<cuda::cuda_unified_memory_space>;

    EXPECT_TRUE(traits::is_host_space);  // Accessible from host
    EXPECT_TRUE(traits::is_device_space);  // Accessible from device
    EXPECT_TRUE(traits::is_unified_space);
}

// =============================================================================
// Container Placement Policy Integration Tests
// =============================================================================

class CudaPlacementTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = cuda::device_count();
        if (device_count <= 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

TEST_F(CudaPlacementTest, DistributedVectorDeviceOnlyAllocatesOnDevice) {
    using vec_t = distributed_vector<float, device_only<0>>;

    static_assert(!has_local_view<vec_t>);
    static_assert(!has_local_index_access<vec_t>);
    static_assert(has_device_view<vec_t>);
    static_assert(!std::is_constructible_v<vec_t, typename vec_t::size_type, const float&>);
    static_assert(!has_value_resize<vec_t>);

    // Verify allocator type at compile time - now uses device-specific space
    static_assert(std::is_same_v<
        typename vec_t::allocator_type,
        memory_space_allocator<float, cuda::cuda_device_memory_space_for<0>>
    >);

    // Create vector
    vec_t vec(1000, 1, 0);  // 1000 elements, 1 rank, rank 0

    // Verify placement policy properties
    EXPECT_FALSE(vec_t::is_host_accessible());
    EXPECT_TRUE(vec_t::is_device_accessible());

    // Verify device affinity
    EXPECT_EQ(vec.device_id(), 0);
    EXPECT_TRUE(vec.has_device_affinity());

    // Verify the pointer is device memory
    float* ptr = vec.local_data();
    ASSERT_NE(ptr, nullptr);

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);

    // Verify it's on the correct device
    EXPECT_EQ(attrs.device, 0);

    auto view = vec.device_view();
    EXPECT_EQ(view.data(), ptr);
    EXPECT_EQ(view.size(), vec.local_size());
    EXPECT_EQ(view.device_id(), 0);
}

TEST_F(CudaPlacementTest, DistributedVectorUnifiedMemoryAllocatesManaged) {
    using vec_t = distributed_vector<float, unified_memory>;

    static_assert(has_local_view<vec_t>);
    static_assert(has_local_index_access<vec_t>);
    static_assert(has_device_view<vec_t>);
    static_assert(std::is_constructible_v<vec_t, typename vec_t::size_type, const float&>);
    static_assert(has_value_resize<vec_t>);

    // Verify allocator type at compile time
    static_assert(std::is_same_v<
        typename vec_t::allocator_type,
        memory_space_allocator<float, cuda::cuda_unified_memory_space>
    >);

    // Create vector
    vec_t vec(1000, 1, 0);

    // Verify placement policy properties
    EXPECT_TRUE(vec_t::is_host_accessible());
    EXPECT_TRUE(vec_t::is_device_accessible());

    // Verify the pointer is managed memory
    float* ptr = vec.local_data();
    ASSERT_NE(ptr, nullptr);

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeManaged);

    auto device_view = vec.device_view();
    EXPECT_EQ(device_view.data(), ptr);
    EXPECT_EQ(device_view.size(), vec.local_size());
}

TEST_F(CudaPlacementTest, DeviceOnlyGlobalViewDoesNotExposeHostFastPath) {
    using vec_t = distributed_vector<int, device_only<0>>;

    vec_t vec(16, 1, 0);
    auto global = vec.global_view();
    auto ref = global[0];

    EXPECT_TRUE(global.is_local(0));
    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.get().has_error());
}

TEST_F(CudaPlacementTest, DistributedTensorDeviceOnlyAllocatesOnDevice) {
    using tensor_t = distributed_tensor<double, 2, device_only<0>>;

    static_assert(!has_local_view<tensor_t>);
    static_assert(!has_local_span<tensor_t>);
    static_assert(!has_tensor_local_index_access<tensor_t>);
    static_assert(has_device_view<tensor_t>);
    static_assert(!std::is_constructible_v<
                  tensor_t,
                  const typename tensor_t::extent_type&,
                  typename tensor_t::size_type,
                  rank_t,
                  rank_t,
                  const double&>);

    // Verify allocator type at compile time - now uses device-specific space
    static_assert(std::is_same_v<
        typename tensor_t::allocator_type,
        memory_space_allocator<double, cuda::cuda_device_memory_space_for<0>>
    >);

    // Create 100x100 matrix
    tensor_t tensor({100, 100}, 0, 1, 0);

    // Verify placement
    EXPECT_FALSE(tensor_t::is_host_accessible());
    EXPECT_TRUE(tensor_t::is_device_accessible());

    // Verify the pointer is device memory
    double* ptr = tensor.local_data();
    ASSERT_NE(ptr, nullptr);

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);

    // Verify it's on the correct device
    EXPECT_EQ(attrs.device, 0);

    auto view = tensor.device_view();
    EXPECT_EQ(view.data(), ptr);
    EXPECT_EQ(view.size(), tensor.local_size());
    EXPECT_EQ(view.device_id(), 0);
}

TEST_F(CudaPlacementTest, DeviceOnlyTensorGlobalAccessDoesNotExposeHostFastPath) {
    using tensor_t = distributed_tensor<int, 2, device_only<0>>;

    tensor_t tensor({4, 4}, 0, 1, 0);
    auto ref = tensor.global(typename tensor_t::index_type{0, 0});

    EXPECT_FALSE(ref.is_local());
    EXPECT_TRUE(ref.get().has_error());
}

TEST_F(CudaPlacementTest, DistributedArrayDeviceOnlyExposesDeviceViewOnly) {
    using array_t = distributed_array<int, 64, device_only<0>>;

    static_assert(!has_local_view<array_t>);
    static_assert(!has_local_index_access<array_t>);
    static_assert(has_device_view<array_t>);
    static_assert(!has_fill_member<array_t>);

    array_t arr;

    int* ptr = arr.local_data();
    ASSERT_NE(ptr, nullptr);

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
    EXPECT_EQ(attrs.device, 0);

    auto view = arr.device_view();
    EXPECT_EQ(view.data(), ptr);
    EXPECT_EQ(view.size(), arr.local_size());
    EXPECT_EQ(view.device_id(), 0);
}

TEST_F(CudaPlacementTest, UnifiedMemoryAccessibleFromHost) {
    using vec_t = distributed_vector<int, unified_memory>;

    vec_t vec(100, 1, 0);
    auto local = vec.local_view();

    // Should be able to write from host
    for (size_t i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i * 2);
    }

    // Should be able to read from host
    for (size_t i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<int>(i * 2));
    }
}

TEST_F(CudaPlacementTest, DeviceOnlyCanBeUsedWithKernel) {
    using vec_t = distributed_vector<float, device_only<0>>;

    const size_t N = 1000;
    vec_t vec(N, 1, 0);

    // Initialize on device using cudaMemset
    cudaMemset(vec.local_data(), 0, N * sizeof(float));

    // Copy to host for verification
    std::vector<float> host_data(N);
    cudaMemcpy(host_data.data(), vec.local_data(), N * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(host_data[i], 0.0f);
    }
}

// =============================================================================
// Static Memory Space Concept Verification
// =============================================================================

TEST(MemorySpaceConceptTest, CudaDeviceMemorySpaceSatisfiesConcept) {
    static_assert(MemorySpace<cuda::cuda_device_memory_space>,
                  "cuda_device_memory_space must satisfy MemorySpace concept");
    SUCCEED();
}

TEST(MemorySpaceConceptTest, CudaUnifiedMemorySpaceSatisfiesConcept) {
    static_assert(MemorySpace<cuda::cuda_unified_memory_space>,
                  "cuda_unified_memory_space must satisfy MemorySpace concept");
    SUCCEED();
}

// =============================================================================
// Allocator Tests
// =============================================================================

TEST_F(CudaPlacementTest, CudaDeviceAllocatorAllocates) {
    using alloc_t = memory_space_allocator<float, cuda::cuda_device_memory_space>;
    alloc_t alloc;

    float* ptr = alloc.allocate(100);
    ASSERT_NE(ptr, nullptr);

    // Verify it's device memory
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);

    alloc.deallocate(ptr, 100);
}

TEST_F(CudaPlacementTest, CudaUnifiedAllocatorAllocates) {
    using alloc_t = memory_space_allocator<int, cuda::cuda_unified_memory_space>;
    alloc_t alloc;

    int* ptr = alloc.allocate(100);
    ASSERT_NE(ptr, nullptr);

    // Verify it's managed memory
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeManaged);

    alloc.deallocate(ptr, 100);
}

// =============================================================================
// Prefetch Tests
// =============================================================================

TEST_F(CudaPlacementTest, UnifiedMemoryPrefetchToDevice) {
    using alloc_t = memory_space_allocator<float, cuda::cuda_unified_memory_space>;
    alloc_t alloc;

    const size_t count = 10000;
    float* ptr = alloc.allocate(count);
    ASSERT_NE(ptr, nullptr);

    // Initialize on host
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    // Prefetch to device
    cuda::cuda_unified_memory_space::prefetch_to_device(ptr, count * sizeof(float), 0);
    cudaDeviceSynchronize();

    // No exception or error should occur
    alloc.deallocate(ptr, count);
}

#endif  // DTL_ENABLE_CUDA

// =============================================================================
// Host-Only Container Tests (always run)
// =============================================================================

TEST(HostPlacementTest, DistributedVectorHostOnlyAllocatesOnHost) {
    using vec_t = distributed_vector<float, host_only>;

    // Verify allocator type at compile time
    static_assert(std::is_same_v<
        typename vec_t::allocator_type,
        memory_space_allocator<float, host_memory_space>
    >);

    // Create vector
    vec_t vec(1000, 1, 0);

    // Verify placement policy properties
    EXPECT_TRUE(vec_t::is_host_accessible());
    EXPECT_FALSE(vec_t::is_device_accessible());

    // Verify we can read/write
    auto local = vec.local_view();
    for (size_t i = 0; i < local.size(); ++i) {
        local[i] = static_cast<float>(i);
    }

    for (size_t i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<float>(i));
    }
}

TEST(HostPlacementTest, DefaultPlacementIsHostAccessible) {
    // Default distributed_vector with no placement policy specified
    distributed_vector<int> vec(100, 1, 0);

    // Default should be host_only
    using default_vec_t = distributed_vector<int>;
    EXPECT_TRUE(default_vec_t::is_host_accessible());
}

#if DTL_ENABLE_CUDA

// =============================================================================
// Multi-Device Allocation Tests
// =============================================================================

class CudaMultiDevicePlacementTest : public ::testing::Test {
protected:
    void SetUp() override {
        int count = cuda::device_count();
        if (count < 2) {
            GTEST_SKIP() << "Need at least 2 CUDA devices for multi-device tests";
        }
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

TEST_F(CudaMultiDevicePlacementTest, DeviceOnly0And1AllocateOnDifferentDevices) {
    using vec0_t = distributed_vector<float, device_only<0>>;
    using vec1_t = distributed_vector<float, device_only<1>>;

    // Record current device before creating containers
    int original_device = cuda::current_device_id();

    // Create vectors on different devices
    vec0_t vec0(1000, 1, 0);
    vec1_t vec1(1000, 1, 0);

    // Verify device affinity
    EXPECT_EQ(vec0.device_id(), 0);
    EXPECT_EQ(vec1.device_id(), 1);

    // Verify actual allocation locations
    cudaPointerAttributes attrs0, attrs1;
    cudaPointerGetAttributes(&attrs0, vec0.local_data());
    cudaPointerGetAttributes(&attrs1, vec1.local_data());

    EXPECT_EQ(attrs0.device, 0);
    EXPECT_EQ(attrs1.device, 1);

    // Verify current device was restored
    EXPECT_EQ(cuda::current_device_id(), original_device);
}

TEST_F(CudaMultiDevicePlacementTest, AllocationDoesNotChangeCallerDevice) {
    using vec0_t = distributed_vector<float, device_only<0>>;
    using vec1_t = distributed_vector<float, device_only<1>>;

    // Start on device 1
    cudaSetDevice(1);
    EXPECT_EQ(cuda::current_device_id(), 1);

    // Allocate on device 0
    vec0_t vec0(1000, 1, 0);
    EXPECT_EQ(cuda::current_device_id(), 1);  // Should still be on device 1

    // Allocate on device 1
    vec1_t vec1(1000, 1, 0);
    EXPECT_EQ(cuda::current_device_id(), 1);  // Should still be on device 1

    // Verify allocations are correct
    cudaPointerAttributes attrs0, attrs1;
    cudaPointerGetAttributes(&attrs0, vec0.local_data());
    cudaPointerGetAttributes(&attrs1, vec1.local_data());

    EXPECT_EQ(attrs0.device, 0);
    EXPECT_EQ(attrs1.device, 1);
}

TEST_F(CudaMultiDevicePlacementTest, InterleavedAllocationsAndDeallocations) {
    using vec0_t = distributed_vector<float, device_only<0>>;
    using vec1_t = distributed_vector<float, device_only<1>>;

    // Start on device 0
    cudaSetDevice(0);
    int original = cuda::current_device_id();

    for (int i = 0; i < 5; ++i) {
        {
            vec0_t v0(500, 1, 0);
            EXPECT_EQ(cuda::current_device_id(), original);

            vec1_t v1(500, 1, 0);
            EXPECT_EQ(cuda::current_device_id(), original);

            // Verify allocations
            cudaPointerAttributes a0, a1;
            cudaPointerGetAttributes(&a0, v0.local_data());
            cudaPointerGetAttributes(&a1, v1.local_data());

            EXPECT_EQ(a0.device, 0);
            EXPECT_EQ(a1.device, 1);
        }
        // Destructors called, verify device still correct
        EXPECT_EQ(cuda::current_device_id(), original);
    }
}

TEST_F(CudaMultiDevicePlacementTest, DeallocationRestoresDevice) {
    using vec1_t = distributed_vector<float, device_only<1>>;

    // Start on device 0
    cudaSetDevice(0);
    EXPECT_EQ(cuda::current_device_id(), 0);

    {
        vec1_t vec(1000, 1, 0);  // Allocates on device 1
        EXPECT_EQ(cuda::current_device_id(), 0);  // Restored after allocation
    }
    // Destructor deallocates on device 1, then restores to 0
    EXPECT_EQ(cuda::current_device_id(), 0);
}

TEST_F(CudaMultiDevicePlacementTest, DifferentTypesProduceDifferentAllocators) {
    // Compile-time verification that different devices produce different types
    static_assert(!std::is_same_v<
        distributed_vector<float, device_only<0>>,
        distributed_vector<float, device_only<1>>
    >, "device_only<0> and device_only<1> containers must be different types");

    static_assert(!std::is_same_v<
        select_allocator_t<float, device_only<0>>,
        select_allocator_t<float, device_only<1>>
    >, "device_only<0> and device_only<1> must select different allocators");

    SUCCEED();
}

// =============================================================================
// Device Guard Stress Tests
// =============================================================================

TEST_F(CudaMultiDevicePlacementTest, DeviceGuardRestoresUnderMultipleNesting) {
    cudaSetDevice(0);
    int original = cuda::current_device_id();

    {
        cuda::device_guard g1(1);
        EXPECT_EQ(cuda::current_device_id(), 1);

        // Allocate a container while on device 1
        distributed_vector<float, device_only<0>> vec0(100, 1, 0);
        EXPECT_EQ(cuda::current_device_id(), 1);  // Guard should restore to 1

        {
            cuda::device_guard g2(0);
            EXPECT_EQ(cuda::current_device_id(), 0);

            distributed_vector<float, device_only<1>> vec1(100, 1, 0);
            EXPECT_EQ(cuda::current_device_id(), 0);  // Guard should restore to 0
        }
        EXPECT_EQ(cuda::current_device_id(), 1);  // g2 destroyed, back to 1
    }
    EXPECT_EQ(cuda::current_device_id(), original);  // g1 destroyed, back to original
}

#endif  // DTL_ENABLE_CUDA

}  // namespace dtl::test
