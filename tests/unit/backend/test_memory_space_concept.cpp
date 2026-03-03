// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_memory_space_concept.cpp
/// @brief Unit tests for MemorySpace concept
/// @details Verifies concept requirements using mock implementations.

#include <dtl/backend/concepts/memory_space.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <new>

namespace dtl::test {

// =============================================================================
// Mock Memory Space Implementation
// =============================================================================

/// @brief Minimal mock memory space that satisfies the MemorySpace concept
class mock_memory_space {
public:
    using pointer = void*;
    using size_type = dtl::size_type;

    /// @brief Allocate memory
    [[nodiscard]] void* allocate(size_type size) {
        return std::malloc(size);
    }

    /// @brief Allocate aligned memory
    [[nodiscard]] void* allocate(size_type size, size_type alignment) {
        // Use aligned_alloc (C++17)
        if (alignment < sizeof(void*)) {
            alignment = sizeof(void*);
        }
        // Size must be multiple of alignment for aligned_alloc
        size_type aligned_size = ((size + alignment - 1) / alignment) * alignment;
        return std::aligned_alloc(alignment, aligned_size);
    }

    /// @brief Deallocate memory
    void deallocate(void* ptr, size_type /*size*/) {
        std::free(ptr);
    }

    /// @brief Get memory space properties
    [[nodiscard]] memory_space_properties properties() const {
        return memory_space_properties{
            .host_accessible = true,
            .device_accessible = false,
            .unified = false,
            .supports_atomics = true,
            .pageable = true,
            .alignment = alignof(std::max_align_t)
        };
    }

    /// @brief Get memory space name
    [[nodiscard]] static constexpr const char* name() noexcept {
        return "mock_host";
    }
};

/// @brief Mock memory space with typed allocation
class mock_typed_memory_space : public mock_memory_space {
public:
    /// @brief Allocate typed memory
    template <typename T>
    [[nodiscard]] T* allocate_typed(size_type count) {
        return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
    }

    /// @brief Deallocate typed memory
    template <typename T>
    void deallocate_typed(T* ptr, size_type count) {
        deallocate(ptr, count * sizeof(T));
    }

    /// @brief Construct object
    template <typename T>
    void construct(T* ptr) {
        new (ptr) T{};
    }

    /// @brief Destroy object
    template <typename T>
    void destroy(T* ptr) {
        ptr->~T();
    }
};

/// @brief Mock memory space with accessibility queries
class mock_accessible_memory_space : public mock_memory_space {
public:
    [[nodiscard]] bool is_host_accessible() const { return true; }
    [[nodiscard]] bool is_device_accessible() const { return false; }
    [[nodiscard]] bool is_accessible_from_host() const { return true; }
    [[nodiscard]] bool is_accessible_from_device() const { return false; }
};

/// @brief Type that doesn't satisfy MemorySpace concept
struct not_a_memory_space {
    using pointer = void*;
    using size_type = dtl::size_type;

    // Missing: allocate, deallocate, properties, name
};

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST(MemorySpaceConceptTest, MockMemorySpaceSatisfiesConcept) {
    static_assert(MemorySpace<mock_memory_space>);
}

TEST(MemorySpaceConceptTest, MockTypedSatisfiesConcept) {
    static_assert(MemorySpace<mock_typed_memory_space>);
    static_assert(TypedMemorySpace<mock_typed_memory_space, int>);
    static_assert(TypedMemorySpace<mock_typed_memory_space, double>);
}

TEST(MemorySpaceConceptTest, MockAccessibleSatisfiesConcept) {
    static_assert(MemorySpace<mock_accessible_memory_space>);
    static_assert(AccessibleMemorySpace<mock_accessible_memory_space>);
}

TEST(MemorySpaceConceptTest, NonMemorySpaceDoesNotSatisfy) {
    static_assert(!MemorySpace<not_a_memory_space>);
    static_assert(!MemorySpace<int>);
    static_assert(!MemorySpace<void>);
}

TEST(MemorySpaceConceptTest, BasicMemorySpaceDoesNotSatisfyAccessible) {
    static_assert(!AccessibleMemorySpace<mock_memory_space>);
}

// =============================================================================
// Memory Space Properties Tests
// =============================================================================

TEST(MemorySpacePropertiesTest, DefaultProperties) {
    memory_space_properties props;

    EXPECT_TRUE(props.host_accessible);
    EXPECT_FALSE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_TRUE(props.pageable);
    EXPECT_EQ(props.alignment, alignof(std::max_align_t));
}

TEST(MemorySpacePropertiesTest, MockSpaceProperties) {
    mock_memory_space space;
    auto props = space.properties();

    EXPECT_TRUE(props.host_accessible);
    EXPECT_FALSE(props.device_accessible);
    EXPECT_STREQ(space.name(), "mock_host");
}

// =============================================================================
// Memory Allocation Tests
// =============================================================================

TEST(MemorySpaceConceptTest, BasicAllocation) {
    mock_memory_space space;

    void* ptr = space.allocate(1024);
    ASSERT_NE(ptr, nullptr);

    space.deallocate(ptr, 1024);
}

TEST(MemorySpaceConceptTest, AlignedAllocation) {
    mock_memory_space space;

    constexpr size_type alignment = 64;
    void* ptr = space.allocate(1024, alignment);
    ASSERT_NE(ptr, nullptr);

    // Check alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0);

    space.deallocate(ptr, 1024);
}

TEST(MemorySpaceConceptTest, TypedAllocation) {
    mock_typed_memory_space space;

    int* ints = space.allocate_typed<int>(10);
    ASSERT_NE(ints, nullptr);

    // Use the memory
    for (size_type i = 0; i < 10; ++i) {
        ints[i] = static_cast<int>(i);
    }

    space.deallocate_typed(ints, 10);
}

TEST(MemorySpaceConceptTest, ConstructDestroy) {
    mock_typed_memory_space space;

    struct TestObj {
        int value = 42;
        ~TestObj() = default;
    };

    TestObj* obj = space.allocate_typed<TestObj>(1);
    ASSERT_NE(obj, nullptr);

    space.construct(obj);
    EXPECT_EQ(obj->value, 42);

    space.destroy(obj);
    space.deallocate_typed(obj, 1);
}

// =============================================================================
// Tag Type Tests
// =============================================================================

TEST(MemorySpaceTagTest, TagsAreDistinct) {
    static_assert(!std::is_same_v<host_memory_space_tag, device_memory_space_tag>);
    static_assert(!std::is_same_v<host_memory_space_tag, unified_memory_space_tag>);
    static_assert(!std::is_same_v<device_memory_space_tag, unified_memory_space_tag>);
    static_assert(!std::is_same_v<host_memory_space_tag, pinned_memory_space_tag>);
}

// =============================================================================
// Utility Function Tests
// =============================================================================

TEST(MemorySpaceUtilityTest, SpacesCompatible) {
    // All mock spaces should be compatible (stub implementation returns true)
    EXPECT_TRUE((spaces_compatible<mock_memory_space, mock_memory_space>()));
}

TEST(MemorySpaceUtilityTest, SpaceAlignment) {
    EXPECT_EQ((space_alignment<mock_memory_space, int>()), alignof(int));
    EXPECT_EQ((space_alignment<mock_memory_space, double>()), alignof(double));
}

}  // namespace dtl::test
