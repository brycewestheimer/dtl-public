// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_loudness_tags.cpp
/// @brief Tests for loudness tags and STL parity contract

#include <gtest/gtest.h>

#include <dtl/core/loudness_tags.hpp>

namespace dtl::test {

// ============================================================================
// Tag Type Tests
// ============================================================================

TEST(LoudnessTags, CollectiveTag) {
    // Verify collective_t is default constructible
    [[maybe_unused]] collective_t tag{};

    // Verify global instance exists
    [[maybe_unused]] auto& instance = collective;
}

TEST(LoudnessTags, CommunicatingTag) {
    [[maybe_unused]] communicating_t tag{};
    [[maybe_unused]] auto& instance = communicating;
}

TEST(LoudnessTags, FailableTag) {
    [[maybe_unused]] failable_t tag{};
    [[maybe_unused]] auto& instance = failable;
}

TEST(LoudnessTags, BlockingTag) {
    [[maybe_unused]] blocking_op_t tag{};
    [[maybe_unused]] auto& instance = blocking_op;
}

TEST(LoudnessTags, AllocatingTag) {
    [[maybe_unused]] allocating_t tag{};
    [[maybe_unused]] auto& instance = allocating;
}

TEST(LoudnessTags, RemoteTag) {
    [[maybe_unused]] remote_op_t tag{};
    [[maybe_unused]] auto& instance = remote_op;
}

TEST(LoudnessTags, InvalidatingTag) {
    [[maybe_unused]] invalidating_t tag{};
    [[maybe_unused]] auto& instance = invalidating;
}

// ============================================================================
// Combined Tag Tests
// ============================================================================

TEST(LoudnessTags, CollectiveBlockingCombined) {
    collective_blocking_t tag{};

    // Should be both collective and blocking
    static_assert(std::is_base_of_v<collective_t, collective_blocking_t>);
    static_assert(std::is_base_of_v<blocking_op_t, collective_blocking_t>);

    [[maybe_unused]] auto& instance = collective_blocking;
}

TEST(LoudnessTags, CollectiveFailableCombined) {
    collective_failable_t tag{};

    static_assert(std::is_base_of_v<collective_t, collective_failable_t>);
    static_assert(std::is_base_of_v<failable_t, collective_failable_t>);

    [[maybe_unused]] auto& instance = collective_failable;
}

TEST(LoudnessTags, CommunicatingFailableCombined) {
    communicating_failable_t tag{};

    static_assert(std::is_base_of_v<communicating_t, communicating_failable_t>);
    static_assert(std::is_base_of_v<failable_t, communicating_failable_t>);

    [[maybe_unused]] auto& instance = communicating_failable;
}

// ============================================================================
// Tag Trait Tests
// ============================================================================

TEST(TagTraits, IsCollective) {
    static_assert(is_collective_v<collective_t>);
    static_assert(is_collective_v<collective_blocking_t>);
    static_assert(is_collective_v<collective_failable_t>);

    static_assert(!is_collective_v<communicating_t>);
    static_assert(!is_collective_v<failable_t>);
    static_assert(!is_collective_v<blocking_op_t>);
}

TEST(TagTraits, IsCommunicating) {
    // Collective implies communicating
    static_assert(is_communicating_v<collective_t>);
    static_assert(is_communicating_v<collective_blocking_t>);

    // Explicitly communicating
    static_assert(is_communicating_v<communicating_t>);
    static_assert(is_communicating_v<communicating_failable_t>);

    // Not communicating
    static_assert(!is_communicating_v<failable_t>);
    static_assert(!is_communicating_v<blocking_op_t>);
    static_assert(!is_communicating_v<allocating_t>);
}

TEST(TagTraits, IsFailable) {
    static_assert(is_failable_v<failable_t>);
    static_assert(is_failable_v<collective_failable_t>);
    static_assert(is_failable_v<communicating_failable_t>);

    static_assert(!is_failable_v<collective_t>);
    static_assert(!is_failable_v<communicating_t>);
    static_assert(!is_failable_v<blocking_op_t>);
}

TEST(TagTraits, IsBlocking) {
    static_assert(is_blocking_v<blocking_op_t>);
    static_assert(is_blocking_v<collective_blocking_t>);

    static_assert(!is_blocking_v<collective_t>);
    static_assert(!is_blocking_v<communicating_t>);
    static_assert(!is_blocking_v<failable_t>);
}

TEST(TagTraits, IsAllocating) {
    static_assert(is_allocating_v<allocating_t>);

    static_assert(!is_allocating_v<collective_t>);
    static_assert(!is_allocating_v<failable_t>);
}

TEST(TagTraits, IsRemote) {
    static_assert(is_remote_v<remote_op_t>);

    static_assert(!is_remote_v<collective_t>);
    static_assert(!is_remote_v<communicating_t>);
}

TEST(TagTraits, IsInvalidating) {
    static_assert(is_invalidating_v<invalidating_t>);

    static_assert(!is_invalidating_v<collective_t>);
    static_assert(!is_invalidating_v<failable_t>);
}

// ============================================================================
// Loudness Level Tests
// ============================================================================

TEST(LoudnessLevel, OperationLoudnessCollective) {
    constexpr auto level = operation_loudness<collective_t>();
    EXPECT_EQ(level, loudness_level::l2_statically_detectable);
}

TEST(LoudnessLevel, OperationLoudnessRemote) {
    constexpr auto level = operation_loudness<remote_op_t>();
    EXPECT_EQ(level, loudness_level::l1_syntactically_loud);
}

TEST(LoudnessLevel, OperationLoudnessCommunicating) {
    // Communicating but not collective gets L2
    constexpr auto level = operation_loudness<communicating_t>();
    EXPECT_EQ(level, loudness_level::l2_statically_detectable);
}

TEST(LoudnessLevel, OperationLoudnessFailable) {
    // Failable alone is L0 (doesn't affect loudness)
    constexpr auto level = operation_loudness<failable_t>();
    EXPECT_EQ(level, loudness_level::l0_stl_compatible);
}

TEST(LoudnessLevel, OperationLoudnessBlocking) {
    constexpr auto level = operation_loudness<blocking_op_t>();
    EXPECT_EQ(level, loudness_level::l0_stl_compatible);
}

// ============================================================================
// Compile-Time Assertion Tests (Commented - would fail compilation)
// ============================================================================

// These tests verify the static_assert mechanism works.
// Uncomment to verify they cause compilation failures.

// TEST(LoudnessAssertions, FailsForCollective) {
//     // This would fail compilation:
//     // assert_stl_compatible<collective_t>();
// }

// TEST(LoudnessAssertions, FailsWithoutCollectiveTag) {
//     // This would fail compilation:
//     // assert_collective_acknowledged<failable_t>();
// }

// ============================================================================
// Usage Pattern Tests
// ============================================================================

// Example function using loudness tags
template <typename Tag>
void example_collective_op(Tag) {
    assert_collective_acknowledged<Tag>();
    // Operation implementation...
}

TEST(UsagePatterns, CollectiveTagRequired) {
    // This compiles:
    example_collective_op(collective);
    example_collective_op(collective_blocking);
    example_collective_op(collective_failable);

    // These would fail compilation:
    // example_collective_op(communicating);
    // example_collective_op(failable);
}

// Example function demonstrating tag-based dispatch
template <typename Tag>
std::string describe_operation(Tag) {
    if constexpr (is_collective_v<Tag>) {
        return "collective";
    } else if constexpr (is_communicating_v<Tag>) {
        return "communicating";
    } else if constexpr (is_failable_v<Tag>) {
        return "failable";
    } else {
        return "local";
    }
}

TEST(UsagePatterns, TagBasedDispatch) {
    EXPECT_EQ(describe_operation(collective), "collective");
    EXPECT_EQ(describe_operation(collective_blocking), "collective");
    EXPECT_EQ(describe_operation(communicating), "communicating");
    EXPECT_EQ(describe_operation(failable), "failable");
    EXPECT_EQ(describe_operation(blocking_op), "local");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(EdgeCases, EmptyTag) {
    struct empty_tag {};

    static_assert(!is_collective_v<empty_tag>);
    static_assert(!is_communicating_v<empty_tag>);
    static_assert(!is_failable_v<empty_tag>);
    static_assert(!is_blocking_v<empty_tag>);
    static_assert(!is_allocating_v<empty_tag>);
    static_assert(!is_remote_v<empty_tag>);
    static_assert(!is_invalidating_v<empty_tag>);
}

TEST(EdgeCases, MultipleInheritance) {
    // Custom tag inheriting from multiple base tags
    struct custom_tag : collective_t, failable_t, invalidating_t {
        explicit constexpr custom_tag() = default;
    };

    static_assert(is_collective_v<custom_tag>);
    static_assert(is_failable_v<custom_tag>);
    static_assert(is_invalidating_v<custom_tag>);
    static_assert(is_communicating_v<custom_tag>);  // Via collective
}

}  // namespace dtl::test
