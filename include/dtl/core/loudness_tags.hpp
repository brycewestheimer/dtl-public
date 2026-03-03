// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file loudness_tags.hpp
/// @brief STL Parity Contract loudness tags
/// @details Provides compile-time tags to mark operations that deviate from
///          STL semantics, enabling "loudness" requirements for distributed ops.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#include <type_traits>

namespace dtl {

// ============================================================================
// Loudness Tag Types
// ============================================================================

/// @brief Tag indicating a collective operation (all ranks must participate)
/// @details Operations marked with this tag require synchronization across
///          all ranks in the communicator. Calling from only a subset of
///          ranks will deadlock.
///
/// @par Usage:
/// @code
/// // Function signature showing collective nature
/// void allreduce(collective_t, T& value);
///
/// // Called as:
/// allreduce(collective, my_value);
/// @endcode
struct collective_t {
    explicit constexpr collective_t() = default;
};

/// @brief Global instance for collective tag
inline constexpr collective_t collective{};

/// @brief Tag indicating operation may communicate
/// @details Operations marked with this tag may involve network communication
///          but do not require all ranks to participate (point-to-point).
///
/// @par Usage:
/// @code
/// void send(communicating_t, rank_t dest, const T& data);
/// send(communicating, 1, my_data);
/// @endcode
struct communicating_t {
    explicit constexpr communicating_t() = default;
};

/// @brief Global instance for communicating tag
inline constexpr communicating_t communicating{};

/// @brief Tag indicating operation may fail
/// @details Operations marked with this tag may return an error result
///          or throw exceptions. The caller should handle potential failures.
///
/// @par Usage:
/// @code
/// result<T> try_get(failable_t, index_t idx);
/// auto result = try_get(failable, 0);
/// @endcode
struct failable_t {
    explicit constexpr failable_t() = default;
};

/// @brief Global instance for failable tag
inline constexpr failable_t failable{};

/// @brief Tag indicating operation is blocking
/// @details Operations marked with this tag will block until complete.
///          The caller's thread will wait for the operation to finish.
struct blocking_op_t {
    explicit constexpr blocking_op_t() = default;
};

/// @brief Global instance for blocking tag
inline constexpr blocking_op_t blocking_op{};

/// @brief Tag indicating operation allocates memory
/// @details Operations marked with this tag may allocate memory on the
///          heap, host, or device. Allocation may fail.
struct allocating_t {
    explicit constexpr allocating_t() = default;
};

/// @brief Global instance for allocating tag
inline constexpr allocating_t allocating{};

/// @brief Tag indicating operation is remote (affects non-local data)
/// @details Operations marked with this tag operate on data owned by
///          other ranks. Requires explicit handling of communication.
struct remote_op_t {
    explicit constexpr remote_op_t() = default;
};

/// @brief Global instance for remote_op tag
inline constexpr remote_op_t remote_op{};

/// @brief Tag indicating operation invalidates iterators/views
/// @details Operations marked with this tag may invalidate existing
///          iterators, views, or references to container data.
struct invalidating_t {
    explicit constexpr invalidating_t() = default;
};

/// @brief Global instance for invalidating tag
inline constexpr invalidating_t invalidating{};

// ============================================================================
// Tag Combinations
// ============================================================================

/// @brief Combined tag for collective + blocking operations
struct collective_blocking_t : collective_t, blocking_op_t {
    explicit constexpr collective_blocking_t() = default;
};

/// @brief Global instance for collective_blocking tag
inline constexpr collective_blocking_t collective_blocking{};

/// @brief Combined tag for collective + failable operations
struct collective_failable_t : collective_t, failable_t {
    explicit constexpr collective_failable_t() = default;
};

/// @brief Global instance for collective_failable tag
inline constexpr collective_failable_t collective_failable{};

/// @brief Combined tag for communicating + failable operations
struct communicating_failable_t : communicating_t, failable_t {
    explicit constexpr communicating_failable_t() = default;
};

/// @brief Global instance for communicating_failable tag
inline constexpr communicating_failable_t communicating_failable{};

// ============================================================================
// Tag Traits
// ============================================================================

/// @brief Check if a tag indicates collective operation
template <typename T>
struct is_collective : std::is_base_of<collective_t, T> {};

template <typename T>
inline constexpr bool is_collective_v = is_collective<T>::value;

/// @brief Check if a tag indicates communicating operation
template <typename T>
struct is_communicating : std::bool_constant<
    std::is_base_of_v<communicating_t, T> ||
    std::is_base_of_v<collective_t, T>  // Collective implies communicating
> {};

template <typename T>
inline constexpr bool is_communicating_v = is_communicating<T>::value;

/// @brief Check if a tag indicates failable operation
template <typename T>
struct is_failable : std::is_base_of<failable_t, T> {};

template <typename T>
inline constexpr bool is_failable_v = is_failable<T>::value;

/// @brief Check if a tag indicates blocking operation
template <typename T>
struct is_blocking : std::is_base_of<blocking_op_t, T> {};

template <typename T>
inline constexpr bool is_blocking_v = is_blocking<T>::value;

/// @brief Check if a tag indicates allocating operation
template <typename T>
struct is_allocating : std::is_base_of<allocating_t, T> {};

template <typename T>
inline constexpr bool is_allocating_v = is_allocating<T>::value;

/// @brief Check if a tag indicates remote operation
template <typename T>
struct is_remote : std::is_base_of<remote_op_t, T> {};

template <typename T>
inline constexpr bool is_remote_v = is_remote<T>::value;

/// @brief Check if a tag indicates invalidating operation
template <typename T>
struct is_invalidating : std::is_base_of<invalidating_t, T> {};

template <typename T>
inline constexpr bool is_invalidating_v = is_invalidating<T>::value;

// ============================================================================
// Loudness Level Classification
// ============================================================================

/// @brief STL Parity levels
/// @details
/// - L0: STL-compatible (local_view provides full STL semantics)
/// - L1: Syntactically loud (remote_ref has no implicit conversions)
/// - L2: Statically detectable (type-based distinction)
/// - L3: Explicitly documented (behavioral differences)
enum class loudness_level : uint8_t {
    /// @brief L0: Full STL compatibility via local_view
    l0_stl_compatible = 0,

    /// @brief L1: Syntactically loud (explicit operations required)
    l1_syntactically_loud = 1,

    /// @brief L2: Statically detectable deviations
    l2_statically_detectable = 2,

    /// @brief L3: Documented behavioral differences only
    l3_documented = 3
};

/// @brief Get loudness level for an operation type
/// @tparam Tag Operation tag type
template <typename Tag>
constexpr loudness_level operation_loudness() {
    if constexpr (is_collective_v<Tag>) {
        return loudness_level::l2_statically_detectable;
    } else if constexpr (is_remote_v<Tag>) {
        return loudness_level::l1_syntactically_loud;
    } else if constexpr (is_communicating_v<Tag>) {
        return loudness_level::l2_statically_detectable;
    } else {
        return loudness_level::l0_stl_compatible;
    }
}

// ============================================================================
// Loudness Assertions (Compile-Time Checks)
// ============================================================================

/// @brief Assert that an operation is at most L0 (STL-compatible)
/// @details Fails compilation if operation requires loudness above L0.
template <typename Tag>
constexpr void assert_stl_compatible() {
    static_assert(operation_loudness<Tag>() == loudness_level::l0_stl_compatible,
                  "Operation is not STL-compatible (requires loud invocation)");
}

/// @brief Assert that caller acknowledges collective nature
/// @details Use at function entry to ensure caller passed collective tag.
template <typename Tag>
constexpr void assert_collective_acknowledged() {
    static_assert(is_collective_v<Tag>,
                  "Collective operation requires collective_t tag");
}

/// @brief Assert that caller acknowledges potential communication
template <typename Tag>
constexpr void assert_communication_acknowledged() {
    static_assert(is_communicating_v<Tag>,
                  "Operation may communicate; requires communicating_t or collective_t tag");
}

// ============================================================================
// Documentation Helpers
// ============================================================================

/// @brief Marker for operations with L0 (STL-compatible) semantics
#define DTL_L0_OPERATION /* STL-compatible */

/// @brief Marker for operations with L1 (syntactically loud) semantics
#define DTL_L1_OPERATION /* Syntactically loud */

/// @brief Marker for operations with L2 (statically detectable) semantics
#define DTL_L2_OPERATION /* Statically detectable deviation */

/// @brief Marker for operations with L3 (documented) semantics
#define DTL_L3_OPERATION /* Documented behavioral difference */

}  // namespace dtl
