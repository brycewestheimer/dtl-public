// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file event.hpp
/// @brief Event concept for synchronization primitives
/// @details Defines requirements for asynchronous event handling.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <concepts>
#include <chrono>

namespace dtl {

// ============================================================================
// Event Status
// ============================================================================

/// @brief Status of an asynchronous event
enum class event_status {
    pending,    ///< Event has not completed
    complete,   ///< Event has completed successfully
    error       ///< Event completed with error
};

// ============================================================================
// Event Concept
// ============================================================================

/// @brief Core event concept for synchronization
/// @details Defines minimum requirements for an event type.
///
/// @par Required Operations:
/// - wait(): Block until event completes
/// - query(): Check if event has completed
/// - synchronize(): Ensure event is complete (alias for wait)
template <typename T>
concept Event = requires(T& event, const T& cevent) {
    // Block until complete
    { event.wait() } -> std::same_as<void>;

    // Non-blocking status check
    { cevent.query() } -> std::same_as<event_status>;

    // Synchronize (ensure completion)
    { event.synchronize() } -> std::same_as<void>;
};

// ============================================================================
// Timed Event Concept
// ============================================================================

/// @brief Event with timing capabilities
/// @details Supports timeout-based waiting and elapsed time queries.
template <typename T>
concept TimedEvent = Event<T> &&
    requires(T& event, const T& cevent,
             std::chrono::milliseconds timeout) {
    // Wait with timeout (returns true if completed, false if timeout)
    { event.wait_for(timeout) } -> std::same_as<bool>;

    // Get elapsed time since event was recorded
    { cevent.elapsed() } -> std::convertible_to<std::chrono::nanoseconds>;
};

// ============================================================================
// Recordable Event Concept
// ============================================================================

/// @brief Event that can be recorded on a stream/queue
template <typename T>
concept RecordableEvent = Event<T> &&
    requires(T& event) {
    // Record event on current stream/queue
    { event.record() } -> std::same_as<void>;

    // Check if event was recorded
    { event.is_recorded() } -> std::same_as<bool>;
};

// ============================================================================
// Event Properties
// ============================================================================

/// @brief Properties describing an event
struct event_properties {
    /// @brief Whether event supports timing queries
    bool supports_timing = false;

    /// @brief Whether event can be recorded multiple times
    bool reusable = true;

    /// @brief Whether event requires explicit recording
    bool requires_recording = false;
};

// ============================================================================
// Event Traits
// ============================================================================

/// @brief Traits for event types
template <typename E>
struct event_traits {
    /// @brief Whether event supports timing
    static constexpr bool supports_timing = false;

    /// @brief Whether event is reusable
    static constexpr bool is_reusable = true;

    /// @brief Whether event type is GPU-based
    static constexpr bool is_gpu_event = false;
};

// ============================================================================
// Event Tag Types
// ============================================================================

/// @brief Tag for host-side events
struct host_event_tag {};

/// @brief Tag for CUDA events
struct cuda_event_tag {};

/// @brief Tag for HIP events
struct hip_event_tag {};

/// @brief Tag for SYCL events
struct sycl_event_tag {};

// ============================================================================
// Null Event
// ============================================================================

/// @brief No-op event for synchronous operations
/// @details Always reports complete, wait() returns immediately.
class null_event {
public:
    using tag_type = host_event_tag;

    /// @brief Wait (no-op, returns immediately)
    void wait() noexcept {}

    /// @brief Query status (always complete)
    [[nodiscard]] static constexpr event_status query() noexcept {
        return event_status::complete;
    }

    /// @brief Synchronize (no-op)
    void synchronize() noexcept {}

    /// @brief Wait with timeout (always returns true immediately)
    [[nodiscard]] static bool wait_for(std::chrono::milliseconds /*timeout*/) noexcept {
        return true;
    }

    /// @brief Elapsed time (always zero)
    [[nodiscard]] static constexpr std::chrono::nanoseconds elapsed() noexcept {
        return std::chrono::nanoseconds{0};
    }
};

/// @brief Traits specialization for null_event
template <>
struct event_traits<null_event> {
    static constexpr bool supports_timing = true;
    static constexpr bool is_reusable = true;
    static constexpr bool is_gpu_event = false;
};

// ============================================================================
// Event Utilities
// ============================================================================

/// @brief Wait for multiple events
/// @tparam Events Event types
/// @param events The events to wait for
template <Event... Events>
void wait_all(Events&... events) {
    (events.wait(), ...);
}

/// @brief Check if all events are complete
/// @tparam Events Event types
/// @param events The events to check
/// @return true if all events are complete
template <Event... Events>
[[nodiscard]] bool all_complete(const Events&... events) {
    return ((events.query() == event_status::complete) && ...);
}

/// @brief Check if any event is complete
/// @tparam Events Event types
/// @param events The events to check
/// @return true if any event is complete
template <Event... Events>
[[nodiscard]] bool any_complete(const Events&... events) {
    return ((events.query() == event_status::complete) || ...);
}

}  // namespace dtl
