// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hip_event.hpp
/// @brief HIP event wrapper for synchronization and timing
/// @details Provides RAII wrapper for HIP events used for
///          stream synchronization and kernel timing.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/event.hpp>

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <memory>

namespace dtl {
namespace hip {

// ============================================================================
// Event Flags
// ============================================================================

/// @brief Flags for HIP event creation
enum class event_flags : unsigned int {
    /// @brief Default event
    default_event = 0,

    /// @brief Disable timing (faster synchronization)
    disable_timing = 1,

    /// @brief Event is valid for inter-process communication
    interprocess = 2,

    /// @brief Use blocking synchronization
    blocking_sync = 4
};

/// @brief Combine event flags
[[nodiscard]] constexpr event_flags operator|(event_flags a, event_flags b) noexcept {
    return static_cast<event_flags>(
        static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

/// @brief Check if flag is set
[[nodiscard]] constexpr bool has_flag(event_flags flags, event_flags flag) noexcept {
    return (static_cast<unsigned int>(flags) & static_cast<unsigned int>(flag)) != 0;
}

// ============================================================================
// HIP Event
// ============================================================================

/// @brief RAII wrapper for HIP events
/// @details Provides automatic resource management for HIP events,
///          used for stream synchronization and timing measurements.
///          Satisfies the Event concept.
class hip_event {
public:
    /// @brief Default constructor (creates event with timing)
    hip_event() : hip_event(event_flags::default_event) {}

    /// @brief Construct with flags
    /// @param flags Event creation flags
    explicit hip_event(event_flags flags) {
#if DTL_ENABLE_HIP
        unsigned int hip_flags = 0;
        if (has_flag(flags, event_flags::disable_timing)) {
            hip_flags |= hipEventDisableTiming;
        }
        if (has_flag(flags, event_flags::interprocess)) {
            hip_flags |= hipEventInterprocess;
        }
        if (has_flag(flags, event_flags::blocking_sync)) {
            hip_flags |= hipEventBlockingSync;
        }

        hipError_t err = hipEventCreateWithFlags(&event_, hip_flags);
        if (err != hipSuccess) {
            event_ = nullptr;
        }
        flags_ = flags;
#else
        (void)flags;
#endif
    }

    /// @brief Destructor
    ~hip_event() {
#if DTL_ENABLE_HIP
        if (event_ != nullptr) {
            hipEventDestroy(event_);
        }
#endif
    }

    // Non-copyable
    hip_event(const hip_event&) = delete;
    hip_event& operator=(const hip_event&) = delete;

    // Movable
    hip_event(hip_event&& other) noexcept
#if DTL_ENABLE_HIP
        : event_(other.event_)
        , flags_(other.flags_)
#endif
    {
#if DTL_ENABLE_HIP
        other.event_ = nullptr;
#endif
    }

    hip_event& operator=(hip_event&& other) noexcept {
        if (this != &other) {
#if DTL_ENABLE_HIP
            if (event_ != nullptr) {
                hipEventDestroy(event_);
            }
            event_ = other.event_;
            flags_ = other.flags_;
            other.event_ = nullptr;
#endif
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Event Interface
    // ------------------------------------------------------------------------

    /// @brief Check if event is valid
    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_HIP
        return event_ != nullptr;
#else
        return false;
#endif
    }

    /// @brief Record event on a stream
    /// @param stream HIP stream (nullptr for default stream)
    /// @return Success or error
    result<void> record(void* stream = nullptr) {
#if DTL_ENABLE_HIP
        if (event_ == nullptr) {
            return make_error<void>(status_code::invalid_state,
                                   "Event not initialized");
        }

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        hipError_t err = hipEventRecord(event_, hip_stream);
        if (err != hipSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "hipEventRecord failed");
        }
        recorded_ = true;
        return {};
#else
        (void)stream;
        return make_error<void>(status_code::not_supported,
                               "HIP support not enabled");
#endif
    }

    /// @brief Wait for event to complete (blocking, concept-compliant)
    void wait() {
#if DTL_ENABLE_HIP
        if (event_ != nullptr) {
            hipEventSynchronize(event_);
        }
#endif
    }

    /// @brief Wait for event to complete (blocking)
    /// @return Success or error
    void synchronize() {
#if DTL_ENABLE_HIP
        if (event_ != nullptr) {
            hipEventSynchronize(event_);
        }
#endif
    }

    /// @brief Query if event has completed (non-blocking)
    /// @return event_status::complete if completed, event_status::pending if still running
    [[nodiscard]] event_status query() const noexcept {
#if DTL_ENABLE_HIP
        if (event_ == nullptr) return event_status::complete;

        hipError_t err = hipEventQuery(event_);
        if (err == hipSuccess) return event_status::complete;
        if (err == hipErrorNotReady) return event_status::pending;
        return event_status::error;
#else
        return event_status::complete;
#endif
    }

    /// @brief Check if event is complete (alias for query)
    [[nodiscard]] bool is_complete() const noexcept { return query() == event_status::complete; }

    // ------------------------------------------------------------------------
    // Timing Methods
    // ------------------------------------------------------------------------

    /// @brief Calculate elapsed time between two events
    /// @param start Start event
    /// @param end End event
    /// @return Elapsed time in milliseconds or error
    [[nodiscard]] static result<float> elapsed_time(const hip_event& start,
                                                     const hip_event& end) {
#if DTL_ENABLE_HIP
        if (!start.valid() || !end.valid()) {
            return make_error<float>(status_code::invalid_argument,
                                    "Invalid event");
        }

        if (has_flag(start.flags_, event_flags::disable_timing) ||
            has_flag(end.flags_, event_flags::disable_timing)) {
            return make_error<float>(status_code::invalid_argument,
                                    "Timing disabled for event");
        }

        float ms;
        hipError_t err = hipEventElapsedTime(&ms, start.event_, end.event_);
        if (err != hipSuccess) {
            return make_error<float>(status_code::backend_error,
                                    "hipEventElapsedTime failed");
        }
        return ms;
#else
        (void)start; (void)end;
        return make_error<float>(status_code::not_supported,
                                "HIP support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Native Handle Access
    // ------------------------------------------------------------------------

#if DTL_ENABLE_HIP
    /// @brief Get the underlying HIP event
    [[nodiscard]] hipEvent_t native_handle() const noexcept { return event_; }
#endif

    /// @brief Get event flags
    [[nodiscard]] event_flags flags() const noexcept { return flags_; }

    /// @brief Check if event has been recorded
    [[nodiscard]] bool recorded() const noexcept { return recorded_; }

private:
#if DTL_ENABLE_HIP
    hipEvent_t event_ = nullptr;
#endif
    event_flags flags_ = event_flags::default_event;
    bool recorded_ = false;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a HIP event optimized for synchronization (no timing)
/// @return Event without timing capability
[[nodiscard]] inline hip_event make_sync_event() {
    return hip_event(event_flags::disable_timing);
}

/// @brief Create a HIP event for timing measurements
/// @return Event with timing capability
[[nodiscard]] inline hip_event make_timing_event() {
    return hip_event(event_flags::default_event);
}

/// @brief Create a blocking HIP event
/// @return Event with blocking synchronization
[[nodiscard]] inline hip_event make_blocking_event() {
    return hip_event(event_flags::blocking_sync | event_flags::disable_timing);
}

// ============================================================================
// Scoped Timer
// ============================================================================

/// @brief RAII scoped timer using HIP events
class hip_scoped_timer {
public:
    /// @brief Start timing
    explicit hip_scoped_timer(void* stream = nullptr)
        : stream_(stream) {
        start_.record(stream);
    }

    /// @brief Stop timing and get elapsed time
    ~hip_scoped_timer() = default;

    /// @brief Get elapsed time so far
    /// @return Elapsed time in milliseconds or error
    [[nodiscard]] result<float> elapsed() {
        if (!end_.recorded()) {
            end_.record(stream_);
            end_.synchronize();
        }
        return hip_event::elapsed_time(start_, end_);
    }

    /// @brief Stop the timer (records end event)
    result<void> stop() {
        if (!end_.recorded()) {
            return end_.record(stream_);
        }
        return {};
    }

private:
    hip_event start_;
    hip_event end_;
    void* stream_;
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Event<hip_event>, "hip_event must satisfy Event concept");

}  // namespace hip
}  // namespace dtl
