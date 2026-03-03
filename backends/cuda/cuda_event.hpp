// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_event.hpp
/// @brief CUDA event wrapper for synchronization and timing
/// @details Provides RAII wrapper for CUDA events used for
///          stream synchronization and kernel timing.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/event.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>

namespace dtl {
namespace cuda {

// ============================================================================
// Event Flags
// ============================================================================

/// @brief Flags for CUDA event creation
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
// CUDA Event
// ============================================================================

/// @brief RAII wrapper for CUDA events
/// @details Provides automatic resource management for CUDA events,
///          used for stream synchronization and timing measurements.
///          Satisfies the Event concept.
class cuda_event {
public:
    /// @brief Default constructor (creates event with timing)
    cuda_event() : cuda_event(event_flags::default_event) {}

    /// @brief Construct with flags
    /// @param flags Event creation flags
    explicit cuda_event(event_flags flags) {
#if DTL_ENABLE_CUDA
        unsigned int cuda_flags = 0;
        if (has_flag(flags, event_flags::disable_timing)) {
            cuda_flags |= cudaEventDisableTiming;
        }
        if (has_flag(flags, event_flags::interprocess)) {
            cuda_flags |= cudaEventInterprocess;
        }
        if (has_flag(flags, event_flags::blocking_sync)) {
            cuda_flags |= cudaEventBlockingSync;
        }

        cudaError_t err = cudaEventCreateWithFlags(&event_, cuda_flags);
        if (err != cudaSuccess) {
            event_ = nullptr;
        }
        flags_ = flags;
#else
        (void)flags;
#endif
    }

    /// @brief Destructor
    ~cuda_event() {
#if DTL_ENABLE_CUDA
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
#endif
    }

    // Non-copyable
    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;

    // Movable
    cuda_event(cuda_event&& other) noexcept
#if DTL_ENABLE_CUDA
        : event_(other.event_)
        , flags_(other.flags_)
#endif
    {
#if DTL_ENABLE_CUDA
        other.event_ = nullptr;
#endif
    }

    cuda_event& operator=(cuda_event&& other) noexcept {
        if (this != &other) {
#if DTL_ENABLE_CUDA
            if (event_ != nullptr) {
                cudaEventDestroy(event_);
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
#if DTL_ENABLE_CUDA
        return event_ != nullptr;
#else
        return false;
#endif
    }

    /// @brief Record event on a stream
    /// @param stream CUDA stream (nullptr for default stream)
    /// @return Success or error
    result<void> record(void* stream = nullptr) {
#if DTL_ENABLE_CUDA
        if (event_ == nullptr) {
            return make_error<void>(status_code::invalid_state,
                                   "Event not initialized");
        }

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t err = cudaEventRecord(event_, cuda_stream);
        if (err != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaEventRecord failed");
        }
        recorded_ = true;
        return {};
#else
        (void)stream;
        return make_error<void>(status_code::not_supported,
                               "CUDA support not enabled");
#endif
    }

    /// @brief Wait for event to complete (blocking, concept-compliant)
    void wait() {
#if DTL_ENABLE_CUDA
        if (event_ != nullptr) {
            cudaEventSynchronize(event_);
        }
#endif
    }

    /// @brief Wait for event to complete (blocking)
    /// @return Success or error
    void synchronize() {
#if DTL_ENABLE_CUDA
        if (event_ != nullptr) {
            cudaEventSynchronize(event_);
        }
#endif
    }

    /// @brief Query if event has completed (non-blocking)
    /// @return event_status::complete if completed, event_status::pending if still running
    [[nodiscard]] event_status query() const noexcept {
#if DTL_ENABLE_CUDA
        if (event_ == nullptr) return event_status::complete;

        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) return event_status::complete;
        if (err == cudaErrorNotReady) return event_status::pending;
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
    [[nodiscard]] static result<float> elapsed_time(const cuda_event& start,
                                                     const cuda_event& end) {
#if DTL_ENABLE_CUDA
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
        cudaError_t err = cudaEventElapsedTime(&ms, start.event_, end.event_);
        if (err != cudaSuccess) {
            return make_error<float>(status_code::backend_error,
                                    "cudaEventElapsedTime failed");
        }
        return ms;
#else
        (void)start; (void)end;
        return make_error<float>(status_code::not_supported,
                                "CUDA support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Native Handle Access
    // ------------------------------------------------------------------------

#if DTL_ENABLE_CUDA
    /// @brief Get the underlying CUDA event
    [[nodiscard]] cudaEvent_t native_handle() const noexcept { return event_; }
#endif

    /// @brief Get event flags
    [[nodiscard]] event_flags flags() const noexcept { return flags_; }

    /// @brief Check if event has been recorded
    [[nodiscard]] bool recorded() const noexcept { return recorded_; }

private:
#if DTL_ENABLE_CUDA
    cudaEvent_t event_ = nullptr;
#endif
    event_flags flags_ = event_flags::default_event;
    bool recorded_ = false;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a CUDA event optimized for synchronization (no timing)
/// @return Event without timing capability
[[nodiscard]] inline cuda_event make_sync_event() {
    return cuda_event(event_flags::disable_timing);
}

/// @brief Create a CUDA event for timing measurements
/// @return Event with timing capability
[[nodiscard]] inline cuda_event make_timing_event() {
    return cuda_event(event_flags::default_event);
}

/// @brief Create a blocking CUDA event
/// @return Event with blocking synchronization
[[nodiscard]] inline cuda_event make_blocking_event() {
    return cuda_event(event_flags::blocking_sync | event_flags::disable_timing);
}

// ============================================================================
// Scoped Timer
// ============================================================================

/// @brief RAII scoped timer using CUDA events
class cuda_scoped_timer {
public:
    /// @brief Start timing
    explicit cuda_scoped_timer(void* stream = nullptr)
        : stream_(stream) {
        start_.record(stream);
    }

    /// @brief Stop timing and get elapsed time
    ~cuda_scoped_timer() = default;

    /// @brief Get elapsed time so far
    /// @return Elapsed time in milliseconds or error
    [[nodiscard]] result<float> elapsed() {
        if (!end_.recorded()) {
            end_.record(stream_);
            end_.synchronize();
        }
        return cuda_event::elapsed_time(start_, end_);
    }

    /// @brief Stop the timer (records end event)
    result<void> stop() {
        if (!end_.recorded()) {
            return end_.record(stream_);
        }
        return {};
    }

private:
    cuda_event start_;
    cuda_event end_;
    void* stream_;
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Event<cuda_event>, "cuda_event must satisfy Event concept");

}  // namespace cuda
}  // namespace dtl
