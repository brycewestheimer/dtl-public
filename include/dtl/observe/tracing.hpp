// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file tracing.hpp
/// @brief Tracing types for DTL observability
/// @details Provides RAII trace span and scoped trace types for
///          instrumenting DTL operation durations and call hierarchies.
///          All types are no-op stubs when DTL_ENABLE_OBSERVABILITY
///          is not defined, guaranteeing zero overhead in production builds.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

namespace dtl::observe {

// =============================================================================
// Observability Feature Gate
// =============================================================================

#ifndef DTL_ENABLE_OBSERVABILITY
    #define DTL_ENABLE_OBSERVABILITY 0
#endif

// =============================================================================
// Span Status
// =============================================================================

/// @brief Status of a completed trace span
/// @since 0.1.0
enum class span_status : std::uint8_t {
    ok = 0,      ///< Operation completed successfully
    error = 1,   ///< Operation completed with an error
    unset = 2    ///< Status not explicitly set
};

// =============================================================================
// Trace Span
// =============================================================================

/// @brief RAII trace span for measuring operation duration
/// @details Records the start time on construction and the end time on
///          destruction. When DTL_ENABLE_OBSERVABILITY is disabled, the
///          span is an empty struct with no state and no runtime cost.
///
/// Usage:
/// @code
/// void my_allreduce(auto& comm) {
///     dtl::observe::trace_span span("dtl.comm.allreduce");
///     // ... perform allreduce ...
///     // span records duration automatically on destruction
/// }
/// @endcode
///
/// @since 0.1.0
class trace_span {
public:
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;
    using duration = clock::duration;

    /// @brief Construct and start a trace span
    /// @param name The operation name (e.g., "dtl.comm.allreduce")
    /// @param parent_id Optional parent span ID for hierarchical tracing
    explicit trace_span([[maybe_unused]] std::string_view name,
                        [[maybe_unused]] std::uint64_t parent_id = 0) noexcept
#if DTL_ENABLE_OBSERVABILITY
        : name_{name}, parent_id_{parent_id}, status_{span_status::unset},
          start_{clock::now()}, ended_{false}
#endif
    {}

    /// @brief Destructor -- ends the span if not already ended
    ~trace_span() {
#if DTL_ENABLE_OBSERVABILITY
        end();
#endif
    }

    // Non-copyable, non-movable (RAII semantics)
    trace_span(const trace_span&) = delete;
    trace_span& operator=(const trace_span&) = delete;
    trace_span(trace_span&&) = delete;
    trace_span& operator=(trace_span&&) = delete;

    /// @brief Explicitly end the span
    /// @details Records the end time. Subsequent calls are no-ops.
    void end() noexcept {
#if DTL_ENABLE_OBSERVABILITY
        if (!ended_) {
            end_ = clock::now();
            ended_ = true;
        }
#endif
    }

    /// @brief Set the span status
    /// @param s The status to set
    void set_status([[maybe_unused]] span_status s) noexcept {
#if DTL_ENABLE_OBSERVABILITY
        status_ = s;
#endif
    }

    /// @brief Get the span duration
    /// @return The elapsed duration, or zero when observability is disabled
    [[nodiscard]] duration elapsed() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        if (ended_) {
            return end_ - start_;
        }
        return clock::now() - start_;
#else
        return duration::zero();
#endif
    }

    /// @brief Get the span name
    /// @return The name, or empty string when observability is disabled
    [[nodiscard]] std::string_view name() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return name_;
#else
        return {};
#endif
    }

    /// @brief Get the span status
    /// @return The status, or unset when observability is disabled
    [[nodiscard]] span_status status() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return status_;
#else
        return span_status::unset;
#endif
    }

    /// @brief Check if the span has ended
    /// @return True if ended, false otherwise (always false when disabled)
    [[nodiscard]] bool ended() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return ended_;
#else
        return false;
#endif
    }

private:
#if DTL_ENABLE_OBSERVABILITY
    std::string name_;
    std::uint64_t parent_id_;
    span_status status_;
    time_point start_;
    time_point end_;
    bool ended_;
#endif
};

// =============================================================================
// Scoped Trace
// =============================================================================

/// @brief Convenience wrapper that creates a trace span with automatic naming
/// @details Combines a trace_span with a predefined operation name for
///          common instrumentation patterns. Identical to trace_span in
///          behavior but provides a more expressive API name.
///
/// Usage:
/// @code
/// {
///     dtl::observe::scoped_trace trace("dtl.algorithm.reduce");
///     // ... perform reduce ...
/// }  // trace ends here
/// @endcode
///
/// @since 0.1.0
class scoped_trace {
public:
    /// @brief Construct a scoped trace
    /// @param operation_name The operation being traced
    explicit scoped_trace([[maybe_unused]] std::string_view operation_name) noexcept
#if DTL_ENABLE_OBSERVABILITY
        : span_{operation_name}
#endif
    {}

    /// @brief Set the status of the underlying span
    /// @param s The status to set
    void set_status([[maybe_unused]] span_status s) noexcept {
#if DTL_ENABLE_OBSERVABILITY
        span_.set_status(s);
#endif
    }

    /// @brief Mark the trace as successful
    void set_ok() noexcept {
#if DTL_ENABLE_OBSERVABILITY
        span_.set_status(span_status::ok);
#endif
    }

    /// @brief Mark the trace as failed
    void set_error() noexcept {
#if DTL_ENABLE_OBSERVABILITY
        span_.set_status(span_status::error);
#endif
    }

    /// @brief Get the elapsed duration
    /// @return The elapsed duration, or zero when observability is disabled
    [[nodiscard]] trace_span::duration elapsed() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return span_.elapsed();
#else
        return trace_span::duration::zero();
#endif
    }

private:
#if DTL_ENABLE_OBSERVABILITY
    trace_span span_;
#endif
};

}  // namespace dtl::observe
