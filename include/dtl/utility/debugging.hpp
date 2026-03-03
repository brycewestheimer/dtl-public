// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file debugging.hpp
/// @brief Debug output, assertions, and diagnostics
/// @details Provides debugging utilities for DTL development and troubleshooting.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <source_location>

namespace dtl {

// ============================================================================
// Debug Levels
// ============================================================================

/// @brief Debug verbosity levels
enum class debug_level {
    /// @brief No debug output
    none = 0,

    /// @brief Errors only
    error = 1,

    /// @brief Warnings and errors
    warning = 2,

    /// @brief Informational messages
    info = 3,

    /// @brief Detailed debug output
    debug = 4,

    /// @brief Extremely verbose trace output
    trace = 5
};

/// @brief Get string representation of debug level
[[nodiscard]] constexpr std::string_view to_string(debug_level level) noexcept {
    switch (level) {
        case debug_level::none: return "NONE";
        case debug_level::error: return "ERROR";
        case debug_level::warning: return "WARN";
        case debug_level::info: return "INFO";
        case debug_level::debug: return "DEBUG";
        case debug_level::trace: return "TRACE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Debug Configuration
// ============================================================================

namespace detail {

/// @brief Global debug level (atomic for thread safety)
inline std::atomic<debug_level> g_debug_level{
#ifdef DTL_DEBUG
    debug_level::debug
#else
    debug_level::warning
#endif
};

/// @brief Cached rank for debug output (atomic for thread safety)
inline std::atomic<rank_t> g_debug_rank{no_rank};

/// @brief Whether to include rank in debug output (atomic for thread safety)
inline std::atomic<bool> g_debug_show_rank{true};

/// @brief Whether to include source location in debug output (atomic for thread safety)
inline std::atomic<bool> g_debug_show_location{true};

/// @brief Output stream for debug messages (atomic pointer for thread safety)
inline std::atomic<std::ostream*> g_debug_stream{&std::cerr};

/// @brief Mutex protecting writes to the configured debug stream
inline std::mutex g_debug_stream_mutex;

}  // namespace detail

/// @brief Set the global debug level
/// @param level New debug level
inline void set_debug_level(debug_level level) noexcept {
    detail::g_debug_level.store(level, std::memory_order_relaxed);
}

/// @brief Get the current debug level
[[nodiscard]] inline debug_level get_debug_level() noexcept {
    return detail::g_debug_level.load(std::memory_order_relaxed);
}

/// @brief Set the rank for debug output
/// @param rank Current process rank
inline void set_debug_rank(rank_t rank) noexcept {
    detail::g_debug_rank.store(rank, std::memory_order_relaxed);
}

/// @brief Configure debug output options
/// @param show_rank Whether to show rank in output
/// @param show_location Whether to show source location
inline void configure_debug(bool show_rank, bool show_location) noexcept {
    detail::g_debug_show_rank.store(show_rank, std::memory_order_relaxed);
    detail::g_debug_show_location.store(show_location, std::memory_order_relaxed);
}

/// @brief Set the debug output stream
/// @param stream Output stream to use
inline void set_debug_stream(std::ostream& stream) noexcept {
    std::lock_guard<std::mutex> lock(detail::g_debug_stream_mutex);
    detail::g_debug_stream.store(&stream, std::memory_order_relaxed);
}

// ============================================================================
// Debug Output Functions
// ============================================================================

namespace detail {

/// @brief Format and output a debug message
template <typename... Args>
void debug_output(debug_level level, const std::source_location& loc, Args&&... args) {
    if (level > g_debug_level.load(std::memory_order_relaxed)) return;

    const auto show_rank = g_debug_show_rank.load(std::memory_order_relaxed);
    const auto debug_rank = g_debug_rank.load(std::memory_order_relaxed);
    const auto show_location = g_debug_show_location.load(std::memory_order_relaxed);
    auto* stream = g_debug_stream.load(std::memory_order_relaxed);

    std::ostringstream oss;

    // Level prefix
    oss << "[" << to_string(level) << "]";

    // Rank prefix
    if (show_rank && debug_rank != no_rank) {
        oss << "[R" << debug_rank << "]";
    }

    // Source location
    if (show_location) {
        oss << " " << loc.file_name() << ":" << loc.line();
        if (loc.function_name()[0] != '\0') {
            oss << " (" << loc.function_name() << ")";
        }
    }

    oss << ": ";

    // Message
    ((oss << std::forward<Args>(args)), ...);
    oss << "\n";

    {
        std::lock_guard<std::mutex> lock(g_debug_stream_mutex);
        *stream << oss.str();
        stream->flush();
    }
}

}  // namespace detail

/// @brief Output an error message
template <typename... Args>
void debug_error(Args&&... args,
                 const std::source_location loc = std::source_location::current()) {
    detail::debug_output(debug_level::error, loc, std::forward<Args>(args)...);
}

/// @brief Output a warning message
template <typename... Args>
void debug_warning(Args&&... args,
                   const std::source_location loc = std::source_location::current()) {
    detail::debug_output(debug_level::warning, loc, std::forward<Args>(args)...);
}

/// @brief Output an info message
template <typename... Args>
void debug_info(Args&&... args,
                const std::source_location loc = std::source_location::current()) {
    detail::debug_output(debug_level::info, loc, std::forward<Args>(args)...);
}

/// @brief Output a debug message
template <typename... Args>
void debug_msg(Args&&... args,
               const std::source_location loc = std::source_location::current()) {
    detail::debug_output(debug_level::debug, loc, std::forward<Args>(args)...);
}

/// @brief Output a trace message
template <typename... Args>
void debug_trace(Args&&... args,
                 const std::source_location loc = std::source_location::current()) {
    detail::debug_output(debug_level::trace, loc, std::forward<Args>(args)...);
}

// ============================================================================
// Assertions
// ============================================================================

/// @brief DTL assertion failure handler
/// @param condition The failed condition as string
/// @param message Optional message
/// @param loc Source location
[[noreturn]] inline void assertion_failed(
    std::string_view condition,
    std::string_view message,
    const std::source_location& loc) {

    std::ostringstream oss;
    oss << "DTL Assertion Failed!\n";
    oss << "  Condition: " << condition << "\n";
    if (!message.empty()) {
        oss << "  Message: " << message << "\n";
    }
    oss << "  Location: " << loc.file_name() << ":" << loc.line() << "\n";
    oss << "  Function: " << loc.function_name() << "\n";

    if (detail::g_debug_rank.load(std::memory_order_relaxed) != no_rank) {
        oss << "  Rank: " << detail::g_debug_rank.load(std::memory_order_relaxed) << "\n";
    }

    std::cerr << oss.str() << std::flush;
    std::abort();
}

/// @brief DTL assertion with message
/// @param condition Condition to check
/// @param message Message if assertion fails
#ifdef DTL_ASSERT_MSG
#undef DTL_ASSERT_MSG
#endif
#define DTL_ASSERT_MSG(condition, message)                                    \
    do {                                                                       \
        if (!(condition)) {                                                    \
            ::dtl::assertion_failed(#condition, message,                       \
                                    std::source_location::current());          \
        }                                                                      \
    } while (false)

/// @brief DTL assertion
/// @param condition Condition to check
#ifdef DTL_ASSERT
#undef DTL_ASSERT
#endif
#define DTL_ASSERT(condition) DTL_ASSERT_MSG(condition, "")

/// @brief Debug-only assertion (disabled in release builds)
#ifdef DTL_DEBUG_ASSERT
#undef DTL_DEBUG_ASSERT
#endif
#ifdef DTL_DEBUG_ASSERT_MSG
#undef DTL_DEBUG_ASSERT_MSG
#endif
#ifdef DTL_DEBUG
#define DTL_DEBUG_ASSERT(condition) DTL_ASSERT(condition)
#define DTL_DEBUG_ASSERT_MSG(condition, message) DTL_ASSERT_MSG(condition, message)
#else
#define DTL_DEBUG_ASSERT(condition) ((void)0)
#define DTL_DEBUG_ASSERT_MSG(condition, message) ((void)0)
#endif

/// @brief Precondition check
#ifdef DTL_PRECONDITION
#undef DTL_PRECONDITION
#endif
#define DTL_PRECONDITION(condition) DTL_ASSERT_MSG(condition, "Precondition violated")

/// @brief Postcondition check
#ifdef DTL_POSTCONDITION
#undef DTL_POSTCONDITION
#endif
#define DTL_POSTCONDITION(condition) DTL_ASSERT_MSG(condition, "Postcondition violated")

/// @brief Invariant check
#ifdef DTL_INVARIANT
#undef DTL_INVARIANT
#endif
#define DTL_INVARIANT(condition) DTL_ASSERT_MSG(condition, "Invariant violated")

/// @brief Unreachable code marker
#ifdef DTL_UNREACHABLE
#undef DTL_UNREACHABLE
#endif
#define DTL_UNREACHABLE()                                                     \
    do {                                                                       \
        ::dtl::assertion_failed("unreachable", "Code path should be unreachable", \
                                std::source_location::current());              \
    } while (false)

// ============================================================================
// Diagnostic Utilities
// ============================================================================
// NOTE: memory_diagnostics/communication_diagnostics structs and their
// accessor functions (get_memory_diagnostics, reset_memory_diagnostics,
// get_communication_diagnostics, reset_communication_diagnostics) were
// removed because they were declared but never defined (linker bombs).
// If diagnostics are needed in the future, implement them as a complete
// feature with both declaration and definition.

// ============================================================================
// Timer Utilities
// ============================================================================

/// @brief Simple scoped timer for performance measurement
class scoped_timer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using duration = std::chrono::duration<double>;

    /// @brief Construct and start timer
    /// @param name Timer name for output
    /// @param level Debug level for output
    /// @param loc Source location associated with this timer
    explicit scoped_timer(
        std::string name,
        debug_level level = debug_level::info,
        const std::source_location loc = std::source_location::current())
        : name_(std::move(name))
        , level_(level)
        , loc_(loc)
        , start_(clock::now()) {}

    /// @brief Destructor - outputs elapsed time
    ~scoped_timer() {
        auto end = clock::now();
        duration elapsed = end - start_;
        detail::debug_output(level_, loc_, name_, " took ", elapsed.count(), "s");
    }

    /// @brief Get elapsed time so far
    [[nodiscard]] double elapsed() const {
        duration d = clock::now() - start_;
        return d.count();
    }

    // Non-copyable, non-movable
    scoped_timer(const scoped_timer&) = delete;
    scoped_timer& operator=(const scoped_timer&) = delete;
    scoped_timer(scoped_timer&&) = delete;
    scoped_timer& operator=(scoped_timer&&) = delete;

private:
    std::string name_;
    debug_level level_;
    std::source_location loc_;
    time_point start_;
};

/// @brief Convenience macro for scoped timing
/// @details Uses two-level macro indirection to properly expand __LINE__
#define DTL_CONCAT_IMPL_(a, b) a##b
#define DTL_CONCAT_(a, b) DTL_CONCAT_IMPL_(a, b)
#define DTL_TIMED_SCOPE(name) ::dtl::scoped_timer DTL_CONCAT_(_dtl_timer_, __LINE__)(name)

// ============================================================================
// Value Inspection
// ============================================================================

/// @brief Print a value with its name and location
#define DTL_INSPECT(value)                                                    \
    ::dtl::debug_msg(#value, " = ", value)

/// @brief Print multiple values
#define DTL_INSPECT2(v1, v2)                                                  \
    ::dtl::debug_msg(#v1, " = ", v1, ", ", #v2, " = ", v2)

#define DTL_INSPECT3(v1, v2, v3)                                              \
    ::dtl::debug_msg(#v1, " = ", v1, ", ", #v2, " = ", v2, ", ", #v3, " = ", v3)

// ============================================================================
// Rank-Specific Output
// ============================================================================

/// @brief Output only from rank 0
template <typename... Args>
void debug_root(Args&&... args,
                const std::source_location loc = std::source_location::current()) {
    if (detail::g_debug_rank.load(std::memory_order_relaxed) == 0 || detail::g_debug_rank.load(std::memory_order_relaxed) == no_rank) {
        detail::debug_output(debug_level::info, loc, std::forward<Args>(args)...);
    }
}

/// @brief Output only from a specific rank
template <typename... Args>
void debug_rank(rank_t rank, Args&&... args,
                const std::source_location loc = std::source_location::current()) {
    if (detail::g_debug_rank.load(std::memory_order_relaxed) == rank) {
        detail::debug_output(debug_level::info, loc, std::forward<Args>(args)...);
    }
}

// NOTE: debug_serialized() was removed because it was declared but never
// defined (linker bomb). If serialized debug output is needed, implement it
// as a complete feature.

}  // namespace dtl
