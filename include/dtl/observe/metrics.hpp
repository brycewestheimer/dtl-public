// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file metrics.hpp
/// @brief Metric types for DTL observability
/// @details Provides counter, gauge, and histogram metric types for
///          instrumenting DTL operations. All types are no-op stubs when
///          DTL_ENABLE_OBSERVABILITY is not defined, guaranteeing zero
///          overhead in production builds.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace dtl::observe {

// =============================================================================
// Observability Feature Gate
// =============================================================================

#ifndef DTL_ENABLE_OBSERVABILITY
    #define DTL_ENABLE_OBSERVABILITY 0
#endif

// =============================================================================
// Metric Counter
// =============================================================================

/// @brief Monotonically increasing counter metric
/// @details Tracks cumulative values such as total bytes transferred,
///          total operations completed, or total errors encountered.
///          When DTL_ENABLE_OBSERVABILITY is disabled, all operations
///          are compiled away to nothing.
/// @since 0.1.0
class metric_counter {
public:
    /// @brief Construct a named counter
    /// @param name Human-readable metric name (e.g., "dtl.comm.bytes_sent")
    /// @param description Optional description of what this counter measures
    explicit metric_counter([[maybe_unused]] std::string_view name,
                            [[maybe_unused]] std::string_view description = "")
#if DTL_ENABLE_OBSERVABILITY
        : name_{name}, description_{description}, value_{0}
#endif
    {}

    /// @brief Increment the counter by one
    void increment() noexcept {
#if DTL_ENABLE_OBSERVABILITY
        value_.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    /// @brief Increment the counter by a specified amount
    /// @param delta The amount to add (must be non-negative)
    void increment([[maybe_unused]] std::int64_t delta) noexcept {
#if DTL_ENABLE_OBSERVABILITY
        value_.fetch_add(delta, std::memory_order_relaxed);
#endif
    }

    /// @brief Get the current counter value
    /// @return The cumulative count, or 0 when observability is disabled
    [[nodiscard]] std::int64_t value() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return value_.load(std::memory_order_relaxed);
#else
        return 0;
#endif
    }

    /// @brief Reset the counter to zero
    void reset() noexcept {
#if DTL_ENABLE_OBSERVABILITY
        value_.store(0, std::memory_order_relaxed);
#endif
    }

    /// @brief Get the metric name
    /// @return The name, or empty string when observability is disabled
    [[nodiscard]] std::string_view name() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return name_;
#else
        return {};
#endif
    }

private:
#if DTL_ENABLE_OBSERVABILITY
    std::string name_;
    std::string description_;
    std::atomic<std::int64_t> value_;
#endif
};

// =============================================================================
// Metric Gauge
// =============================================================================

/// @brief Gauge metric that can increase or decrease
/// @details Tracks instantaneous values such as active connections,
///          current memory usage, or queue depth.
///          When DTL_ENABLE_OBSERVABILITY is disabled, all operations
///          are compiled away to nothing.
/// @since 0.1.0
class metric_gauge {
public:
    /// @brief Construct a named gauge
    /// @param name Human-readable metric name (e.g., "dtl.pool.active_connections")
    /// @param description Optional description of what this gauge measures
    explicit metric_gauge([[maybe_unused]] std::string_view name,
                          [[maybe_unused]] std::string_view description = "")
#if DTL_ENABLE_OBSERVABILITY
        : name_{name}, description_{description}, value_{0}
#endif
    {}

    /// @brief Set the gauge to a specific value
    /// @param val The new value
    void set([[maybe_unused]] double val) noexcept {
#if DTL_ENABLE_OBSERVABILITY
        value_.store(val, std::memory_order_relaxed);
#endif
    }

    /// @brief Increment the gauge by a specified amount
    /// @param delta The amount to add
    void increment([[maybe_unused]] double delta = 1.0) noexcept {
#if DTL_ENABLE_OBSERVABILITY
        auto current = value_.load(std::memory_order_relaxed);
        while (!value_.compare_exchange_weak(
            current, current + delta,
            std::memory_order_relaxed, std::memory_order_relaxed)) {}
#endif
    }

    /// @brief Decrement the gauge by a specified amount
    /// @param delta The amount to subtract
    void decrement([[maybe_unused]] double delta = 1.0) noexcept {
#if DTL_ENABLE_OBSERVABILITY
        increment(-delta);
#endif
    }

    /// @brief Get the current gauge value
    /// @return The current value, or 0.0 when observability is disabled
    [[nodiscard]] double value() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return value_.load(std::memory_order_relaxed);
#else
        return 0.0;
#endif
    }

    /// @brief Get the metric name
    /// @return The name, or empty string when observability is disabled
    [[nodiscard]] std::string_view name() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return name_;
#else
        return {};
#endif
    }

private:
#if DTL_ENABLE_OBSERVABILITY
    std::string name_;
    std::string description_;
    std::atomic<double> value_;
#endif
};

// =============================================================================
// Metric Histogram
// =============================================================================

/// @brief Histogram metric for distribution tracking
/// @details Records observations (e.g., latency, message sizes) and
///          aggregates them into count, sum, min, and max.
///          When DTL_ENABLE_OBSERVABILITY is disabled, all operations
///          are compiled away to nothing.
/// @since 0.1.0
class metric_histogram {
public:
    /// @brief Construct a named histogram
    /// @param name Human-readable metric name (e.g., "dtl.comm.latency_us")
    /// @param description Optional description of what this histogram measures
    explicit metric_histogram([[maybe_unused]] std::string_view name,
                              [[maybe_unused]] std::string_view description = "")
#if DTL_ENABLE_OBSERVABILITY
        : name_{name}, description_{description}, count_{0}, sum_{0.0},
          min_{std::numeric_limits<double>::max()},
          max_{std::numeric_limits<double>::lowest()}
#endif
    {}

    /// @brief Record an observation
    /// @param value The observed value
    void observe([[maybe_unused]] double value) noexcept {
#if DTL_ENABLE_OBSERVABILITY
        count_.fetch_add(1, std::memory_order_relaxed);

        // Atomic add for sum
        auto current_sum = sum_.load(std::memory_order_relaxed);
        while (!sum_.compare_exchange_weak(
            current_sum, current_sum + value,
            std::memory_order_relaxed, std::memory_order_relaxed)) {}

        // Atomic min
        auto current_min = min_.load(std::memory_order_relaxed);
        while (value < current_min && !min_.compare_exchange_weak(
            current_min, value,
            std::memory_order_relaxed, std::memory_order_relaxed)) {}

        // Atomic max
        auto current_max = max_.load(std::memory_order_relaxed);
        while (value > current_max && !max_.compare_exchange_weak(
            current_max, value,
            std::memory_order_relaxed, std::memory_order_relaxed)) {}
#endif
    }

    /// @brief Get the number of recorded observations
    /// @return The count, or 0 when observability is disabled
    [[nodiscard]] std::int64_t count() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return count_.load(std::memory_order_relaxed);
#else
        return 0;
#endif
    }

    /// @brief Get the sum of all recorded observations
    /// @return The sum, or 0.0 when observability is disabled
    [[nodiscard]] double sum() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return sum_.load(std::memory_order_relaxed);
#else
        return 0.0;
#endif
    }

    /// @brief Get the minimum recorded observation
    /// @return The minimum, or 0.0 when observability is disabled
    [[nodiscard]] double min() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        auto c = count_.load(std::memory_order_relaxed);
        return c > 0 ? min_.load(std::memory_order_relaxed) : 0.0;
#else
        return 0.0;
#endif
    }

    /// @brief Get the maximum recorded observation
    /// @return The maximum, or 0.0 when observability is disabled
    [[nodiscard]] double max() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        auto c = count_.load(std::memory_order_relaxed);
        return c > 0 ? max_.load(std::memory_order_relaxed) : 0.0;
#else
        return 0.0;
#endif
    }

    /// @brief Reset all histogram state
    void reset() noexcept {
#if DTL_ENABLE_OBSERVABILITY
        count_.store(0, std::memory_order_relaxed);
        sum_.store(0.0, std::memory_order_relaxed);
        min_.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
        max_.store(std::numeric_limits<double>::lowest(), std::memory_order_relaxed);
#endif
    }

    /// @brief Get the metric name
    /// @return The name, or empty string when observability is disabled
    [[nodiscard]] std::string_view name() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        return name_;
#else
        return {};
#endif
    }

private:
#if DTL_ENABLE_OBSERVABILITY
    std::string name_;
    std::string description_;
    std::atomic<std::int64_t> count_;
    std::atomic<double> sum_;
    std::atomic<double> min_;
    std::atomic<double> max_;
#endif
};

}  // namespace dtl::observe
