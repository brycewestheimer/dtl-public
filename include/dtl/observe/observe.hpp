// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file observe.hpp
/// @brief Master include for DTL observability support
/// @details Provides single-header access to metrics, tracing, and the
///          metrics registry. All types are no-op stubs when
///          DTL_ENABLE_OBSERVABILITY is not defined, guaranteeing zero
///          overhead in production builds.
/// @since 0.1.0

#pragma once

// Metric types (counter, gauge, histogram)
#include <dtl/observe/metrics.hpp>

// Tracing types (trace_span, scoped_trace)
#include <dtl/observe/tracing.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dtl::observe {

// =============================================================================
// Observability Feature Gate
// =============================================================================

#ifndef DTL_ENABLE_OBSERVABILITY
    #define DTL_ENABLE_OBSERVABILITY 0
#endif

// =============================================================================
// Metrics Registry
// =============================================================================

/// @brief Singleton registry for managing named metrics
/// @details Provides a central point for creating and retrieving metrics
///          by name. When DTL_ENABLE_OBSERVABILITY is disabled, all
///          operations are no-ops that return references to static
///          thread-local stub objects.
///
/// Usage:
/// @code
/// auto& registry = dtl::observe::metrics_registry::instance();
/// auto& bytes_sent = registry.counter("dtl.comm.bytes_sent",
///                                     "Total bytes sent");
/// bytes_sent.increment(message_size);
/// @endcode
///
/// @since 0.1.0
class metrics_registry {
public:
    /// @brief Get the singleton instance
    /// @return Reference to the global metrics registry
    static metrics_registry& instance() {
        static metrics_registry registry;
        return registry;
    }

    /// @brief Get or create a counter metric
    /// @param name The metric name
    /// @param description Optional description
    /// @return Reference to the counter
    metric_counter& counter([[maybe_unused]] std::string_view name,
                            [[maybe_unused]] std::string_view description = "") {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        auto key = std::string{name};
        auto it = counters_.find(key);
        if (it == counters_.end()) {
            auto [inserted, _] = counters_.emplace(
                key, std::make_unique<metric_counter>(name, description));
            return *inserted->second;
        }
        return *it->second;
#else
        static metric_counter stub{"", ""};
        return stub;
#endif
    }

    /// @brief Get or create a gauge metric
    /// @param name The metric name
    /// @param description Optional description
    /// @return Reference to the gauge
    metric_gauge& gauge([[maybe_unused]] std::string_view name,
                        [[maybe_unused]] std::string_view description = "") {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        auto key = std::string{name};
        auto it = gauges_.find(key);
        if (it == gauges_.end()) {
            auto [inserted, _] = gauges_.emplace(
                key, std::make_unique<metric_gauge>(name, description));
            return *inserted->second;
        }
        return *it->second;
#else
        static metric_gauge stub{"", ""};
        return stub;
#endif
    }

    /// @brief Get or create a histogram metric
    /// @param name The metric name
    /// @param description Optional description
    /// @return Reference to the histogram
    metric_histogram& histogram([[maybe_unused]] std::string_view name,
                                [[maybe_unused]] std::string_view description = "") {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        auto key = std::string{name};
        auto it = histograms_.find(key);
        if (it == histograms_.end()) {
            auto [inserted, _] = histograms_.emplace(
                key, std::make_unique<metric_histogram>(name, description));
            return *inserted->second;
        }
        return *it->second;
#else
        static metric_histogram stub{"", ""};
        return stub;
#endif
    }

    /// @brief Get all registered counter names
    /// @return Vector of counter names, empty when observability is disabled
    [[nodiscard]] std::vector<std::string> counter_names() const {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        std::vector<std::string> names;
        names.reserve(counters_.size());
        for (const auto& [name, _] : counters_) {
            names.push_back(name);
        }
        return names;
#else
        return {};
#endif
    }

    /// @brief Get all registered gauge names
    /// @return Vector of gauge names, empty when observability is disabled
    [[nodiscard]] std::vector<std::string> gauge_names() const {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        std::vector<std::string> names;
        names.reserve(gauges_.size());
        for (const auto& [name, _] : gauges_) {
            names.push_back(name);
        }
        return names;
#else
        return {};
#endif
    }

    /// @brief Get all registered histogram names
    /// @return Vector of histogram names, empty when observability is disabled
    [[nodiscard]] std::vector<std::string> histogram_names() const {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        std::vector<std::string> names;
        names.reserve(histograms_.size());
        for (const auto& [name, _] : histograms_) {
            names.push_back(name);
        }
        return names;
#else
        return {};
#endif
    }

    /// @brief Get the total number of registered metrics
    /// @return Count of all metrics, 0 when observability is disabled
    [[nodiscard]] std::size_t size() const noexcept {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        return counters_.size() + gauges_.size() + histograms_.size();
#else
        return 0;
#endif
    }

    /// @brief Check if any metrics are registered
    /// @return True if empty, always true when observability is disabled
    [[nodiscard]] bool empty() const noexcept {
        return size() == 0;
    }

    /// @brief Clear all registered metrics
    void clear() noexcept {
#if DTL_ENABLE_OBSERVABILITY
        std::lock_guard<std::mutex> lock{mutex_};
        counters_.clear();
        gauges_.clear();
        histograms_.clear();
#endif
    }

private:
    metrics_registry() = default;
    ~metrics_registry() = default;

    metrics_registry(const metrics_registry&) = delete;
    metrics_registry& operator=(const metrics_registry&) = delete;

#if DTL_ENABLE_OBSERVABILITY
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<metric_counter>> counters_;
    std::unordered_map<std::string, std::unique_ptr<metric_gauge>> gauges_;
    std::unordered_map<std::string, std::unique_ptr<metric_histogram>> histograms_;
#endif
};

// =============================================================================
// Observe Module Summary
// =============================================================================
//
// The observe module provides lightweight instrumentation for DTL operations.
// All types are zero-overhead no-ops when DTL_ENABLE_OBSERVABILITY is not
// defined (the default).
//
// ============================================================================
// Metrics
// ============================================================================
//
// Three metric types are available:
//
// - metric_counter: Monotonically increasing counter
//   - increment(), increment(delta), value(), reset()
//
// - metric_gauge: Value that can increase or decrease
//   - set(val), increment(delta), decrement(delta), value()
//
// - metric_histogram: Distribution tracker
//   - observe(value), count(), sum(), min(), max(), reset()
//
// ============================================================================
// Tracing
// ============================================================================
//
// Two tracing types are available:
//
// - trace_span: RAII span with start/end timing
//   - end(), set_status(s), elapsed(), name(), status()
//
// - scoped_trace: Convenience wrapper for trace_span
//   - set_ok(), set_error(), elapsed()
//
// ============================================================================
// Registry
// ============================================================================
//
// The metrics_registry singleton manages named metrics:
//
// @code
// auto& reg = dtl::observe::metrics_registry::instance();
// reg.counter("bytes_sent").increment(n);
// reg.gauge("active_conns").set(count);
// reg.histogram("latency_us").observe(elapsed);
// @endcode
//
// ============================================================================

}  // namespace dtl::observe
