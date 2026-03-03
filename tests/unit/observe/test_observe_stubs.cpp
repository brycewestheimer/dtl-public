// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_observe_stubs.cpp
/// @brief Unit tests for dtl/observe/ no-op stubs
/// @details Verifies that all observe types compile, are default-constructible,
///          and return expected zero/empty values when DTL_ENABLE_OBSERVABILITY
///          is not defined.

#include <dtl/observe/observe.hpp>

#include <gtest/gtest.h>

namespace dtl::observe::test {

// =============================================================================
// Metric Counter Stub Tests
// =============================================================================

TEST(MetricCounterStubTest, DefaultValueIsZero) {
    metric_counter counter{"test.counter", "A test counter"};
    EXPECT_EQ(counter.value(), 0);
}

TEST(MetricCounterStubTest, IncrementIsNoOp) {
    metric_counter counter{"test.counter"};
    counter.increment();
    counter.increment(42);
    // When observability is disabled, value remains 0
    EXPECT_EQ(counter.value(), 0);
}

TEST(MetricCounterStubTest, ResetIsNoOp) {
    metric_counter counter{"test.counter"};
    counter.reset();
    EXPECT_EQ(counter.value(), 0);
}

// =============================================================================
// Metric Gauge Stub Tests
// =============================================================================

TEST(MetricGaugeStubTest, DefaultValueIsZero) {
    metric_gauge gauge{"test.gauge", "A test gauge"};
    EXPECT_DOUBLE_EQ(gauge.value(), 0.0);
}

TEST(MetricGaugeStubTest, SetAndIncrementAreNoOps) {
    metric_gauge gauge{"test.gauge"};
    gauge.set(100.0);
    gauge.increment(5.0);
    gauge.decrement(3.0);
    // When observability is disabled, value remains 0
    EXPECT_DOUBLE_EQ(gauge.value(), 0.0);
}

// =============================================================================
// Metric Histogram Stub Tests
// =============================================================================

TEST(MetricHistogramStubTest, DefaultValuesAreZero) {
    metric_histogram histogram{"test.histogram", "A test histogram"};
    EXPECT_EQ(histogram.count(), 0);
    EXPECT_DOUBLE_EQ(histogram.sum(), 0.0);
    EXPECT_DOUBLE_EQ(histogram.min(), 0.0);
    EXPECT_DOUBLE_EQ(histogram.max(), 0.0);
}

TEST(MetricHistogramStubTest, ObserveIsNoOp) {
    metric_histogram histogram{"test.histogram"};
    histogram.observe(1.5);
    histogram.observe(3.0);
    histogram.observe(7.5);
    // When observability is disabled, all aggregates remain 0
    EXPECT_EQ(histogram.count(), 0);
    EXPECT_DOUBLE_EQ(histogram.sum(), 0.0);
}

// =============================================================================
// Trace Span Stub Tests
// =============================================================================

TEST(TraceSpanStubTest, ConstructionAndDestructionAreNoOps) {
    // Should compile and run without error
    {
        trace_span span{"test.operation"};
        span.set_status(span_status::ok);
        span.end();
        EXPECT_EQ(span.elapsed().count(), 0);
    }
}

TEST(TraceSpanStubTest, ScopedTraceIsNoOp) {
    scoped_trace trace{"test.scoped"};
    trace.set_ok();
    trace.set_error();
    EXPECT_EQ(trace.elapsed().count(), 0);
}

// =============================================================================
// Metrics Registry Stub Tests
// =============================================================================

TEST(MetricsRegistryStubTest, SingletonAccessWorks) {
    auto& reg1 = metrics_registry::instance();
    auto& reg2 = metrics_registry::instance();
    EXPECT_EQ(&reg1, &reg2);
}

TEST(MetricsRegistryStubTest, RegistryReturnsStubs) {
    auto& reg = metrics_registry::instance();

    auto& counter = reg.counter("test.reg.counter", "A registered counter");
    counter.increment(10);
    EXPECT_EQ(counter.value(), 0);

    auto& gauge = reg.gauge("test.reg.gauge");
    gauge.set(42.0);
    EXPECT_DOUBLE_EQ(gauge.value(), 0.0);

    auto& hist = reg.histogram("test.reg.histogram");
    hist.observe(1.0);
    EXPECT_EQ(hist.count(), 0);
}

TEST(MetricsRegistryStubTest, RegistryIsEmptyWhenDisabled) {
    auto& reg = metrics_registry::instance();
    EXPECT_TRUE(reg.empty());
    EXPECT_EQ(reg.size(), 0u);
    EXPECT_TRUE(reg.counter_names().empty());
    EXPECT_TRUE(reg.gauge_names().empty());
    EXPECT_TRUE(reg.histogram_names().empty());
}

TEST(MetricsRegistryStubTest, ClearIsNoOp) {
    auto& reg = metrics_registry::instance();
    reg.clear();
    EXPECT_TRUE(reg.empty());
}

}  // namespace dtl::observe::test
