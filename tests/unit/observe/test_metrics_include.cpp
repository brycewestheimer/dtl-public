// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_metrics_include.cpp
/// @brief Verify metrics.hpp is self-contained (includes <limits>)

#include <dtl/observe/metrics.hpp>
#include <gtest/gtest.h>

TEST(MetricsInclude, HistogramConstructsWithoutError) {
    dtl::observe::metric_histogram h("test.latency", "Test histogram");
    EXPECT_EQ(h.count(), 0);
    EXPECT_DOUBLE_EQ(h.sum(), 0.0);
}

TEST(MetricsInclude, CounterAndGaugeBasic) {
    dtl::observe::metric_counter c("test.counter");
    c.increment();
    EXPECT_EQ(c.value(), 0);  // observability disabled by default

    dtl::observe::metric_gauge g("test.gauge");
    g.set(42.0);
    EXPECT_DOUBLE_EQ(g.value(), 0.0);  // observability disabled by default
}
