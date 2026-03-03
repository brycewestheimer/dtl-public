# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DTL Python observe module."""

import pytest
from dtl.observe import MetricCounter, MetricGauge, MetricHistogram


class TestMetricCounter:
    def test_increment(self):
        c = MetricCounter("test.counter", "A test counter")
        assert c.value == 0
        c.increment()
        assert c.value == 1
        c.increment(10)
        assert c.value == 11

    def test_reset(self):
        c = MetricCounter("test.counter")
        c.increment(5)
        c.reset()
        assert c.value == 0

    def test_negative_delta_raises(self):
        c = MetricCounter("test.counter")
        with pytest.raises(ValueError):
            c.increment(-1)

    def test_name_and_description(self):
        c = MetricCounter("my.counter", "Counts things")
        assert c.name == "my.counter"
        assert c.description == "Counts things"


class TestMetricGauge:
    def test_set_and_value(self):
        g = MetricGauge("test.gauge")
        g.set(42.0)
        assert g.value == 42.0

    def test_increment_decrement(self):
        g = MetricGauge("test.gauge")
        g.set(10.0)
        g.increment(5.0)
        assert g.value == 15.0
        g.decrement(3.0)
        assert g.value == 12.0

    def test_reset(self):
        g = MetricGauge("test.gauge")
        g.set(100.0)
        g.reset()
        assert g.value == 0.0


class TestMetricHistogram:
    def test_observe_and_mean(self):
        h = MetricHistogram("test.hist")
        h.observe(1.0)
        h.observe(2.0)
        h.observe(3.0)
        assert h.count == 3
        assert h.sum == 6.0
        assert h.mean == pytest.approx(2.0)

    def test_min_max(self):
        h = MetricHistogram("test.hist")
        h.observe(5.0)
        h.observe(1.0)
        h.observe(10.0)
        assert h.min == 1.0
        assert h.max == 10.0

    def test_empty_histogram(self):
        h = MetricHistogram("test.hist")
        assert h.count == 0
        assert h.mean == 0.0
        assert h.min == 0.0
        assert h.max == 0.0

    def test_reset(self):
        h = MetricHistogram("test.hist")
        h.observe(42.0)
        h.reset()
        assert h.count == 0
        assert h.sum == 0.0
