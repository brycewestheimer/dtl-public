# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""DTL Observability Module — Python bindings for dtl::observe metrics.

Provides MetricCounter, MetricGauge, and MetricHistogram classes that
mirror the C++ dtl::observe metric types. These are pure-Python
implementations for use in Python-based DTL workflows.

Note: These do NOT wrap the C++ implementations. They are standalone
Python classes with the same API semantics.

.. versionadded:: 0.1.0a1
"""

from __future__ import annotations


class MetricCounter:
    """Monotonically increasing counter metric.

    Tracks cumulative values such as total bytes transferred,
    total operations completed, or total errors encountered.

    Example::

        counter = MetricCounter("dtl.comm.bytes_sent", "Total bytes sent")
        counter.increment()
        counter.increment(1024)
        print(counter.value)  # 1025
    """

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._value: int = 0

    @property
    def name(self) -> str:
        """The metric name."""
        return self._name

    @property
    def description(self) -> str:
        """The metric description."""
        return self._description

    @property
    def value(self) -> int:
        """The current counter value."""
        return self._value

    def increment(self, delta: int = 1) -> None:
        """Increment the counter by delta (default 1)."""
        if delta < 0:
            raise ValueError("Counter delta must be non-negative")
        self._value += delta

    def reset(self) -> None:
        """Reset the counter to zero."""
        self._value = 0


class MetricGauge:
    """Gauge metric that can increase or decrease.

    Tracks instantaneous values such as active connections,
    current memory usage, or queue depth.

    Example::

        gauge = MetricGauge("dtl.pool.active", "Active connections")
        gauge.set(10.0)
        gauge.increment(5.0)
        gauge.decrement(3.0)
        print(gauge.value)  # 12.0
    """

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._value: float = 0.0

    @property
    def name(self) -> str:
        """The metric name."""
        return self._name

    @property
    def description(self) -> str:
        """The metric description."""
        return self._description

    @property
    def value(self) -> float:
        """The current gauge value."""
        return self._value

    def set(self, val: float) -> None:
        """Set the gauge to a specific value."""
        self._value = val

    def increment(self, delta: float = 1.0) -> None:
        """Increment the gauge by delta (default 1.0)."""
        self._value += delta

    def decrement(self, delta: float = 1.0) -> None:
        """Decrement the gauge by delta (default 1.0)."""
        self._value -= delta

    def reset(self) -> None:
        """Reset the gauge to zero."""
        self._value = 0.0


class MetricHistogram:
    """Histogram metric for distribution tracking.

    Records observations and aggregates them into count, sum, min, and max.

    Example::

        hist = MetricHistogram("dtl.comm.latency_us", "Communication latency")
        hist.observe(1.5)
        hist.observe(2.5)
        hist.observe(3.0)
        print(hist.mean)  # ~2.33
    """

    def __init__(self, name: str, description: str = "") -> None:
        self._name = name
        self._description = description
        self._count: int = 0
        self._sum: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    @property
    def name(self) -> str:
        """The metric name."""
        return self._name

    @property
    def description(self) -> str:
        """The metric description."""
        return self._description

    @property
    def count(self) -> int:
        """Number of recorded observations."""
        return self._count

    @property
    def sum(self) -> float:
        """Sum of all observations."""
        return self._sum

    @property
    def min(self) -> float:
        """Minimum observed value, or 0.0 if no observations."""
        return self._min if self._count > 0 else 0.0

    @property
    def max(self) -> float:
        """Maximum observed value, or 0.0 if no observations."""
        return self._max if self._count > 0 else 0.0

    @property
    def mean(self) -> float:
        """Mean of all observations, or 0.0 if no observations."""
        return self._sum / self._count if self._count > 0 else 0.0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self._count += 1
        self._sum += value
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    def reset(self) -> None:
        """Reset all histogram state."""
        self._count = 0
        self._sum = 0.0
        self._min = float("inf")
        self._max = float("-inf")
