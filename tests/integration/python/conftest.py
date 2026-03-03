# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Pytest configuration and fixtures for DTL Python MPI integration tests.
"""

import pytest
import sys
from typing import Generator

# Check if mpi4py is available without forcing MPI runtime initialization.
try:
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    HAS_MPI4PY = True
except Exception:
    HAS_MPI4PY = False

# Check if DTL module is available
try:
    import dtl
    HAS_DTL = True
except ImportError:
    HAS_DTL = False


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "mpi: mark test as requiring MPI",
    )
    config.addinivalue_line(
        "markers",
        "multirank: mark test as requiring multiple MPI ranks",
    )
    config.addinivalue_line(
        "markers",
        "cuda: mark test as requiring CUDA",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests based on available features."""
    skip_dtl = pytest.mark.skip(reason="DTL module not available")
    skip_mpi = pytest.mark.skip(reason="mpi4py not available")

    for item in items:
        if not HAS_DTL:
            item.add_marker(skip_dtl)
            continue

        if "mpi" in item.keywords and not HAS_MPI4PY:
            item.add_marker(skip_mpi)

        if "multirank" in item.keywords:
            if not HAS_MPI4PY:
                item.add_marker(skip_mpi)
            elif MPI.COMM_WORLD.Get_size() < 2:
                item.add_marker(pytest.mark.skip(
                    reason="Test requires multiple MPI ranks (run with mpirun -np N)"
                ))

        if "cuda" in item.keywords:
            if not HAS_DTL or not dtl.has_cuda():
                item.add_marker(pytest.mark.skip(reason="CUDA not available"))


@pytest.fixture
def context() -> Generator:
    """Provide a DTL context for tests."""
    import dtl

    with dtl.Context() as ctx:
        yield ctx


@pytest.fixture
def mpi_context() -> Generator:
    """Provide a DTL context with MPI communicator."""
    import dtl

    if HAS_MPI4PY:
        from mpi4py import MPI
        with dtl.Context(comm=MPI.COMM_WORLD) as ctx:
            yield ctx
    else:
        with dtl.Context() as ctx:
            yield ctx
