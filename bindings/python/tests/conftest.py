# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Pytest configuration and fixtures for DTL Python bindings tests.
"""

import pytest
import sys
import os
from typing import Generator

# Check if mpi4py is available without forcing MPI runtime initialization.
# This keeps non-MPI test runs stable in environments where importing MPI
# eagerly can fail (e.g., restricted CI/sandbox containers).
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


def _has_mpi_launcher_env() -> bool:
    """Return True when running under an MPI launcher environment."""
    return any(
        name in os.environ
        for name in (
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_RANK",
            "MV2_COMM_WORLD_SIZE",
            "MPI_LOCALNRANKS",
        )
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "mpi: mark test as requiring MPI (deselect with '-m \"not mpi\"')",
    )
    config.addinivalue_line(
        "markers",
        "cuda: mark test as requiring CUDA (deselect with '-m \"not cuda\"')",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests based on available features."""
    skip_mpi = pytest.mark.skip(
        reason=(
            "MPI tests require mpi4py, DTL MPI-enabled build, and an MPI-launched runtime"
        )
    )
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_dtl = pytest.mark.skip(reason="DTL module not available")

    dtl_mpi_enabled = HAS_DTL and bool(getattr(dtl, "has_mpi", lambda: False)())
    mpi_runtime_usable = HAS_MPI4PY and dtl_mpi_enabled and _has_mpi_launcher_env()

    for item in items:
        # Skip if DTL not available
        if not HAS_DTL:
            item.add_marker(skip_dtl)
            continue

        # Skip MPI tests unless mpi4py is available, the DTL build has MPI,
        # and tests are running under an MPI launcher.
        if "mpi" in item.keywords and not mpi_runtime_usable:
            item.add_marker(skip_mpi)

        # Skip CUDA tests if CUDA not available
        if "cuda" in item.keywords:
            if not HAS_DTL or not dtl.has_cuda():
                item.add_marker(skip_cuda)


@pytest.fixture
def context() -> Generator["dtl.Context", None, None]:
    """Provide a DTL context for tests."""
    import dtl

    with dtl.Context() as ctx:
        yield ctx


@pytest.fixture
def mpi_context() -> Generator["dtl.Context", None, None]:
    """Provide a DTL context with MPI communicator."""
    import dtl

    if not HAS_MPI4PY:
        pytest.skip("mpi4py not available")
    if not dtl.has_mpi():
        pytest.skip("DTL was built without MPI support")
    if not _has_mpi_launcher_env():
        pytest.skip("MPI tests must run under an MPI launcher (mpirun/mpiexec)")

    from mpi4py import MPI

    with dtl.Context(comm=MPI.COMM_WORLD) as ctx:
        yield ctx


@pytest.fixture
def cuda_context() -> Generator["dtl.Context", None, None]:
    """Provide a DTL context with CUDA device."""
    import dtl

    if dtl.has_cuda():
        with dtl.Context(device_id=0) as ctx:
            yield ctx
    else:
        pytest.skip("CUDA not available")


@pytest.fixture
def rank() -> int:
    """Get current MPI rank (0 if not using MPI)."""
    if HAS_MPI4PY:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank()
    return 0


@pytest.fixture
def size() -> int:
    """Get MPI world size (1 if not using MPI)."""
    if HAS_MPI4PY:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size()
    return 1


def is_root() -> bool:
    """Check if this is the root rank."""
    if HAS_MPI4PY:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank() == 0
    return True
