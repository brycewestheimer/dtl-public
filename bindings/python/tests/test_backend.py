# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DTL backend detection module."""

def test_backends_module_exists():
    """backends module is accessible."""
    import dtl
    assert hasattr(dtl, 'backends')

def test_has_mpi_returns_bool():
    import dtl
    result = dtl.backends.has_mpi()
    assert isinstance(result, bool)

def test_has_cuda_returns_bool():
    import dtl
    result = dtl.backends.has_cuda()
    assert isinstance(result, bool)

def test_has_hip_returns_bool():
    import dtl
    result = dtl.backends.has_hip()
    assert isinstance(result, bool)

def test_has_nccl_returns_bool():
    import dtl
    result = dtl.backends.has_nccl()
    assert isinstance(result, bool)

def test_has_shmem_returns_bool():
    import dtl
    result = dtl.backends.has_shmem()
    assert isinstance(result, bool)

def test_available_returns_list():
    import dtl
    result = dtl.backends.available()
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

def test_name_returns_string():
    import dtl
    result = dtl.backends.name()
    assert isinstance(result, str)
    assert len(result) > 0

def test_count_returns_int():
    import dtl
    result = dtl.backends.count()
    assert isinstance(result, int)
    assert result >= 0

def test_count_matches_available():
    import dtl
    assert dtl.backends.count() == len(dtl.backends.available())

def test_available_subset():
    """Available backends should be from known set."""
    import dtl
    known = {"MPI", "CUDA", "HIP", "NCCL", "SHMEM"}
    for backend in dtl.backends.available():
        assert backend in known

def test_existing_has_mpi_still_works():
    """Original top-level has_mpi() still works."""
    import dtl
    result = dtl.has_mpi()
    assert isinstance(result, bool)

def test_existing_has_cuda_still_works():
    import dtl
    result = dtl.has_cuda()
    assert isinstance(result, bool)

def test_version_updated():
    import dtl
    assert dtl.__version__ == "0.1.0a1"

def test_version_info_tuple():
    import dtl
    base = re.sub(r"(a|b|rc)\d+$", "", dtl.__version__)
    assert dtl.version_info == tuple(int(part) for part in base.split("."))
import re
