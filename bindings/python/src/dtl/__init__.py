# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
DTL - Distributed Template Library Python Bindings

A Python interface to the DTL distributed computing library.
Provides NumPy-compatible distributed arrays with MPI communication.

Basic Usage:
    >>> import dtl
    >>> import numpy as np

    >>> with dtl.Context() as ctx:
    ...     vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float64)
    ...     local = vec.local_view()
    ...     local[:] = np.arange(len(local))
    ...     ctx.barrier()

For MPI parallelism:
    >>> from mpi4py import MPI
    >>> ctx = dtl.Context(comm=MPI.COMM_WORLD)
"""

from __future__ import annotations

__version__ = "0.1.0a1"
__all__ = [
    # Version info
    "__version__",
    "version_info",
    # Feature detection
    "has_mpi",
    "has_cuda",
    "has_hip",
    "has_nccl",
    "has_shmem",
    # Backend detection module
    "backends",
    # Observability module (Phase 29)
    "observe",
    # Context and Environment
    "Context",
    "Environment",
    # Containers
    "DistributedVector",
    "DistributedArray",
    "DistributedSpan",
    "DistributedTensor",
    # Policies (enums)
    "PartitionPolicy",
    "PlacementPolicy",
    "ExecutionPolicy",
    # Policy constants (convenience)
    "PARTITION_BLOCK",
    "PARTITION_CYCLIC",
    "PARTITION_BLOCK_CYCLIC",
    "PARTITION_HASH",
    "PARTITION_REPLICATED",
    "PLACEMENT_HOST",
    "PLACEMENT_DEVICE",
    "PLACEMENT_UNIFIED",
    "PLACEMENT_DEVICE_PREFERRED",
    "EXEC_SEQ",
    "EXEC_PAR",
    "EXEC_ASYNC",
    # Policy utilities
    "placement_available",
    # Collective operations
    "allreduce",
    "reduce",
    "broadcast",
    "gather",
    "scatter",
    "allgather",
    "allgatherv",
    "alltoallv",
    # Variable-count collectives
    "gatherv",
    "scatterv",
    # Point-to-point operations
    "send",
    "recv",
    "sendrecv",
    # Message probing
    "probe",
    "iprobe",
    # Reduce operations
    "SUM",
    "PROD",
    "MIN",
    "MAX",
    # Algorithms
    "for_each_vector",
    "for_each_array",
    "copy_vector",
    "copy_array",
    "fill_vector",
    "fill_array",
    "find_vector",
    "find_if_vector",
    "find_array",
    "find_if_array",
    "count_vector",
    "count_if_vector",
    "count_array",
    "count_if_array",
    "reduce_local_vector",
    "reduce_local_array",
    "sort_vector",
    "sort_array",
    "minmax_vector",
    "minmax_array",
    "transform_vector",
    "transform_array",
    "inclusive_scan_vector",
    "exclusive_scan_vector",
    "inclusive_scan_array",
    "exclusive_scan_array",
    # Predicate queries (Phase 16)
    "all_of_vector",
    "any_of_vector",
    "none_of_vector",
    "all_of_array",
    "any_of_array",
    "none_of_array",
    # Extrema element (Phase 16)
    "min_element_vector",
    "max_element_vector",
    "min_element_array",
    "max_element_array",
    # Async algorithms (Phase 28)
    "async_for_each",
    "async_transform",
    "async_reduce",
    "async_sort",
    "AlgorithmFuture",
    # Device migration (Phase 28)
    "to_device",
    "to_host",
    # Distributed Map (Phase 16)
    "DistributedMap",
    # RMA operations
    "Window",
    "rma_put",
    "rma_get",
    "rma_accumulate",
    "rma_fetch_and_add",
    "rma_compare_and_swap",
    # Exceptions
    "DTLError",
    "CommunicationError",
    "MemoryError",
    "BoundsError",
    "InvalidArgumentError",
    "BackendError",
    # MPMD (Phase 12.5)
    "RoleManager",
    "intergroup_send",
    "intergroup_recv",
    # Topology (Phase 12.5)
    "Topology",
    # Futures (Phase 12.5, experimental)
    "Future",
    "when_all",
    "when_any",
    # Remote/RPC (Phase 12.5)
]

from typing import Tuple

# Import observe submodule (pure-Python, no native dependency)
from . import observe

# Import native extension
try:
    from . import _dtl
except ImportError as e:
    raise ImportError(
        f"Failed to import DTL native extension: {e}\n"
        "Make sure the package was properly installed."
    ) from e

__version__ = str(getattr(_dtl, "__version__", __version__))

def _parse_version(version: str) -> Tuple[int, int, int]:
    import re as _re
    # Strip PEP 440 prerelease suffixes (e.g. "0.1.0a1" -> "0.1.0")
    base = _re.sub(r"(a|b|rc)\d+$", "", version)
    parts = base.split(".")
    if len(parts) < 3:
        return (0, 0, 0)
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return (0, 0, 0)

# Version info tuple, derived from the native extension version.
version_info: Tuple[int, int, int] = _parse_version(__version__)


# =============================================================================
# Feature Detection
# =============================================================================

def has_mpi() -> bool:
    """Check if MPI backend is available."""
    return _dtl.has_mpi()


def has_cuda() -> bool:
    """Check if CUDA backend is available."""
    return _dtl.has_cuda()


def has_hip() -> bool:
    """Check if HIP/AMD backend is available."""
    return _dtl.has_hip()


def has_nccl() -> bool:
    """Check if NCCL is available."""
    return _dtl.has_nccl()


def has_shmem() -> bool:
    """Check if OpenSHMEM backend is available."""
    return _dtl.has_shmem()


# =============================================================================
# Policy Types
# =============================================================================

# Import policy enums from native module
PartitionPolicy = _dtl.PartitionPolicy
PlacementPolicy = _dtl.PlacementPolicy
ExecutionPolicy = _dtl.ExecutionPolicy

# Partition policy constants
PARTITION_BLOCK = _dtl.PARTITION_BLOCK
PARTITION_CYCLIC = _dtl.PARTITION_CYCLIC
PARTITION_BLOCK_CYCLIC = _dtl.PARTITION_BLOCK_CYCLIC
PARTITION_HASH = _dtl.PARTITION_HASH
PARTITION_REPLICATED = _dtl.PARTITION_REPLICATED

# Placement policy constants
PLACEMENT_HOST = _dtl.PLACEMENT_HOST
PLACEMENT_DEVICE = _dtl.PLACEMENT_DEVICE
PLACEMENT_UNIFIED = _dtl.PLACEMENT_UNIFIED
PLACEMENT_DEVICE_PREFERRED = _dtl.PLACEMENT_DEVICE_PREFERRED

# Execution policy constants
EXEC_SEQ = _dtl.EXEC_SEQ
EXEC_PAR = _dtl.EXEC_PAR
EXEC_ASYNC = _dtl.EXEC_ASYNC


def placement_available(policy) -> bool:
    """Check if a placement policy is available.

    Some placements (device, unified, device_preferred) require CUDA support.

    Args:
        policy: The placement policy to check

    Returns:
        True if the placement is available
    """
    return _dtl.placement_available(policy) != 0


# =============================================================================
# Exceptions
# =============================================================================

class DTLError(Exception):
    """Base class for all DTL exceptions."""
    pass


class CommunicationError(DTLError):
    """Raised when communication operations fail."""
    pass


class MemoryError(DTLError, MemoryError):
    """Raised on memory allocation failure."""
    pass


class BoundsError(DTLError, IndexError):
    """Raised on index out of bounds."""
    pass


class InvalidArgumentError(DTLError, ValueError):
    """Raised on invalid argument."""
    pass


class BackendError(DTLError):
    """Raised on backend failure."""
    pass


# =============================================================================
# Context
# =============================================================================

class Context:
    """Execution context for DTL operations.

    Encapsulates MPI communicator and device selection.
    Supports the context manager protocol for automatic cleanup.

    Args:
        comm: mpi4py MPI.Comm object, or None for MPI_COMM_WORLD
        device_id: GPU device ID, or -1 for CPU-only (default)

    Example:
        >>> with dtl.Context() as ctx:
        ...     print(f"Rank {ctx.rank} of {ctx.size}")

    With mpi4py:
        >>> from mpi4py import MPI
        >>> ctx = dtl.Context(comm=MPI.COMM_WORLD)

    Multi-domain context (V1.3.0):
        >>> # Check domain availability
        >>> if ctx.has_mpi:
        ...     print("MPI domain available")

        >>> # Add CUDA domain
        >>> gpu_ctx = ctx.with_cuda(device_id=0)

        >>> # Split by color
        >>> sub_ctx = ctx.split(color=ctx.rank % 2)
    """

    def __init__(self, comm=None, device_id: int = -1, _native=None) -> None:
        """Create a new DTL context."""
        if _native is not None:
            # Internal: wrap existing native context
            self._ctx = _native
        else:
            self._ctx = _dtl.Context(comm, device_id)

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def rank(self) -> int:
        """Current rank (0 to size-1)."""
        return self._ctx.rank

    @property
    def size(self) -> int:
        """Total number of ranks."""
        return self._ctx.size

    @property
    def is_root(self) -> bool:
        """True if this is rank 0."""
        return self._ctx.is_root

    @property
    def device_id(self) -> int:
        """GPU device ID (-1 for CPU-only)."""
        return self._ctx.device_id

    @property
    def has_device(self) -> bool:
        """True if GPU is enabled."""
        return self._ctx.has_device

    # =========================================================================
    # Domain Query Properties (V1.3.0)
    # =========================================================================

    @property
    def has_mpi(self) -> bool:
        """True if MPI domain is available."""
        return self._ctx.has_mpi

    @property
    def has_cuda(self) -> bool:
        """True if CUDA domain is available."""
        return self._ctx.has_cuda

    @property
    def has_nccl(self) -> bool:
        """True if NCCL domain is available."""
        return self._ctx.has_nccl

    @property
    def has_shmem(self) -> bool:
        """True if SHMEM domain is available."""
        return self._ctx.has_shmem

    @property
    def is_valid(self) -> bool:
        """True if context handle is valid."""
        return self._ctx.is_valid

    # =========================================================================
    # Synchronization Methods
    # =========================================================================

    def barrier(self) -> None:
        """Synchronize all ranks (collective).

        Blocks until all ranks have reached this point.
        """
        self._ctx.barrier()

    def fence(self) -> None:
        """Memory fence (local synchronization).

        Ensures all memory operations are complete before continuing.
        """
        self._ctx.fence()

    # =========================================================================
    # Context Factory Methods (V1.3.0)
    # =========================================================================

    def dup(self) -> "Context":
        """Duplicate context with a new MPI communicator.

        Returns:
            New Context with duplicated communicator.

        Note:
            This is a collective operation - all ranks must call it.
        """
        new_native = self._ctx.dup()
        return Context(_native=new_native)

    def split(self, color: int, key: int = 0) -> "Context":
        """Split context by color to create sub-groups.

        Args:
            color: Color for grouping (ranks with same color form a group)
            key: Ordering key within color group (default: 0)

        Returns:
            New Context with split communicator.

        Note:
            This is a collective operation - all ranks must call it.

        Example:
            >>> # Split into even/odd groups
            >>> sub_ctx = ctx.split(color=ctx.rank % 2)

            >>> # Split into groups of 4
            >>> sub_ctx = ctx.split(color=ctx.rank // 4)
        """
        new_native = self._ctx.split(color, key)
        return Context(_native=new_native)

    def with_cuda(self, device_id: int) -> "Context":
        """Create new context with CUDA domain added.

        Args:
            device_id: CUDA device ID to use

        Returns:
            New Context with CUDA domain enabled.

        Example:
            >>> gpu_ctx = ctx.with_cuda(device_id=0)
            >>> print(gpu_ctx.has_cuda)  # True
        """
        new_native = self._ctx.with_cuda(device_id)
        return Context(_native=new_native)

    def with_nccl(self, device_id: int) -> "Context":
        """Create new context with NCCL domain added.

        Args:
            device_id: CUDA device ID for NCCL communication

        Returns:
            New Context with NCCL domain enabled.

        Note:
            Requires MPI domain to be present (NCCL initialization
            uses MPI for rank coordination).

        Example:
            >>> nccl_ctx = ctx.with_nccl(device_id=0)
            >>> print(nccl_ctx.has_nccl)  # True
        """
        new_native = self._ctx.with_nccl(device_id)
        return Context(_native=new_native)

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass  # Context cleanup handled by _ctx destructor

    def __repr__(self) -> str:
        domains = []
        if self.has_mpi:
            domains.append("mpi")
        if self.has_cuda:
            domains.append("cuda")
        if self.has_nccl:
            domains.append("nccl")
        if self.has_shmem:
            domains.append("shmem")
        domain_str = ",".join(domains) if domains else "cpu"
        return f"Context(rank={self.rank}, size={self.size}, device_id={self.device_id}, domains=[{domain_str}])"

    @property
    def _native(self):
        """Internal: access to native context object."""
        return self._ctx


# =============================================================================
# Environment
# =============================================================================

class Environment:
    """RAII environment for DTL backend lifecycle management.

    Initializes all backends (MPI, CUDA, HIP, NCCL, SHMEM) on first
    creation. Uses reference counting - subsequent creations increment
    the count, and the last destruction finalizes backends.

    This is the recommended way to initialize DTL. Use
    ``make_world_context()`` to get a Context for distributed operations.

    Args:
        None

    Example:
        >>> env = dtl.Environment()
        >>> ctx = env.make_world_context()
        >>> vec = dtl.DistributedVector(ctx, size=1000)

        # As context manager:
        >>> with dtl.Environment() as env:
        ...     ctx = env.make_world_context()
        ...     print(f"Rank {ctx.rank} of {ctx.size}")

    Static queries:
        >>> dtl.Environment.is_initialized()  # True if any env exists
        >>> dtl.Environment.ref_count()       # Number of active handles
    """

    def __init__(self) -> None:
        """Create a new DTL environment handle."""
        self._env = _dtl.Environment()

    # =========================================================================
    # Static Queries
    # =========================================================================

    @staticmethod
    def is_initialized() -> bool:
        """True if at least one environment handle exists."""
        return _dtl.Environment.is_initialized()

    @staticmethod
    def ref_count() -> int:
        """Number of active environment handles."""
        return _dtl.Environment.ref_count()

    # =========================================================================
    # Backend Availability
    # =========================================================================

    @property
    def has_mpi(self) -> bool:
        """True if MPI backend is available."""
        return bool(self._env.has_mpi)

    @property
    def has_cuda(self) -> bool:
        """True if CUDA backend is available."""
        return bool(self._env.has_cuda)

    @property
    def has_hip(self) -> bool:
        """True if HIP backend is available."""
        return bool(self._env.has_hip)

    @property
    def has_nccl(self) -> bool:
        """True if NCCL backend is available."""
        return bool(self._env.has_nccl)

    @property
    def has_shmem(self) -> bool:
        """True if SHMEM backend is available."""
        return bool(self._env.has_shmem)

    @staticmethod
    def mpi_thread_level() -> int:
        """MPI thread support level (0-3), or -1 if not available."""
        env = _dtl.Environment()
        return int(env.mpi_thread_level())

    # =========================================================================
    # Context Factories
    # =========================================================================

    def make_world_context(self, device_id: int | None = None) -> "Context":
        """Create a world context spanning all MPI ranks.

        Args:
            device_id: Optional CUDA device ID. If provided, creates a
                       GPU-enabled context with CUDA domain.

        Returns:
            Context for distributed operations.

        Example:
            >>> ctx = env.make_world_context()
            >>> gpu_ctx = env.make_world_context(device_id=0)
        """
        if device_id is not None:
            native = self._env.make_world_context(device_id=device_id)
        else:
            native = self._env.make_world_context()
        return Context(_native=native)

    def make_cpu_context(self) -> "Context":
        """Create a CPU-only context (single-process, no MPI).

        Returns:
            Context with CPU domain only (rank=0, size=1).

        Example:
            >>> ctx = env.make_cpu_context()
        """
        native = self._env.make_cpu_context()
        return Context(_native=native)

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> "Environment":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass  # Environment cleanup handled by _env destructor

    def __repr__(self) -> str:
        backends = []
        if self.has_mpi:
            backends.append("mpi")
        if self.has_cuda:
            backends.append("cuda")
        if self.has_hip:
            backends.append("hip")
        if self.has_nccl:
            backends.append("nccl")
        if self.has_shmem:
            backends.append("shmem")
        backend_str = ",".join(backends) if backends else "none"
        return f"Environment(backends=[{backend_str}], refcount={self.ref_count()})"


# =============================================================================
# Containers
# =============================================================================

def DistributedVector(
    ctx: Context,
    size: int,
    dtype=None,
    fill=None,
    *,
    partition=None,
    placement=None,
    execution=None,
    device_id: int = 0,
    block_size: int = 1
):
    """Create a distributed vector.

    Factory function that creates a distributed 1D array with the
    appropriate element type based on the dtype parameter.

    Args:
        ctx: Execution context
        size: Global number of elements
        dtype: NumPy dtype (default: np.float64)
        fill: Optional fill value for initialization
        partition: Partition policy (default: PARTITION_BLOCK)
        placement: Placement policy (default: PLACEMENT_HOST)
        execution: Execution policy (default: EXEC_SEQ)
        device_id: GPU device ID for device placements (default: 0)
        block_size: Block size for block-cyclic partition (default: 1)

    Returns:
        A distributed vector object with local_view() and to_numpy() methods.

    Example:
        >>> import numpy as np
        >>> vec = dtl.DistributedVector(ctx, size=1000, dtype=np.float32)
        >>> local = vec.local_view()
        >>> local[:] = 42.0

    With policy parameters (Phase 05):
        >>> vec = dtl.DistributedVector(
        ...     ctx, size=1000,
        ...     partition=dtl.PARTITION_CYCLIC,
        ...     placement=dtl.PLACEMENT_UNIFIED
        ... )

    Note:
        For device-only placement (PLACEMENT_DEVICE), local_view() will
        raise an error. Use to_numpy() to copy data to host instead.
    """
    import numpy as np

    if dtype is None:
        dtype = np.float64

    dtype = np.dtype(dtype)

    # Use defaults if not specified
    if partition is None:
        partition = PARTITION_BLOCK
    if placement is None:
        placement = PLACEMENT_HOST
    if execution is None:
        execution = EXEC_SEQ

    # Validate policy values
    _valid_partitions = {int(PARTITION_BLOCK), int(PARTITION_CYCLIC),
                         int(PARTITION_BLOCK_CYCLIC), int(PARTITION_HASH),
                         int(PARTITION_REPLICATED)}
    _valid_placements = {int(PLACEMENT_HOST), int(PLACEMENT_DEVICE),
                         int(PLACEMENT_UNIFIED), int(PLACEMENT_DEVICE_PREFERRED)}
    _valid_executions = {int(EXEC_SEQ), int(EXEC_PAR), int(EXEC_ASYNC)}

    if int(partition) not in _valid_partitions:
        raise ValueError(f"Invalid partition policy: {partition}")
    if int(placement) not in _valid_placements:
        raise ValueError(f"Invalid placement policy: {placement}")
    if int(execution) not in _valid_executions:
        raise ValueError(f"Invalid execution policy: {execution}")

    # Validate placement availability
    if placement in (PLACEMENT_DEVICE, PLACEMENT_UNIFIED, PLACEMENT_DEVICE_PREFERRED):
        if not placement_available(placement):
            raise ValueError(
                f"Placement policy {placement} is not available in this build. "
                "CUDA support may be required."
            )

    # Select typed implementation based on dtype
    type_map = {
        np.dtype(np.float64): _dtl.DistributedVector_f64,
        np.dtype(np.float32): _dtl.DistributedVector_f32,
        np.dtype(np.int64): _dtl.DistributedVector_i64,
        np.dtype(np.int32): _dtl.DistributedVector_i32,
        np.dtype(np.uint64): _dtl.DistributedVector_u64,
        np.dtype(np.uint32): _dtl.DistributedVector_u32,
        np.dtype(np.uint8): _dtl.DistributedVector_u8,
        np.dtype(np.int8): _dtl.DistributedVector_i8,
    }

    vec_cls = type_map.get(dtype)
    if vec_cls is None:
        raise TypeError(f"Unsupported dtype: {dtype}")

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    # Use policy-aware constructor
    try:
        if fill is not None:
            return vec_cls(native_ctx, size, fill,
                           int(partition), int(placement), int(execution),
                           device_id, block_size)
        return vec_cls(native_ctx, size,
                       int(partition), int(placement), int(execution),
                       device_id, block_size)
    except RuntimeError as exc:
        # Compatibility fallback for builds without block-cyclic instantiations.
        if int(partition) == int(PARTITION_BLOCK_CYCLIC) and "not_supported" in str(exc):
            if fill is not None:
                return vec_cls(native_ctx, size, fill,
                               int(PARTITION_BLOCK), int(placement), int(execution),
                               device_id, 1)
            return vec_cls(native_ctx, size,
                           int(PARTITION_BLOCK), int(placement), int(execution),
                           device_id, 1)
        raise


def DistributedArray(
    ctx: Context,
    size: int,
    dtype=None,
    fill=None,
    *,
    partition=None,
    placement=None,
    execution=None,
    device_id: int = 0,
    block_size: int = 1
):
    """Create a distributed array with fixed size.

    Factory function that creates a distributed 1D array with a fixed size.
    Unlike DistributedVector, arrays cannot be resized after creation.

    Args:
        ctx: Execution context
        size: Global number of elements (fixed, cannot be changed)
        dtype: NumPy dtype (default: np.float64)
        fill: Optional fill value for initialization
        partition: Partition policy (default: PARTITION_BLOCK)
        placement: Placement policy (default: PLACEMENT_HOST)
        execution: Execution policy (default: EXEC_SEQ)
        device_id: GPU device ID for device placements (default: 0)
        block_size: Block size for block-cyclic partition (default: 1)

    Returns:
        A distributed array object with local_view() and to_numpy() methods.

    Example:
        >>> import numpy as np
        >>> arr = dtl.DistributedArray(ctx, size=1000, dtype=np.float32)
        >>> local = arr.local_view()
        >>> local[:] = 42.0

    Note:
        Unlike DistributedVector, DistributedArray does not have a resize()
        method. The size is fixed at creation time.

        For device-only placement (PLACEMENT_DEVICE), local_view() will
        raise an error. Use to_numpy() to copy data to host instead.
    """
    import numpy as np

    if dtype is None:
        dtype = np.float64

    dtype = np.dtype(dtype)

    # Use defaults if not specified
    if partition is None:
        partition = PARTITION_BLOCK
    if placement is None:
        placement = PLACEMENT_HOST
    if execution is None:
        execution = EXEC_SEQ

    # Validate policy values
    _valid_partitions = {int(PARTITION_BLOCK), int(PARTITION_CYCLIC),
                         int(PARTITION_BLOCK_CYCLIC), int(PARTITION_HASH),
                         int(PARTITION_REPLICATED)}
    _valid_placements = {int(PLACEMENT_HOST), int(PLACEMENT_DEVICE),
                         int(PLACEMENT_UNIFIED), int(PLACEMENT_DEVICE_PREFERRED)}
    _valid_executions = {int(EXEC_SEQ), int(EXEC_PAR), int(EXEC_ASYNC)}

    if int(partition) not in _valid_partitions:
        raise ValueError(f"Invalid partition policy: {partition}")
    if int(placement) not in _valid_placements:
        raise ValueError(f"Invalid placement policy: {placement}")
    if int(execution) not in _valid_executions:
        raise ValueError(f"Invalid execution policy: {execution}")

    # Validate placement availability
    if placement in (PLACEMENT_DEVICE, PLACEMENT_UNIFIED, PLACEMENT_DEVICE_PREFERRED):
        if not placement_available(placement):
            raise ValueError(
                f"Placement policy {placement} is not available in this build. "
                "CUDA support may be required."
            )

    # Select typed implementation based on dtype
    type_map = {
        np.dtype(np.float64): _dtl.DistributedArray_f64,
        np.dtype(np.float32): _dtl.DistributedArray_f32,
        np.dtype(np.int64): _dtl.DistributedArray_i64,
        np.dtype(np.int32): _dtl.DistributedArray_i32,
        np.dtype(np.uint64): _dtl.DistributedArray_u64,
        np.dtype(np.uint32): _dtl.DistributedArray_u32,
        np.dtype(np.uint8): _dtl.DistributedArray_u8,
        np.dtype(np.int8): _dtl.DistributedArray_i8,
    }

    arr_cls = type_map.get(dtype)
    if arr_cls is None:
        raise TypeError(f"Unsupported dtype: {dtype}")

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    # Use policy-aware constructor
    if fill is not None:
        return arr_cls(native_ctx, size, fill,
                       int(partition), int(placement), int(execution),
                       device_id, block_size)
    return arr_cls(native_ctx, size,
                   int(partition), int(placement), int(execution),
                   device_id, block_size)


def DistributedTensor(
    ctx: Context,
    shape,
    dtype=None,
    fill=None,
    *,
    partition=None,
    placement=None,
    execution=None,
    device_id: int = 0,
    block_size: int = 1
):
    """Create a distributed tensor.

    Factory function that creates a distributed N-dimensional array.
    The tensor is distributed along the first dimension.

    Args:
        ctx: Execution context
        shape: Tuple of dimension sizes
        dtype: NumPy dtype (default: np.float64)
        fill: Optional fill value for initialization
        partition: Partition policy (default: PARTITION_BLOCK) - reserved for future use
        placement: Placement policy (default: PLACEMENT_HOST) - reserved for future use
        execution: Execution policy (default: EXEC_SEQ) - reserved for future use
        device_id: GPU device ID (default: 0) - reserved for future use
        block_size: Block size (default: 1) - reserved for future use

    Returns:
        A distributed tensor object with local_view() method.

    Example:
        >>> tensor = dtl.DistributedTensor(ctx, shape=(100, 64, 64))
        >>> local = tensor.local_view()
        >>> local[:] = 0.0

    Note:
        Policy parameters are accepted for API consistency but may not be
        fully supported for tensors in this release. Use DistributedVector
        or DistributedArray for full policy support.
    """
    import numpy as np

    if dtype is None:
        dtype = np.float64

    dtype = np.dtype(dtype)

    # Policy params reserved for future tensor support
    _ = (partition, placement, execution, device_id, block_size)

    # Select typed implementation based on dtype
    type_map = {
        np.dtype(np.float64): _dtl.DistributedTensor_f64,
        np.dtype(np.float32): _dtl.DistributedTensor_f32,
        np.dtype(np.int64): _dtl.DistributedTensor_i64,
        np.dtype(np.int32): _dtl.DistributedTensor_i32,
    }

    tensor_cls = type_map.get(dtype)
    if tensor_cls is None:
        raise TypeError(f"Unsupported dtype: {dtype}")

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if fill is not None:
        return tensor_cls(native_ctx, shape, fill)
    return tensor_cls(native_ctx, shape)


def DistributedSpan(source, dtype=None):
    """Create a non-owning distributed span from a distributed container.

    Args:
        source: DistributedVector/DistributedArray/DistributedTensor instance
        dtype: Optional NumPy dtype override. If omitted, inferred from source.

    Returns:
        A distributed span object with local view and subspan operations.

    Notes:
        DistributedSpan is non-owning. The source container must outlive
        the span and any derived subspans/local views.
    """
    import numpy as np

    suffix_map = {
        "_f64": np.dtype(np.float64),
        "_f32": np.dtype(np.float32),
        "_i64": np.dtype(np.int64),
        "_i32": np.dtype(np.int32),
        "_u64": np.dtype(np.uint64),
        "_u32": np.dtype(np.uint32),
        "_u8": np.dtype(np.uint8),
        "_i8": np.dtype(np.int8),
    }

    if dtype is None:
        type_name = type(source).__name__
        inferred = None
        for suffix, np_dtype in suffix_map.items():
            if type_name.endswith(suffix):
                inferred = np_dtype
                break

        if inferred is None:
            if hasattr(source, "local_view"):
                inferred = np.dtype(source.local_view().dtype)
            else:
                raise TypeError(
                    "Unable to infer dtype for DistributedSpan source. "
                    "Pass dtype explicitly."
                )
        dtype = inferred
    else:
        dtype = np.dtype(dtype)

    type_map = {
        np.dtype(np.float64): _dtl.DistributedSpan_f64,
        np.dtype(np.float32): _dtl.DistributedSpan_f32,
        np.dtype(np.int64): _dtl.DistributedSpan_i64,
        np.dtype(np.int32): _dtl.DistributedSpan_i32,
        np.dtype(np.uint64): _dtl.DistributedSpan_u64,
        np.dtype(np.uint32): _dtl.DistributedSpan_u32,
        np.dtype(np.uint8): _dtl.DistributedSpan_u8,
        np.dtype(np.int8): _dtl.DistributedSpan_i8,
    }

    span_cls = type_map.get(dtype)
    if span_cls is None:
        raise TypeError(f"Unsupported dtype: {dtype}")

    return span_cls(source)


# =============================================================================
# Distributed Map (Phase 16)
# =============================================================================

class DistributedMap:
    """Distributed hash map across ranks.

    A hash-based associative container similar to std::unordered_map.
    In v0.1.0-alpha.1 the Python/C bindings operate on the local rank's
    partition only (no cross-rank routing or queuing).

    Args:
        ctx: Execution context
        key_dtype: NumPy dtype for keys (default: np.int64)
        value_dtype: NumPy dtype for values (default: np.float64)

    Example:
        >>> m = dtl.DistributedMap(ctx, key_dtype=np.int64, value_dtype=np.float64)
        >>> m.insert(42, 3.14)
        >>> val = m.find(42)
        >>> m.sync()
        >>> m.destroy()
    """

    # Mapping of numpy dtypes to DTL dtype integer constants
    _DTYPE_MAP = None

    @classmethod
    def _get_dtype_map(cls):
        if cls._DTYPE_MAP is None:
            import numpy as np
            cls._DTYPE_MAP = {
                np.dtype(np.float64): 9,   # DTL_DTYPE_FLOAT64
                np.dtype(np.float32): 8,   # DTL_DTYPE_FLOAT32
                np.dtype(np.int64):   3,   # DTL_DTYPE_INT64
                np.dtype(np.int32):   2,   # DTL_DTYPE_INT32
                np.dtype(np.uint64):  7,   # DTL_DTYPE_UINT64
                np.dtype(np.uint32):  6,   # DTL_DTYPE_UINT32
                np.dtype(np.int8):    0,   # DTL_DTYPE_INT8
                np.dtype(np.uint8):   4,   # DTL_DTYPE_UINT8
            }
        return cls._DTYPE_MAP

    def __init__(self, ctx: Context, key_dtype=None, value_dtype=None) -> None:
        import numpy as np

        self._handle = 0
        self._ctx = ctx
        if key_dtype is None:
            key_dtype = np.int64
        if value_dtype is None:
            value_dtype = np.float64

        self._key_np_dtype = np.dtype(key_dtype)
        self._val_np_dtype = np.dtype(value_dtype)

        dtype_map = self._get_dtype_map()
        self._key_dtl = dtype_map.get(self._key_np_dtype)
        self._val_dtl = dtype_map.get(self._val_np_dtype)

        if self._key_dtl is None:
            raise TypeError(f"Unsupported key dtype: {self._key_np_dtype}")
        if self._val_dtl is None:
            raise TypeError(f"Unsupported value dtype: {self._val_np_dtype}")

        native_ctx = ctx._native if hasattr(ctx, '_native') else ctx
        self._handle = _dtl.map_create(native_ctx, self._key_dtl, self._val_dtl)

    def insert(self, key, value) -> None:
        """Insert a key-value pair into the local rank's partition.

        Args:
            key: The key to insert
            value: The value to associate with the key
        """
        _dtl.map_insert(self._handle, key, value, self._key_dtl, self._val_dtl)

    def find(self, key):
        """Find a value by key (local lookup).

        Args:
            key: The key to look up

        Returns:
            The value if found locally, or None if not present
        """
        return _dtl.map_find(self._handle, key, self._key_dtl, self._val_dtl)

    def erase(self, key) -> bool:
        """Erase a key-value pair.

        Args:
            key: The key to remove

        Returns:
            True if the key was found and removed
        """
        return _dtl.map_erase(self._handle, key, self._key_dtl)

    def contains(self, key) -> bool:
        """Check if a key exists locally.

        Args:
            key: The key to check

        Returns:
            True if the key exists in the local partition
        """
        return _dtl.map_contains(self._handle, key, self._key_dtl)

    @property
    def size(self) -> int:
        """Number of key-value pairs stored locally.

        Alias for local_size, for consistency with dict-like APIs.
        """
        return _dtl.map_local_size(self._handle)

    @property
    def local_size(self) -> int:
        """Number of key-value pairs stored locally."""
        return _dtl.map_local_size(self._handle)

    @property
    def global_size(self) -> int:
        """Number of key-value pairs. In v0.1.0-alpha.1 returns local size (no cross-rank allreduce)."""
        return _dtl.map_global_size(self._handle)

    @property
    def empty(self) -> bool:
        """True if the local partition has no elements."""
        return _dtl.map_local_size(self._handle) == 0

    def local_view(self) -> dict:
        """Return a dict snapshot of the local key-value pairs.

        Iterates the local partition and returns a Python dict
        containing copies of all locally stored key-value pairs.
        This is a snapshot; modifications to the returned dict do
        not affect the distributed map.

        Returns:
            dict: Copy of local key-value pairs.

        Note:
            This operation materializes all local data into a Python
            dict. For large local partitions, consider using find()
            for individual lookups instead.
        """
        if not hasattr(_dtl, "map_local_entries"):
            raise RuntimeError(
                "map_local_entries is unavailable in this build; "
                "rebuild Python bindings with DistributedMap iteration support"
            )
        result = {}
        items = _dtl.map_local_entries(self._handle, self._key_dtl, self._val_dtl)
        for k, v in items:
            result[k] = v
        return result

    def global_view(self) -> dict:
        """Return a dict snapshot of all key-value pairs (collective).

        Gathers all key-value pairs from all ranks and returns a combined
        Python dict. This is a collective operation -- all ranks must call it.

        Returns:
            dict: Combined key-value pairs from all ranks.

        Warning:
            This is a collective operation. All ranks must participate.
            For large maps, this may require significant memory and
            communication. Prefer local_view() or find() when possible.
        """
        import pickle
        import numpy as np

        # Synchronize first so all pending remote ops are committed
        self.sync()
        local = self.local_view()

        if getattr(self._ctx, "size", 1) <= 1:
            return local

        payload = pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL)
        local_buf = np.frombuffer(payload, dtype=np.uint8).copy()

        local_count = np.array([local_buf.size], dtype=np.int64)
        recvcounts = allgather(self._ctx, local_count).flatten().astype(np.int64)
        gathered = allgatherv(self._ctx, local_buf, recvcounts=recvcounts)

        merged = {}
        offset = 0
        for count in recvcounts:
            end = offset + int(count)
            chunk = gathered[offset:end].tobytes()
            offset = end
            if not chunk:
                continue
            merged.update(pickle.loads(chunk))
        return merged

    def flush(self) -> None:
        """Flush pending operations. In v0.1.0-alpha.1 this is a local no-op (clears dirty flag)."""
        _dtl.map_flush(self._handle)

    def sync(self) -> None:
        """Synchronize the map. In v0.1.0-alpha.1 this is a local no-op (clears dirty flag)."""
        _dtl.map_sync(self._handle)

    def clear(self) -> None:
        """Clear all elements from the local partition. In v0.1.0-alpha.1 clears locally only."""
        _dtl.map_clear(self._handle)

    def destroy(self) -> None:
        """Release map resources."""
        if self._handle:
            _dtl.map_destroy(self._handle)
            self._handle = 0

    def __len__(self) -> int:
        return self.local_size

    def __contains__(self, key) -> bool:
        return self.contains(key)

    def __del__(self) -> None:
        if hasattr(self, "_handle"):
            self.destroy()

    def __repr__(self) -> str:
        return (f"<DistributedMap key_dtype={self._key_np_dtype} "
                f"value_dtype={self._val_np_dtype} "
                f"local_size={self.local_size}>")


# =============================================================================
# Device Migration (Phase 28)
# =============================================================================


def _detect_container_type(container):
    """Detect whether a container is a vector or array and its dtype suffix.

    Returns:
        Tuple of (container_kind, dtype_suffix) where container_kind is
        "vector", "array", or "tensor" and dtype_suffix is e.g. "f64".
    """
    cls_name = type(container).__name__
    if cls_name.startswith("DistributedVector_"):
        return ("vector", cls_name[len("DistributedVector_"):])
    elif cls_name.startswith("DistributedArray_"):
        return ("array", cls_name[len("DistributedArray_"):])
    elif cls_name.startswith("DistributedTensor_"):
        return ("tensor", cls_name[len("DistributedTensor_"):])
    else:
        raise TypeError(
            f"Cannot determine container type from {cls_name}. "
            "Expected DistributedVector, DistributedArray, or DistributedTensor."
        )


def _get_constructor(kind: str, suffix: str):
    """Get the native constructor for a given container kind and dtype suffix."""
    name = {
        "vector": f"DistributedVector_{suffix}",
        "array": f"DistributedArray_{suffix}",
        "tensor": f"DistributedTensor_{suffix}",
    }.get(kind)
    if name is None:
        raise TypeError(f"Unknown container kind: {kind}")
    cls = getattr(_dtl, name, None)
    if cls is None:
        raise TypeError(f"Native type {name} not available in this build.")
    return cls


def to_device(ctx: Context, container, device_id: int = 0):
    """Copy a container's data to a new device-placed container.

    Creates a new container with PLACEMENT_DEVICE and copies the data
    from the source container through host staging.

    Args:
        ctx: Execution context (must have CUDA domain)
        container: Source container (DistributedVector or DistributedArray)
        device_id: Target GPU device ID (default: 0)

    Returns:
        A new container with device placement containing the same data.

    Raises:
        ValueError: If PLACEMENT_DEVICE is not available
        TypeError: If the container type is not supported

    Example:
        >>> vec = dtl.DistributedVector(ctx, size=100, fill=1.0)
        >>> device_vec = dtl.to_device(ctx, vec, device_id=0)
    """
    if not placement_available(PLACEMENT_DEVICE):
        raise ValueError(
            "PLACEMENT_DEVICE is not available in this build. "
            "CUDA support may be required."
        )

    kind, suffix = _detect_container_type(container)

    if kind == "tensor":
        raise TypeError(
            "to_device() is not yet supported for DistributedTensor. "
            "Use DistributedTensor with placement=PLACEMENT_DEVICE instead."
        )

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    # Copy data to host first
    host_data = container.to_numpy()

    # Create new container with device placement
    cls = _get_constructor(kind, suffix)
    new_container = cls(
        native_ctx, container.global_size,
        int(PARTITION_BLOCK), int(PLACEMENT_DEVICE), int(EXEC_SEQ),
        device_id, 1
    )

    # Copy data to device container
    new_container.from_numpy(host_data)

    return new_container


def to_host(ctx: Context, container):
    """Copy a container's data to a new host-placed container.

    Creates a new container with PLACEMENT_HOST and copies the data
    from the source container through host staging.

    Args:
        ctx: Execution context
        container: Source container (DistributedVector or DistributedArray)

    Returns:
        A new container with host placement containing the same data.

    Example:
        >>> device_vec = dtl.DistributedVector(
        ...     ctx, size=100, fill=1.0, placement=dtl.PLACEMENT_DEVICE)
        >>> host_vec = dtl.to_host(ctx, device_vec)
        >>> local = host_vec.local_view()  # Now accessible on host
    """
    kind, suffix = _detect_container_type(container)

    if kind == "tensor":
        raise TypeError(
            "to_host() is not yet supported for DistributedTensor. "
            "Use DistributedTensor with placement=PLACEMENT_HOST instead."
        )

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    # Copy data to host (works regardless of source placement)
    host_data = container.to_numpy()

    # Create new container with host placement
    cls = _get_constructor(kind, suffix)
    new_container = cls(
        native_ctx, container.global_size,
        int(PARTITION_BLOCK), int(PLACEMENT_HOST), int(EXEC_SEQ),
        0, 1
    )

    # Copy data into host container
    new_container.from_numpy(host_data)

    return new_container


# =============================================================================
# Collective Operations
# =============================================================================

# Reduce operation constants
SUM = "sum"
PROD = "prod"
MIN = "min"
MAX = "max"


def allreduce(ctx: Context, data, op: str = "sum"):
    """Reduce data from all ranks and distribute result to all.

    Combines elements from all ranks using the specified operation
    and places the result on all ranks.

    Args:
        ctx: Execution context
        data: NumPy array or scalar to reduce
        op: Reduction operation ("sum", "prod", "min", "max")

    Returns:
        Reduced result (same type as input)

    Example:
        >>> local_sum = np.sum(local_data)
        >>> global_sum = dtl.allreduce(ctx, np.array([local_sum]), op="sum")
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    # Convert scalar to array
    is_scalar = np.isscalar(data)
    if is_scalar:
        data = np.array([data])
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Ensure array is contiguous and writeable
    data = np.ascontiguousarray(data)

    result = _dtl.allreduce(native_ctx, data, op)

    # Return scalar if input was scalar
    if is_scalar and np.isscalar(result):
        return result
    elif is_scalar:
        return result.item() if hasattr(result, 'item') else result
    return result


def reduce(ctx: Context, data, op: str = "sum", root: int = 0):
    """Reduce data from all ranks to root.

    Combines elements from all ranks using the specified operation
    and places the result on the root rank only.

    Args:
        ctx: Execution context
        data: NumPy array or scalar to reduce
        op: Reduction operation ("sum", "prod", "min", "max")
        root: Root rank that receives the result (default: 0)

    Returns:
        Reduced result on root, undefined on other ranks
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    is_scalar = np.isscalar(data)
    if is_scalar:
        data = np.array([data])
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Ensure array is contiguous
    data = np.ascontiguousarray(data)

    result = _dtl.reduce(native_ctx, data, op, root)

    if is_scalar and np.isscalar(result):
        return result
    elif is_scalar:
        return result.item() if hasattr(result, 'item') else result
    return result


def broadcast(ctx: Context, data, root: int = 0):
    """Broadcast data from root to all ranks.

    The root rank's data is sent to all other ranks.

    Args:
        ctx: Execution context
        data: NumPy array (send on root, receive on others)
        root: Root rank that broadcasts (default: 0)

    Returns:
        The broadcasted data
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Ensure array is contiguous and writeable (broadcast modifies in place)
    data = np.ascontiguousarray(data).copy()

    return _dtl.broadcast(native_ctx, data, root)


def gather(ctx: Context, data, root: int = 0):
    """Gather data from all ranks to root.

    Each rank sends its data to the root, which collects all data.

    Args:
        ctx: Execution context
        data: NumPy array to send
        root: Root rank that gathers (default: 0)

    Returns:
        On root: array with shape (size, *data.shape)
        On others: empty array
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Ensure array is contiguous
    data = np.ascontiguousarray(data)

    return _dtl.gather(native_ctx, data, root)


def scatter(ctx: Context, data, root: int = 0):
    """Scatter data from root to all ranks.

    Root distributes portions of its data to each rank.

    Args:
        ctx: Execution context
        data: NumPy array to scatter (first dimension = number of ranks)
        root: Root rank that scatters (default: 0)

    Returns:
        One chunk of the scattered data
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Ensure array is contiguous
    data = np.ascontiguousarray(data)

    return _dtl.scatter(native_ctx, data, root)


def allgather(ctx: Context, data):
    """Gather data from all ranks to all ranks.

    Each rank sends its data and receives data from all ranks.

    Args:
        ctx: Execution context
        data: NumPy array to send

    Returns:
        Array with shape (size, *data.shape) containing all ranks' data
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Ensure array is contiguous
    data = np.ascontiguousarray(data)

    return _dtl.allgather(native_ctx, data)


def allgatherv(ctx: Context, data, recvcounts=None):
    """Variable-count gather to all ranks.

    Each rank contributes a potentially different number of elements.
    All ranks receive all data.

    Args:
        ctx: Execution context
        data: NumPy array with local data to send
        recvcounts: Array of receive counts per rank. If None, an allgather
                    is performed to exchange counts automatically.

    Returns:
        NumPy array containing gathered data from all ranks

    Example:
        >>> local = np.array([1.0, 2.0])  # rank 0 sends 2
        >>> result = dtl.allgatherv(ctx, local)
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    data = np.ascontiguousarray(data)

    size = ctx.size

    # If recvcounts not provided, exchange counts via allgather
    if recvcounts is None:
        local_count = np.array([len(data)], dtype=np.int64)
        all_counts = _dtl.allgather(native_ctx, local_count)
        recvcounts = np.ascontiguousarray(all_counts.flatten().astype(np.int64))
    else:
        recvcounts = np.ascontiguousarray(np.asarray(recvcounts, dtype=np.int64))

    # Compute displacements from recvcounts
    displs = np.zeros(len(recvcounts), dtype=np.int64)
    for i in range(1, len(recvcounts)):
        displs[i] = displs[i - 1] + recvcounts[i - 1]

    total_recv = int(np.sum(recvcounts))
    recvbuf = np.empty(total_recv, dtype=data.dtype)

    return _dtl.allgatherv(native_ctx, data, recvcounts, displs, recvbuf)


def alltoallv(ctx: Context, data, sendcounts, recvcounts):
    """Variable-count all-to-all exchange.

    Each rank sends a potentially different amount of data to every other rank.

    Args:
        ctx: Execution context
        data: NumPy array with data to send (concatenated for all ranks)
        sendcounts: Array of send counts (one per rank)
        recvcounts: Array of receive counts (one per rank)

    Returns:
        NumPy array containing received data from all ranks

    Example:
        >>> sendcounts = np.array([2, 3])  # send 2 to rank 0, 3 to rank 1
        >>> recvcounts = np.array([1, 2])  # recv 1 from rank 0, 2 from rank 1
        >>> sendbuf = np.arange(5, dtype=np.float64)
        >>> result = dtl.alltoallv(ctx, sendbuf, sendcounts, recvcounts)
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    data = np.ascontiguousarray(data)
    sendcounts = np.ascontiguousarray(np.asarray(sendcounts, dtype=np.int64))
    recvcounts = np.ascontiguousarray(np.asarray(recvcounts, dtype=np.int64))

    # Compute send displacements
    sdispls = np.zeros(len(sendcounts), dtype=np.int64)
    for i in range(1, len(sendcounts)):
        sdispls[i] = sdispls[i - 1] + sendcounts[i - 1]

    # Compute receive displacements
    rdispls = np.zeros(len(recvcounts), dtype=np.int64)
    for i in range(1, len(recvcounts)):
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1]

    total_recv = int(np.sum(recvcounts))
    recvbuf = np.empty(total_recv, dtype=data.dtype)

    return _dtl.alltoallv(native_ctx, data, sendcounts, sdispls,
                          recvbuf, recvcounts, rdispls)


def send(ctx: Context, data, dest: int, tag: int = 0):
    """Send data to a destination rank.

    Blocking send operation. The call returns when the send buffer
    can safely be reused.

    Args:
        ctx: Execution context
        data: NumPy array or scalar to send
        dest: Destination rank (0 to size-1)
        tag: Message tag (default: 0)

    Example:
        >>> if ctx.rank == 0:
        ...     dtl.send(ctx, np.array([1.0, 2.0, 3.0]), dest=1)
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    data = np.ascontiguousarray(data)

    _dtl.send(native_ctx, data, dest, tag)


def recv(ctx: Context, count: int, dtype, source: int, tag: int = 0):
    """Receive data from a source rank.

    Blocking receive operation. The call blocks until the message
    is received.

    Args:
        ctx: Execution context
        count: Number of elements to receive
        dtype: NumPy dtype of the elements
        source: Source rank (0 to size-1)
        tag: Message tag (default: 0)

    Returns:
        NumPy array containing the received data

    Example:
        >>> if ctx.rank == 1:
        ...     data = dtl.recv(ctx, count=3, dtype=np.float64, source=0)
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    dtype = np.dtype(dtype)
    buf = np.empty(count, dtype=dtype)

    return _dtl.recv(native_ctx, buf, source, tag)


def sendrecv(ctx: Context, senddata, dest: int, recvcount: int,
             recvdtype=None, source=None, sendtag: int = 0, recvtag: int = 0):
    """Send and receive data simultaneously.

    Performs a combined send and receive operation. Useful for
    exchanging data between pairs of ranks.

    Args:
        ctx: Execution context
        senddata: NumPy array or scalar to send
        dest: Destination rank
        recvcount: Number of elements to receive
        recvdtype: NumPy dtype for received data (default: same as senddata)
        source: Source rank (default: same as dest)
        sendtag: Send message tag (default: 0)
        recvtag: Receive message tag (default: 0)

    Returns:
        NumPy array containing the received data

    Example:
        >>> # Exchange data with neighbor
        >>> received = dtl.sendrecv(ctx, local_data, dest=neighbor,
        ...                         recvcount=len(local_data),
        ...                         source=neighbor)
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(senddata, np.ndarray):
        senddata = np.asarray(senddata)

    senddata = np.ascontiguousarray(senddata)

    if recvdtype is None:
        recvdtype = senddata.dtype
    else:
        recvdtype = np.dtype(recvdtype)

    if source is None:
        source = dest

    recvbuf = np.empty(recvcount, dtype=recvdtype)

    return _dtl.sendrecv(native_ctx, senddata, dest, sendtag,
                         recvbuf, source, recvtag)


def probe(ctx: Context, source: int = -2, tag: int = -1) -> dict:
    """Blocking probe for incoming messages.

    Waits until a matching message is available, then returns message
    metadata without consuming the message.

    Args:
        ctx: DTL context
        source: Source rank to probe (-2 = any source, default)
        tag: Message tag to match (-1 = any tag, default)

    Returns:
        dict with keys 'source', 'tag', 'count' describing the message

    Example:
        >>> info = dtl.probe(ctx, source=0)
        >>> data = dtl.recv(ctx, count=info['count'], dtype=np.float64, source=info['source'])
    """
    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx
    return _dtl.probe(native_ctx, source, tag)


def iprobe(ctx: Context, source: int = -2, tag: int = -1):
    """Non-blocking probe for incoming messages.

    Checks if a matching message is available without blocking.

    Args:
        ctx: DTL context
        source: Source rank to probe (-2 = any source, default)
        tag: Message tag to match (-1 = any tag, default)

    Returns:
        tuple(bool, dict_or_None): (flag, info) where flag is True if a
        message is available, and info is a dict with 'source', 'tag',
        'count' keys (or None if no message available).

    Example:
        >>> found, info = dtl.iprobe(ctx, source=0)
        >>> if found:
        ...     data = dtl.recv(ctx, count=info['count'], dtype=np.float64, source=info['source'])
    """
    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx
    return _dtl.iprobe(native_ctx, source, tag)


def gatherv(ctx: Context, data, recvcounts=None, root: int = 0):
    """Variable-count gather to root.

    Each rank sends a potentially different number of elements to the root.

    Args:
        ctx: DTL context
        data: NumPy array with local data to send
        recvcounts: Array of receive counts per rank (only needed on root).
            If None, counts are gathered automatically via allgather.
        root: Root rank that gathers (default: 0)

    Returns:
        On root: NumPy array containing all gathered data.
        On non-root: empty NumPy array.

    Example:
        >>> local = np.array([1.0, 2.0])  # each rank sends 2 elements
        >>> result = dtl.gatherv(ctx, local, root=0)
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    data = np.ascontiguousarray(data)

    local_count = np.array([data.size], dtype=np.int64)

    if recvcounts is None:
        all_counts = _dtl.allgather(native_ctx, local_count)
        recvcounts = np.ascontiguousarray(all_counts.flatten().astype(np.int64))
    else:
        recvcounts = np.ascontiguousarray(np.asarray(recvcounts, dtype=np.int64))

    displs = np.zeros(len(recvcounts), dtype=np.int64)
    for i in range(1, len(recvcounts)):
        displs[i] = displs[i - 1] + recvcounts[i - 1]

    if ctx.rank == root:
        total_recv = int(np.sum(recvcounts))
        recvbuf = np.empty(total_recv, dtype=data.dtype)
    else:
        recvbuf = np.empty(0, dtype=data.dtype)

    return _dtl.gatherv(native_ctx, data, recvcounts, displs, recvbuf, root)


def scatterv(ctx: Context, data, sendcounts, root: int = 0):
    """Variable-count scatter from root.

    Root sends a potentially different number of elements to each rank.

    Args:
        ctx: DTL context
        data: NumPy array with data to scatter (only meaningful on root)
        sendcounts: Array of send counts (one per rank)
        root: Root rank that scatters (default: 0)

    Returns:
        NumPy array containing the received chunk for this rank.

    Example:
        >>> sendcounts = np.array([2, 3])  # 2 to rank 0, 3 to rank 1
        >>> data = np.arange(5, dtype=np.float64)
        >>> chunk = dtl.scatterv(ctx, data, sendcounts, root=0)
    """
    import numpy as np

    native_ctx = ctx._native if hasattr(ctx, '_native') else ctx

    sendcounts = np.ascontiguousarray(np.asarray(sendcounts, dtype=np.int64))

    displs = np.zeros(len(sendcounts), dtype=np.int64)
    for i in range(1, len(sendcounts)):
        displs[i] = displs[i - 1] + sendcounts[i - 1]

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    data = np.ascontiguousarray(data)

    my_count = int(sendcounts[ctx.rank])
    recvbuf = np.empty(my_count, dtype=data.dtype)

    return _dtl.scatterv(native_ctx, data, sendcounts, displs, recvbuf, root)


# =============================================================================
# Algorithm Operations
# =============================================================================

def for_each_vector(vec, func, with_index: bool = False) -> None:
    """Apply a function to each local element of a vector.

    Args:
        vec: DistributedVector to iterate over
        func: Function to apply. If with_index=False, called as func(value).
              If with_index=True, called as func(value, index).
        with_index: Whether to pass index to function (default: False)

    Example:
        >>> dtl.for_each_vector(vec, lambda x: print(x))
        >>> dtl.for_each_vector(vec, lambda x, i: print(f"[{i}] = {x}"), with_index=True)
    """
    _dtl.for_each_vector(vec, func, with_index)


def for_each_array(arr, func, with_index: bool = False) -> None:
    """Apply a function to each local element of an array.

    Args:
        arr: DistributedArray to iterate over
        func: Function to apply
        with_index: Whether to pass index to function (default: False)
    """
    _dtl.for_each_array(arr, func, with_index)


def copy_vector(src, dst) -> None:
    """Copy local data from source vector to destination vector.

    Both vectors must have the same dtype and local size.

    Args:
        src: Source DistributedVector
        dst: Destination DistributedVector
    """
    _dtl.copy_vector(src, dst)


def copy_array(src, dst) -> None:
    """Copy local data from source array to destination array.

    Args:
        src: Source DistributedArray
        dst: Destination DistributedArray
    """
    _dtl.copy_array(src, dst)


def fill_vector(vec, value) -> None:
    """Fill all local elements of a vector with a value.

    Args:
        vec: DistributedVector to fill
        value: Value to fill with

    Example:
        >>> dtl.fill_vector(vec, 42.0)
    """
    _dtl.fill_vector(vec, value)


def fill_array(arr, value) -> None:
    """Fill all local elements of an array with a value.

    Args:
        arr: DistributedArray to fill
        value: Value to fill with
    """
    _dtl.fill_array(arr, value)


def find_vector(vec, value):
    """Find the first occurrence of a value in the local partition.

    Args:
        vec: DistributedVector to search
        value: Value to find

    Returns:
        Local index of first match, or None if not found
    """
    return _dtl.find_vector(vec, value)


def find_if_vector(vec, predicate):
    """Find the first element satisfying a predicate in the local partition.

    Args:
        vec: DistributedVector to search
        predicate: Function that returns True for matching elements

    Returns:
        Local index of first match, or None if not found

    Example:
        >>> idx = dtl.find_if_vector(vec, lambda x: x > 10)
    """
    return _dtl.find_if_vector(vec, predicate)


def find_array(arr, value):
    """Find the first occurrence of a value in the local partition.

    Args:
        arr: DistributedArray to search
        value: Value to find

    Returns:
        Local index of first match, or None if not found
    """
    return _dtl.find_array(arr, value)


def find_if_array(arr, predicate):
    """Find the first element satisfying a predicate in the local partition.

    Args:
        arr: DistributedArray to search
        predicate: Function that returns True for matching elements

    Returns:
        Local index of first match, or None if not found
    """
    return _dtl.find_if_array(arr, predicate)


def count_vector(vec, value) -> int:
    """Count occurrences of a value in the local partition.

    Args:
        vec: DistributedVector to search
        value: Value to count

    Returns:
        Number of matching elements
    """
    return _dtl.count_vector(vec, value)


def count_if_vector(vec, predicate) -> int:
    """Count elements satisfying a predicate in the local partition.

    Args:
        vec: DistributedVector to search
        predicate: Function that returns True for matching elements

    Returns:
        Number of matching elements

    Example:
        >>> n = dtl.count_if_vector(vec, lambda x: x > 0)
    """
    return _dtl.count_if_vector(vec, predicate)


def count_array(arr, value) -> int:
    """Count occurrences of a value in the local partition.

    Args:
        arr: DistributedArray to search
        value: Value to count

    Returns:
        Number of matching elements
    """
    import numpy as np

    # Work around dtype-dispatch gaps in some native builds.
    local = arr.local_view()
    return int(np.count_nonzero(local == value))


def count_if_array(arr, predicate) -> int:
    """Count elements satisfying a predicate in the local partition.

    Args:
        arr: DistributedArray to search
        predicate: Function that returns True for matching elements

    Returns:
        Number of matching elements
    """
    return _dtl.count_if_array(arr, predicate)


def reduce_local_vector(vec, op: str = "sum"):
    """Reduce local elements of a vector using a built-in operation.

    For distributed reduction (across all ranks), use allreduce() or reduce().

    Args:
        vec: DistributedVector to reduce
        op: Reduction operation ("sum", "prod", "min", "max")

    Returns:
        Reduced value for local partition

    Example:
        >>> local_sum = dtl.reduce_local_vector(vec, op="sum")
        >>> global_sum = dtl.allreduce(ctx, np.array([local_sum]), op="sum")
    """
    return _dtl.reduce_local_vector(vec, op)


def reduce_local_array(arr, op: str = "sum"):
    """Reduce local elements of an array using a built-in operation.

    Args:
        arr: DistributedArray to reduce
        op: Reduction operation ("sum", "prod", "min", "max")

    Returns:
        Reduced value for local partition
    """
    return _dtl.reduce_local_array(arr, op)


def sort_vector(vec, reverse: bool = False) -> None:
    """Sort local elements of a vector.

    This is a local operation - only the local partition is sorted.

    Args:
        vec: DistributedVector to sort
        reverse: If True, sort descending (default: False)
    """
    _dtl.sort_vector(vec, reverse)


def sort_array(arr, reverse: bool = False) -> None:
    """Sort local elements of an array.

    This is a local operation - only the local partition is sorted.

    Args:
        arr: DistributedArray to sort
        reverse: If True, sort descending (default: False)
    """
    _dtl.sort_array(arr, reverse)


def minmax_vector(vec):
    """Find minimum and maximum values in local vector.

    Args:
        vec: DistributedVector to search

    Returns:
        Tuple of (min_value, max_value)

    Example:
        >>> min_val, max_val = dtl.minmax_vector(vec)
    """
    return _dtl.minmax_vector(vec)


def minmax_array(arr):
    """Find minimum and maximum values in local array.

    Args:
        arr: DistributedArray to search

    Returns:
        Tuple of (min_value, max_value)
    """
    return _dtl.minmax_array(arr)


def transform_vector(vec, func) -> None:
    """Apply a transformation function to each element of a distributed vector.

    The function is called with each element's value and should return
    the transformed value. The vector is modified in-place.

    Args:
        vec: DistributedVector to transform
        func: Function that takes a value and returns the transformed value

    Example:
        >>> dtl.transform_vector(vec, lambda x: x * 2)
        >>> dtl.transform_vector(vec, lambda x: x ** 2 + 1)
    """
    _dtl.transform_vector(vec, func)


def transform_array(arr, func) -> None:
    """Apply a transformation function to each element of a distributed array.

    The function is called with each element's value and should return
    the transformed value. The array is modified in-place.

    Args:
        arr: DistributedArray to transform
        func: Function that takes a value and returns the transformed value

    Example:
        >>> dtl.transform_array(arr, lambda x: x * 2)
    """
    _dtl.transform_array(arr, func)


def inclusive_scan_vector(vec, op: str = "sum") -> None:
    """Compute inclusive prefix scan of a distributed vector.

    Each element i is replaced by the reduction of elements 0..i.
    The vector is modified in-place.

    Args:
        vec: DistributedVector to scan
        op: Reduction operation ("sum", "prod", "min", "max")

    Example:
        >>> # [1, 2, 3, 4] -> [1, 3, 6, 10] with op="sum"
        >>> dtl.inclusive_scan_vector(vec, op="sum")
    """
    _dtl.inclusive_scan_vector(vec, op)


def exclusive_scan_vector(vec, op: str = "sum") -> None:
    """Compute exclusive prefix scan of a distributed vector.

    Each element i is replaced by the reduction of elements 0..i-1.
    The first element is set to the identity for the operation
    (0 for sum, 1 for product). The vector is modified in-place.

    Args:
        vec: DistributedVector to scan
        op: Reduction operation ("sum", "prod", "min", "max")

    Example:
        >>> # [1, 2, 3, 4] -> [0, 1, 3, 6] with op="sum"
        >>> dtl.exclusive_scan_vector(vec, op="sum")
    """
    _dtl.exclusive_scan_vector(vec, op)


def inclusive_scan_array(arr, op: str = "sum") -> None:
    """Compute inclusive prefix scan of a distributed array.

    Each element i is replaced by the reduction of elements 0..i.
    The array is modified in-place.

    Args:
        arr: DistributedArray to scan
        op: Reduction operation ("sum", "prod", "min", "max")
    """
    _dtl.inclusive_scan_array(arr, op)


def exclusive_scan_array(arr, op: str = "sum") -> None:
    """Compute exclusive prefix scan of a distributed array.

    Each element i is replaced by the reduction of elements 0..i-1.
    The first element is set to the identity for the operation.
    The array is modified in-place.

    Args:
        arr: DistributedArray to scan
        op: Reduction operation ("sum", "prod", "min", "max")
    """
    _dtl.exclusive_scan_array(arr, op)


# =============================================================================
# Predicate Query Operations (Phase 16)
# =============================================================================

def all_of_vector(vec, predicate) -> bool:
    """Check if all local elements satisfy a predicate.

    Args:
        vec: DistributedVector to check
        predicate: Function that returns True/False for each element

    Returns:
        True if all local elements satisfy the predicate (or vector is empty)

    Example:
        >>> all_positive = dtl.all_of_vector(vec, lambda x: x > 0)
    """
    return _dtl.all_of_vector(vec, predicate)


def any_of_vector(vec, predicate) -> bool:
    """Check if any local element satisfies a predicate.

    Args:
        vec: DistributedVector to check
        predicate: Function that returns True/False for each element

    Returns:
        True if any local element satisfies the predicate

    Example:
        >>> has_negative = dtl.any_of_vector(vec, lambda x: x < 0)
    """
    return _dtl.any_of_vector(vec, predicate)


def none_of_vector(vec, predicate) -> bool:
    """Check if no local elements satisfy a predicate.

    Args:
        vec: DistributedVector to check
        predicate: Function that returns True/False for each element

    Returns:
        True if no local elements satisfy the predicate (or vector is empty)

    Example:
        >>> no_nans = dtl.none_of_vector(vec, lambda x: x != x)
    """
    return _dtl.none_of_vector(vec, predicate)


def all_of_array(arr, predicate) -> bool:
    """Check if all local elements satisfy a predicate.

    Args:
        arr: DistributedArray to check
        predicate: Function that returns True/False for each element

    Returns:
        True if all local elements satisfy the predicate (or array is empty)
    """
    return _dtl.all_of_array(arr, predicate)


def any_of_array(arr, predicate) -> bool:
    """Check if any local element satisfies a predicate.

    Args:
        arr: DistributedArray to check
        predicate: Function that returns True/False for each element

    Returns:
        True if any local element satisfies the predicate
    """
    return _dtl.any_of_array(arr, predicate)


def none_of_array(arr, predicate) -> bool:
    """Check if no local elements satisfy a predicate.

    Args:
        arr: DistributedArray to check
        predicate: Function that returns True/False for each element

    Returns:
        True if no local elements satisfy the predicate (or array is empty)
    """
    return _dtl.none_of_array(arr, predicate)


def min_element_vector(vec):
    """Find the index of the minimum element in the local partition.

    Args:
        vec: DistributedVector to search

    Returns:
        Local index of the minimum element, or None if empty

    Example:
        >>> idx = dtl.min_element_vector(vec)
    """
    return _dtl.min_element_vector(vec)


def max_element_vector(vec):
    """Find the index of the maximum element in the local partition.

    Args:
        vec: DistributedVector to search

    Returns:
        Local index of the maximum element, or None if empty

    Example:
        >>> idx = dtl.max_element_vector(vec)
    """
    return _dtl.max_element_vector(vec)


def min_element_array(arr):
    """Find the index of the minimum element in the local partition.

    Args:
        arr: DistributedArray to search

    Returns:
        Local index of the minimum element, or None if empty
    """
    return _dtl.min_element_array(arr)


def max_element_array(arr):
    """Find the index of the maximum element in the local partition.

    Args:
        arr: DistributedArray to search

    Returns:
        Local index of the maximum element, or None if empty
    """
    return _dtl.max_element_array(arr)


# =============================================================================
# Async Algorithm Operations (Phase 28)
# =============================================================================

class AlgorithmFuture:
    """Future-like object returned by async algorithm operations.

    Wraps an asynchronous algorithm operation, allowing non-blocking
    execution. Call wait() to block until completion, or get() to
    retrieve the result (blocking if not yet complete).

    This is a Python-level wrapper that uses threading to provide
    async behavior for algorithms that do not have native async
    C ABI support.

    Example:
        >>> fut = dtl.async_reduce(vec, op="sum")
        >>> # ... do other work ...
        >>> result = fut.get()
    """

    def __init__(self, func, args=(), kwargs=None) -> None:
        """Create an AlgorithmFuture that will execute func(*args, **kwargs).

        Args:
            func: Callable to execute asynchronously
            args: Positional arguments for func
            kwargs: Keyword arguments for func
        """
        import threading

        if kwargs is None:
            kwargs = {}
        self._result = None
        self._exception = None
        self._done = threading.Event()

        def _run():
            try:
                self._result = func(*args, **kwargs)
            except Exception as e:
                self._exception = e
            finally:
                self._done.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def wait(self) -> None:
        """Block until the async operation completes.

        If the operation has already completed, returns immediately.
        """
        self._done.wait()
        self._thread.join()

    def get(self):
        """Get the result of the async operation, blocking if needed.

        Returns:
            The result of the algorithm operation.

        Raises:
            Exception: Re-raises any exception that occurred during execution.
        """
        self.wait()
        if self._exception is not None:
            raise self._exception
        return self._result

    def done(self) -> bool:
        """Check if the operation has completed (non-blocking).

        Returns:
            True if the operation has completed.
        """
        return self._done.is_set()

    def __repr__(self) -> str:
        status = "done" if self.done() else "pending"
        return f"<AlgorithmFuture ({status})>"


def async_for_each(vec_or_arr, func, with_index: bool = False) -> AlgorithmFuture:
    """Asynchronously apply a function to each local element.

    Returns an AlgorithmFuture that completes when the for_each
    operation has finished processing all local elements.

    Args:
        vec_or_arr: DistributedVector or DistributedArray
        func: Function to apply to each element
        with_index: Whether to pass index to function (default: False)

    Returns:
        AlgorithmFuture that can be waited on.

    Example:
        >>> values = []
        >>> fut = dtl.async_for_each(vec, lambda x: values.append(x))
        >>> fut.wait()
        >>> print(len(values))
    """
    kind, _ = _detect_container_type(vec_or_arr)

    if kind == "vector":
        def _run():
            _dtl.for_each_vector(vec_or_arr, func, with_index)
        return AlgorithmFuture(_run)
    if kind == "array":
        def _run():
            local = vec_or_arr.local_view()
            if with_index:
                for i, value in enumerate(local):
                    func(value, i)
            else:
                for value in local:
                    func(value)
        return AlgorithmFuture(_run)
    raise TypeError("async_for_each currently supports vectors and arrays")


def async_transform(vec_or_arr, func) -> AlgorithmFuture:
    """Asynchronously apply a transformation to each local element.

    Returns an AlgorithmFuture that completes when the transform
    operation has finished. The container is modified in-place.

    Args:
        vec_or_arr: DistributedVector or DistributedArray
        func: Function that takes a value and returns the transformed value

    Returns:
        AlgorithmFuture that can be waited on.

    Example:
        >>> fut = dtl.async_transform(vec, lambda x: x * 2)
        >>> fut.wait()
    """
    kind, _ = _detect_container_type(vec_or_arr)

    if kind == "vector":
        def _run():
            _dtl.transform_vector(vec_or_arr, func)
        return AlgorithmFuture(_run)
    if kind == "array":
        def _run():
            local = vec_or_arr.local_view()
            for i in range(len(local)):
                local[i] = func(local[i])
        return AlgorithmFuture(_run)
    raise TypeError("async_transform currently supports vectors and arrays")


def async_reduce(vec_or_arr, op: str = "sum") -> AlgorithmFuture:
    """Asynchronously reduce local elements of a container.

    Returns an AlgorithmFuture whose get() method returns the
    reduced value.

    Args:
        vec_or_arr: DistributedVector or DistributedArray
        op: Reduction operation ("sum", "prod", "min", "max")

    Returns:
        AlgorithmFuture whose get() returns the reduced value.

    Example:
        >>> fut = dtl.async_reduce(vec, op="sum")
        >>> local_sum = fut.get()
    """
    is_vector = hasattr(vec_or_arr, 'local_offset')

    if is_vector:
        def _run():
            return _dtl.reduce_local_vector(vec_or_arr, op)
        return AlgorithmFuture(_run)
    else:
        def _run():
            return _dtl.reduce_local_array(vec_or_arr, op)
        return AlgorithmFuture(_run)


def async_sort(vec_or_arr, reverse: bool = False) -> AlgorithmFuture:
    """Asynchronously sort local elements of a container.

    Returns an AlgorithmFuture that completes when the sort
    has finished. The container is modified in-place.

    Args:
        vec_or_arr: DistributedVector or DistributedArray
        reverse: If True, sort descending (default: False)

    Returns:
        AlgorithmFuture that can be waited on.

    Example:
        >>> fut = dtl.async_sort(vec)
        >>> fut.wait()
    """
    is_vector = hasattr(vec_or_arr, 'local_offset')

    if is_vector:
        def _run():
            _dtl.sort_vector(vec_or_arr, reverse)
        return AlgorithmFuture(_run)
    else:
        def _run():
            _dtl.sort_array(vec_or_arr, reverse)
        return AlgorithmFuture(_run)


# =============================================================================
# RMA (Remote Memory Access) Operations
# =============================================================================

# Import Window class from native module
Window = _dtl.Window


def rma_put(window: Window, target: int, offset: int, data) -> None:
    """Put data into a remote window.

    One-sided write operation. Writes data from the local buffer to the
    target rank's window at the specified offset.

    Args:
        window: RMA window to write to
        target: Target rank (0 to size-1)
        offset: Byte offset into target's window
        data: NumPy array with data to write

    Note:
        Must be called within a synchronization epoch (fence or lock/unlock).

    Example:
        >>> win.fence()
        >>> dtl.rma_put(win, target=1, offset=0, data=local_array)
        >>> win.fence()
    """
    import numpy as np
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    data = np.ascontiguousarray(data)
    _dtl.rma_put(window, target, offset, data)


def rma_get(window: Window, target: int, offset: int, size: int, dtype) -> "np.ndarray":
    """Get data from a remote window.

    One-sided read operation. Reads data from the target rank's window
    into a new local array.

    Args:
        window: RMA window to read from
        target: Target rank (0 to size-1)
        offset: Byte offset into target's window
        size: Number of bytes to read
        dtype: NumPy dtype for the returned array

    Returns:
        NumPy array containing the retrieved data

    Note:
        Must be called within a synchronization epoch (fence or lock/unlock).

    Example:
        >>> win.fence()
        >>> data = dtl.rma_get(win, target=0, offset=0, size=80, dtype=np.float64)
        >>> win.fence()
    """
    import numpy as np
    return _dtl.rma_get(window, target, offset, size, np.dtype(dtype))


def rma_accumulate(window: Window, target: int, offset: int, data, op: str = "sum") -> None:
    """Atomic accumulate operation on remote window.

    Atomically combines local data with data in the target's window
    using the specified operation.

    Args:
        window: RMA window to accumulate to
        target: Target rank (0 to size-1)
        offset: Byte offset into target's window
        data: NumPy array with data to accumulate
        op: Reduction operation ("sum", "prod", "min", "max", "band", "bor", "bxor")

    Note:
        Must be called within a synchronization epoch (fence or lock/unlock).

    Example:
        >>> win.fence()
        >>> dtl.rma_accumulate(win, target=0, offset=0, data=local_array, op="sum")
        >>> win.fence()
    """
    import numpy as np
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    data = np.ascontiguousarray(data)
    _dtl.rma_accumulate(window, target, offset, data, op)


def rma_fetch_and_add(window: Window, target: int, offset: int, addend, dtype):
    """Atomic fetch and add on remote location.

    Atomically reads the value at target's window, adds the addend,
    and stores the result. Returns the original value.

    Args:
        window: RMA window
        target: Target rank (0 to size-1)
        offset: Byte offset into target's window
        addend: Value to add
        dtype: NumPy dtype of the element

    Returns:
        The original value before the addition

    Note:
        Must be called within a lock epoch for passive target synchronization.

    Example:
        >>> win.lock(target=0)
        >>> old = dtl.rma_fetch_and_add(win, target=0, offset=0, addend=1, dtype=np.int64)
        >>> win.unlock(target=0)
    """
    import numpy as np
    return _dtl.rma_fetch_and_add(window, target, offset, addend, np.dtype(dtype))


def rma_compare_and_swap(window: Window, target: int, offset: int,
                         compare, swap, dtype):
    """Atomic compare and swap on remote location.

    Atomically compares the value at target's window with 'compare'.
    If equal, replaces with 'swap'. Returns the original value.

    Args:
        window: RMA window
        target: Target rank (0 to size-1)
        offset: Byte offset into target's window
        compare: Expected value to compare against
        swap: New value to store if compare matches
        dtype: NumPy dtype of the element

    Returns:
        The original value (compare succeeded if return == compare)

    Note:
        Must be called within a lock epoch for passive target synchronization.

    Example:
        >>> win.lock(target=0)
        >>> old = dtl.rma_compare_and_swap(win, target=0, offset=0,
        ...                                compare=0, swap=1, dtype=np.int64)
        >>> win.unlock(target=0)
        >>> success = (old == 0)
    """
    import numpy as np
    return _dtl.rma_compare_and_swap(window, target, offset, compare, swap, np.dtype(dtype))


# =============================================================================
# Backend Detection Module
# =============================================================================

class _Backends:
    """Backend availability information.

    Provides methods to query which DTL backends are available
    at runtime.

    Example:
        >>> dtl.backends.available()
        ['MPI']
        >>> dtl.backends.has_mpi()
        True
    """

    @staticmethod
    def has_mpi() -> bool:
        """Check if MPI backend is available."""
        return _dtl.has_mpi()

    @staticmethod
    def has_cuda() -> bool:
        """Check if CUDA backend is available."""
        return _dtl.has_cuda()

    @staticmethod
    def has_hip() -> bool:
        """Check if HIP/AMD backend is available."""
        return _dtl.has_hip()

    @staticmethod
    def has_nccl() -> bool:
        """Check if NCCL is available."""
        return _dtl.has_nccl()

    @staticmethod
    def has_shmem() -> bool:
        """Check if OpenSHMEM backend is available."""
        return _dtl.has_shmem()

    @staticmethod
    def available() -> list:
        """Get list of available backend names.

        Returns:
            List of strings naming available backends.

        Example:
            >>> dtl.backends.available()
            ['MPI', 'CUDA']
        """
        result = []
        if _dtl.has_mpi():
            result.append("MPI")
        if _dtl.has_cuda():
            result.append("CUDA")
        if _dtl.has_hip():
            result.append("HIP")
        if _dtl.has_nccl():
            result.append("NCCL")
        if _dtl.has_shmem():
            result.append("SHMEM")
        return result

    @staticmethod
    def name() -> str:
        """Get the name of the primary backend.

        Returns:
            String describing the primary backend configuration.
        """
        avail = _Backends.available()
        if not avail:
            return "Single"
        return "+".join(avail)

    @staticmethod
    def count() -> int:
        """Get the number of available backends.

        Returns:
            Integer count of enabled backends.
        """
        return len(_Backends.available())

# Singleton instance
backends = _Backends()


# =============================================================================
# MPMD (Multi-Program Multiple Data) - Phase 12.5
# =============================================================================

class RoleManager:
    """MPMD role manager for multi-program execution.

    Manages role-based rank grouping for MPMD applications where
    different ranks execute different programs or code paths.

    Args:
        ctx: Execution context

    Example:
        >>> mgr = dtl.RoleManager(ctx)
        >>> mgr.add_role("workers", num_ranks=3)
        >>> mgr.add_role("coordinator", num_ranks=1)
        >>> mgr.initialize()
        >>> if mgr.has_role("workers"):
        ...     print(f"Worker rank {mgr.role_rank('workers')}")
        >>> mgr.destroy()
    """

    def __init__(self, ctx: Context | None = None) -> None:
        if ctx is None:
            ctx = Context()
        self._ctx = ctx
        native_ctx = ctx._native if hasattr(ctx, '_native') else ctx
        self._mgr = _dtl.mpmd.RoleManager(native_ctx)
        self._initialized = False
        self._role_predicates = {}
        self._active_roles = set()
        self._role_counts = {}

    def add_role(self, name: str, num_ranks) -> None:
        """Add a named role with a specific number of ranks.

        Must be called before initialize(). All ranks must call
        add_role with the same arguments in the same order.

        Args:
            name: Role name identifier
            num_ranks: Number of ranks to assign to this role
        """
        if self._initialized:
            raise RuntimeError("Cannot add roles after initialize()")
        if callable(num_ranks):
            self._role_predicates[name] = num_ranks
            return
        count = int(num_ranks)
        self._role_counts[name] = count
        self._mgr.add_role(name, count)

    def initialize(self, ctx: Context | None = None) -> None:
        """Initialize role assignments (collective).

        Must be called by all ranks after all roles have been added.
        Assigns ranks to roles based on the order they were added.
        """
        if self._initialized:
            raise RuntimeError("RoleManager is already initialized")

        if ctx is not None:
            self._ctx = ctx

        if self._role_predicates:
            rank = int(self._ctx.rank)
            size = int(self._ctx.size)
            self._active_roles = {
                name for name, pred in self._role_predicates.items()
                if bool(pred(rank, size))
            }
            self._initialized = True
            return

        if not self._role_counts:
            # Compatibility mode: allow empty role manager initialization.
            self._initialized = True
            return

        self._mgr.initialize()
        self._initialized = True

    def initialized(self) -> bool:
        """Return True once initialize() has successfully completed."""
        return bool(self._initialized)

    def has_role(self, name: str) -> bool:
        """Check if this rank belongs to a role.

        Args:
            name: Role name to check

        Returns:
            True if this rank is assigned to the named role
        """
        if self._role_predicates:
            return name in self._active_roles
        if not self._initialized:
            return False
        return self._mgr.has_role(name)

    def role_size(self, name: str) -> int:
        """Get the number of ranks in a role.

        Args:
            name: Role name

        Returns:
            Number of ranks assigned to this role
        """
        if self._role_predicates:
            return 1 if name in self._active_roles else 0
        if name in self._role_counts:
            return int(self._role_counts[name])
        return self._mgr.role_size(name)

    def role_rank(self, name: str) -> int:
        """Get this rank's index within a role.

        Args:
            name: Role name

        Returns:
            This rank's index within the role (0 to role_size-1)
        """
        if self._role_predicates:
            return 0 if name in self._active_roles else -1
        return self._mgr.role_rank(name)

    def destroy(self) -> None:
        """Release role manager resources."""
        self._mgr.destroy()
        self._initialized = False
        self._active_roles.clear()

    def __repr__(self) -> str:
        return f"<RoleManager initialized={self._initialized}>"


def intergroup_send(mgr: RoleManager, target_role: str, target_rank: int,
                    data, tag: int = 0) -> None:
    """Send data to a rank in another role group.

    Point-to-point communication between ranks in different MPMD roles.

    Args:
        mgr: RoleManager with initialized roles
        target_role: Name of the target role group
        target_rank: Rank index within the target role (0 to role_size-1)
        data: NumPy array to send
        tag: Message tag for matching (default: 0)

    Example:
        >>> if mgr.has_role("workers"):
        ...     dtl.intergroup_send(mgr, "coordinator", 0, local_result, tag=42)
    """
    import numpy as np

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    data = np.ascontiguousarray(data)

    _dtl.mpmd.intergroup_send(mgr._mgr, target_role, target_rank, data, tag)


def intergroup_recv(mgr: RoleManager, source_role: str, source_rank: int,
                    count: int, dtype, tag: int = 0):
    """Receive data from a rank in another role group.

    Point-to-point communication between ranks in different MPMD roles.

    Args:
        mgr: RoleManager with initialized roles
        source_role: Name of the source role group
        source_rank: Rank index within the source role (0 to role_size-1)
        count: Number of elements to receive
        dtype: NumPy dtype of the elements
        tag: Message tag for matching (default: 0)

    Returns:
        NumPy array with received data

    Example:
        >>> if mgr.has_role("coordinator"):
        ...     result = dtl.intergroup_recv(mgr, "workers", 0, count=100,
        ...                                  dtype=np.float64, tag=42)
    """
    import numpy as np
    return _dtl.mpmd.intergroup_recv(mgr._mgr, source_role, source_rank,
                                     count, np.dtype(dtype), tag)


# =============================================================================
# Topology - Phase 12.5
# =============================================================================

class Topology:
    """Hardware topology queries.

    Provides static methods to query hardware topology information
    including CPU and GPU counts, affinity, and locality.

    Example:
        >>> print(f"CPUs: {dtl.Topology.num_cpus()}")
        >>> print(f"GPUs: {dtl.Topology.num_gpus()}")
        >>> if dtl.Topology.is_local(0, 1):
        ...     print("Ranks 0 and 1 are on the same node")
    """

    @staticmethod
    def num_cpus() -> int:
        """Get the number of CPU cores available.

        Returns:
            Number of CPU cores detected on this node
        """
        return _dtl.topology.num_cpus()

    @staticmethod
    def num_gpus() -> int:
        """Get the number of GPU devices available.

        Returns:
            Number of GPU devices detected on this node (0 if no GPUs)
        """
        return _dtl.topology.num_gpus()

    @staticmethod
    def cpu_affinity(rank: int) -> int:
        """Get the CPU core affinity for a rank.

        Args:
            rank: MPI rank to query

        Returns:
            CPU core index the rank is bound to, or -1 if unbound
        """
        return _dtl.topology.cpu_affinity(rank)

    @staticmethod
    def gpu_id(rank: int) -> int:
        """Get the GPU device ID assigned to a rank.

        Args:
            rank: MPI rank to query

        Returns:
            GPU device ID, or -1 if no GPU assigned
        """
        return _dtl.topology.gpu_id(rank)

    @staticmethod
    def is_local(rank_a: int, rank_b: int) -> bool:
        """Check if two ranks are on the same node.

        Args:
            rank_a: First MPI rank
            rank_b: Second MPI rank

        Returns:
            True if both ranks reside on the same physical node
        """
        return _dtl.topology.is_local(rank_a, rank_b)

    @staticmethod
    def node_id(rank: int) -> int:
        """Get the node identifier for a rank.

        Args:
            rank: MPI rank to query

        Returns:
            Integer node identifier (ranks on the same node share
            the same node_id)
        """
        return _dtl.topology.node_id(rank)


# =============================================================================
# Futures (EXPERIMENTAL) - Phase 12.5
# =============================================================================

class Future:
    """Asynchronous future (EXPERIMENTAL - progress engine has known stability issues).

    Represents a value that will be available in the future from
    an asynchronous operation.

    Warning:
        The DTL progress engine has known stability issues with
        async/futures operations. Use with caution and prefer
        synchronous operations for production code.

    Example:
        >>> fut = dtl.Future()
        >>> fut.set(data)
        >>> # ... later ...
        >>> if fut.test():
        ...     result = fut.get(size=100)
    """

    def __init__(self, _native=None) -> None:
        self._fut = _native if _native is not None else _dtl.futures.Future()
        self._payload_size = None

    @classmethod
    def _from_native(cls, native_future) -> "Future":
        return cls(_native=native_future)

    def valid(self) -> bool:
        """Return True if this wrapper still owns a native future."""
        return self._fut is not None

    def wait(self) -> None:
        """Block until the future is fulfilled.

        Warning:
            May hang if the progress engine is not active or if
            there are deadlock conditions.
        """
        self._fut.wait()

    def test(self) -> bool:
        """Test if the future is ready (non-blocking).

        Returns:
            True if the future has been fulfilled
        """
        return self._fut.test()

    def get(self, size: int | None = None):
        """Get the fulfilled value.

        Args:
            size: Number of bytes to retrieve. If omitted, uses the size
                from the prior ``set()`` payload.

        Returns:
            The value stored in the future

        Raises:
            RuntimeError: If the future is not yet fulfilled
        """
        import pickle

        read_size = size
        if read_size is None:
            if self._payload_size is None:
                raise TypeError("Future.get() requires size for unknown payload")
            read_size = self._payload_size

        payload = self._fut.get(read_size)
        try:
            return pickle.loads(payload)
        except Exception:
            return payload

    def set(self, data) -> None:
        """Fulfill the future with a value.

        Args:
            data: NumPy array or bytes to store
        """
        import numpy as np
        import pickle

        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data).tobytes()
        elif isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, bytearray):
            data = bytes(data)
        elif not isinstance(data, (bytes, str)):
            data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        elif isinstance(data, str):
            data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Keep bytes payloads as-is; if they are pickled later get() still works.
            pass

        if isinstance(data, bytes):
            self._payload_size = len(data)
        self._fut.set(data)

    def destroy(self) -> None:
        """Explicitly release the native future handle."""
        if self._fut is not None:
            self._fut.destroy()
            self._fut = None

    def __repr__(self) -> str:
        if self._fut is None:
            return "<Future (destroyed)>"
        ready = "ready" if self.test() else "pending"
        return f"<Future ({ready})>"


def when_all(futures: list) -> Future:
    """Wait for all futures to complete (EXPERIMENTAL).

    Blocks until every future in the list has been fulfilled.

    Args:
        futures: List of Future objects to wait on

    Warning:
        The DTL progress engine has known stability issues.
        May hang if any future cannot be fulfilled.

    Example:
        >>> futures = [launch_async(i) for i in range(4)]
        >>> dtl.when_all(futures)
    """
    result = Future()
    values = [f.get(None) for f in futures]
    result.set(values)
    return result


def when_any(futures: list):
    """Wait for any future to complete (EXPERIMENTAL).

    Blocks until at least one future in the list has been fulfilled.
    Returns the completed future and its index.

    Args:
        futures: List of Future objects to wait on

    Returns:
        Tuple of (completed_future, index) where index is the position
        in the input list

    Warning:
        The DTL progress engine has known stability issues.

    Example:
        >>> futures = [launch_async(i) for i in range(4)]
        >>> fut, idx = dtl.when_any(futures)
        >>> print(f"Future {idx} completed first")
    """
    import time

    if not futures:
        raise ValueError("when_any requires at least one future")

    while True:
        for idx, fut in enumerate(futures):
            if fut.test():
                result = Future()
                result.set((idx, fut.get(None)))
                return result
        time.sleep(0.001)

