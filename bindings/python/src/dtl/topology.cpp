// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file topology.cpp
 * @brief DTL Python bindings - Topology query operations
 * @since 0.1.0
 *
 * Provides Python bindings for querying hardware topology information,
 * including CPU/GPU counts, affinity mappings, and node locality.
 */

#include <pybind11/pybind11.h>

#include <dtl/bindings/c/dtl.h>
#include <dtl/bindings/c/dtl_topology.h>

#include "status_exception.hpp"

#include <stdexcept>
#include <string>

namespace py = pybind11;

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/**
 * @brief Convert DTL status to Python exception
 */
void check_status(dtl_status status) {
    ::dtl::python::check_status_or_throw(status);
}

}  // namespace

// ============================================================================
// Module Binding
// ============================================================================

void init_topology(py::module_& m) {
    auto topo = m.def_submodule("topology", "Hardware topology queries");

    topo.def("num_cpus",
        []() -> int {
            int count = 0;
            dtl_status status = dtl_topology_num_cpus(&count);
            check_status(status);
            return count;
        },
        R"doc(
Get the number of available CPUs/cores.

Returns the number of hardware threads (logical CPUs) available
on the current node.

Returns:
    Number of CPUs (may be 0 if undetermined)

Example:
    >>> n = dtl.topology.num_cpus()
    >>> print(f"Available CPUs: {n}")
)doc");

    topo.def("num_gpus",
        []() -> int {
            int count = 0;
            dtl_status status = dtl_topology_num_gpus(&count);
            check_status(status);
            return count;
        },
        R"doc(
Get the number of available GPU devices.

Returns the number of GPU devices visible on the current node.
Returns 0 when no GPU backend is enabled or no devices are found.

Returns:
    Number of GPUs (0 if no GPU backend)

Example:
    >>> n = dtl.topology.num_gpus()
    >>> print(f"Available GPUs: {n}")
)doc");

    topo.def("cpu_affinity",
        [](dtl_rank_t rank) -> int {
            int cpu_id = 0;
            dtl_status status = dtl_topology_cpu_affinity(rank, &cpu_id);
            check_status(status);
            return cpu_id;
        },
        py::arg("rank"),
        R"doc(
Get CPU affinity for a rank.

Returns a CPU identifier that the given rank should be bound to,
using a simple round-robin mapping: cpu_id = rank % num_cpus.

Args:
    rank: The MPI rank to query

Returns:
    CPU identifier (0-based)

Example:
    >>> cpu = dtl.topology.cpu_affinity(ctx.rank)
    >>> print(f"Rank {ctx.rank} -> CPU {cpu}")
)doc");

    topo.def("gpu_id",
        [](dtl_rank_t rank) -> int {
            int gpu_id = 0;
            dtl_status status = dtl_topology_gpu_id(rank, &gpu_id);
            check_status(status);
            return gpu_id;
        },
        py::arg("rank"),
        R"doc(
Get GPU device for a rank.

Returns a GPU device identifier for the given rank, using a
simple round-robin mapping: gpu_id = rank % num_gpus. Returns
-1 if no GPU devices are available.

Args:
    rank: The MPI rank to query

Returns:
    GPU device identifier (0-based), or -1 if no GPUs

Example:
    >>> gpu = dtl.topology.gpu_id(ctx.rank)
    >>> if gpu >= 0:
    ...     print(f"Rank {ctx.rank} -> GPU {gpu}")
)doc");

    topo.def("is_local",
        [](dtl_rank_t rank_a, dtl_rank_t rank_b) -> bool {
            int result = 0;
            dtl_status status = dtl_topology_is_local(rank_a, rank_b, &result);
            check_status(status);
            return result != 0;
        },
        py::arg("rank_a"),
        py::arg("rank_b"),
        R"doc(
Check if two ranks share a node.

Determines whether two MPI ranks are located on the same physical
node. Without MPI, all ranks are assumed to be local.

Args:
    rank_a: First rank
    rank_b: Second rank

Returns:
    True if both ranks are on the same node

Note:
    This is a collective operation when MPI is enabled.

Example:
    >>> if dtl.topology.is_local(0, 1):
    ...     print("Ranks 0 and 1 are on the same node")
)doc");

    topo.def("node_id",
        [](dtl_rank_t rank) -> int {
            int nid = 0;
            dtl_status status = dtl_topology_node_id(rank, &nid);
            check_status(status);
            return nid;
        },
        py::arg("rank"),
        R"doc(
Get node identifier for a rank.

Returns a node identifier for the given rank. Ranks on the same
physical node will have the same node identifier. Without MPI,
all ranks are on node 0.

Args:
    rank: The MPI rank to query

Returns:
    Node identifier (0-based)

Note:
    This is a collective operation when MPI is enabled.

Example:
    >>> nid = dtl.topology.node_id(ctx.rank)
    >>> print(f"Rank {ctx.rank} is on node {nid}")
)doc");
}
