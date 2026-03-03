# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# Mini-app examples demonstrating DTL in realistic HPC workloads
# =============================================================================
# Each mini-app also has its own standalone CMakeLists.txt for out-of-tree
# building against an installed DTL (via find_package(DTL)).
# =============================================================================

dtl_add_mpi_example(cg_solver  cg_solver/cg_solver.cpp)
dtl_add_mpi_example(nbody      nbody/nbody.cpp)
