# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# Backend Benchmarks — MPI, CUDA, NCCL collective performance
# =============================================================================
# NOTE: These benchmarks are registered in the PARENT benchmarks/CMakeLists.txt
# using the dtl_add_benchmark() helper (which accepts relative paths from the
# benchmarks/ directory). This file exists only for documentation.
#
# Source files in this directory:
#   bench_mpi_collectives.cpp   — MPI allreduce, broadcast, barrier, send/recv
#   bench_cuda_operations.cpp   — CUDA malloc, memcpy, executor overhead
#   bench_nccl_collectives.cpp  — NCCL allreduce, broadcast latency
# =============================================================================
