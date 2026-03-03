#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-examples-smoke"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDTL_BUILD_TESTS=OFF \
  -DDTL_BUILD_EXAMPLES=ON \
  -DDTL_BUILD_C_BINDINGS=OFF \
  -DDTL_BUILD_FORTRAN=OFF \
  -DDTL_ENABLE_MPI=ON

cmake --build "${BUILD_DIR}" --target hello_distributed distributed_vector_sum -j"$(nproc)"

"${BUILD_DIR}/examples/hello_distributed"
MPI_RANKS="${DTL_CI_MPI_RANKS:-2}"
if [ "$(nproc)" -lt "${MPI_RANKS}" ]; then
  MPI_RANKS="$(nproc)"
fi
if [ "${MPI_RANKS}" -lt 1 ]; then
  MPI_RANKS=1
fi

mpirun --oversubscribe --bind-to none -np "${MPI_RANKS}" "${BUILD_DIR}/examples/distributed_vector_sum"

echo "Examples smoke validation passed."
