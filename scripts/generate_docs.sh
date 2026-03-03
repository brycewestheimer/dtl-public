#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
# Configure and build the DTL documentation pipeline.
#
# Usage:
#   bash scripts/generate_docs.sh [build_dir] [-- <cmake-build-args...>]
#
# Examples:
#   bash scripts/generate_docs.sh
#   bash scripts/generate_docs.sh build-docs
#   bash scripts/generate_docs.sh build-docs -- -j6
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="build-docs"
BUILD_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/generate_docs.sh [build_dir] [-- <cmake-build-args...>]

Configures a docs-focused build directory and builds the `docs` target.
Documentation output is written to:
  <build_dir>/docs/html

Examples:
  bash scripts/generate_docs.sh
  bash scripts/generate_docs.sh build-docs
  bash scripts/generate_docs.sh build-docs -- -j6
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
    BUILD_DIR="$1"
    shift
fi

if [[ "${1:-}" == "--" ]]; then
    shift
    BUILD_ARGS=("$@")
elif [[ $# -gt 0 ]]; then
    echo "ERROR: unexpected arguments: $*" >&2
    usage >&2
    exit 2
fi

cd "${ROOT_DIR}"

cmake -S . -B "${BUILD_DIR}" \
    -DDTL_BUILD_DOCS=ON \
    -DDTL_ENABLE_MPI=OFF \
    -DDTL_BUILD_TESTS=OFF \
    -DDTL_BUILD_EXAMPLES=OFF \
    -DDTL_BUILD_BENCHMARKS=OFF

cmake --build "${BUILD_DIR}" --target docs "${BUILD_ARGS[@]}"

echo ""
echo "=============================================="
echo "  Documentation generated"
echo "=============================================="
echo "HTML site: ${BUILD_DIR}/docs/html/index.html"
