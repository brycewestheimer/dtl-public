#!/usr/bin/env bash
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
# Run DTL concept compliance tests
set -euo pipefail
BUILD_DIR="${1:-build}"
cd "$(dirname "$0")/.."
ctest --test-dir "$BUILD_DIR" --output-on-failure -R "concept|Concept" "$@"
