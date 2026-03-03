#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent

EXPECTED = {
    ROOT / "CMakeLists.txt": [
        'set(DTL_VERSION_PRERELEASE "alpha.1")',
        'set(DTL_VERSION_PYTHON "0.1.0a1")',
    ],
    ROOT / "README.md": ["Version: **0.1.0-alpha.1**", "Version 0.1.0-alpha.1"],
    ROOT / "CITATION.cff": ['version: "0.1.0-alpha.1"'],
    ROOT / "bindings/python/pyproject.toml": ['version = "0.1.0a1"'],
    ROOT / "bindings/python/src/dtl/__init__.py": ['__version__ = "0.1.0a1"'],
    ROOT / "docs/Doxyfile.in": ["PROJECT_NUMBER         = @DTL_VERSION_FULL@"],
    ROOT / "docs/api_reference/c_api_reference.md": ["**DTL Version:** 0.1.0-alpha.1"],
    ROOT / "docs/api_reference/cpp_quick_reference.md": ["**DTL Version:** 0.1.0-alpha.1"],
    ROOT / "docs/api_reference/python_api_reference.md": ["**DTL Version:** 0.1.0-alpha.1"],
    ROOT / "docs/api_reference/fortran_api_reference.md": ["**DTL Version:** 0.1.0-alpha.1"],
    ROOT / "docs/index.md": ["**Current Version**: 0.1.0-alpha.1"],
    ROOT / "docs/conf.py": ['release = "0.1.0-alpha.1"'],
    ROOT / "include/dtl/dtl.hpp": ["// Version: 0.1.0-alpha.1"],
    ROOT / "examples/fortran/README.md": ["reference only", "bindings/fortran/examples"],
}


def main() -> int:
    errors: list[str] = []
    for path, snippets in EXPECTED.items():
        text = path.read_text()
        for snippet in snippets:
            if snippet not in text:
                errors.append(f"{path.relative_to(ROOT)} missing snippet: {snippet}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("release metadata looks consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
