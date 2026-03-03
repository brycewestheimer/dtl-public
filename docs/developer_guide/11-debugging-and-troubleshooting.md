# 11. Debugging and Troubleshooting

## Build failures

- confirm feature flags in CMake configure output
- verify optional dependencies are either present or disabled
- reconfigure from clean build directory when toggling major options

## Python import/runtime issues

- ensure `PYTHONPATH=<build-dir>/bindings/python` for local extension tests
- verify native module `_dtl` was rebuilt for current Python version
- check mismatch between wrapper assumptions and native signatures

## C ABI runtime issues

- validate handle magic/validity checks first
- inspect backend availability decision branches
- verify destroy paths for partial-initialization failures

## Fortran interop issues

- confirm `bind(c)` signatures match C ABI prototypes exactly
- check `c_ptr` and scalar type compatibility
- reproduce with `fortran_basic_test` before broader debugging

## Documentation issues

- Doxygen param warnings usually indicate signature/comment drift
- missing toctree entries hide docs pages from site navigation
- run docs build locally before opening PR

## Escalation path for hard regressions

1. isolate minimal reproducer
2. identify last known good commit
3. classify by layer (core, C ABI, Python, Fortran, docs)
4. add regression test before final fix
