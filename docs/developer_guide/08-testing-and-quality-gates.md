# 8. Testing and Quality Gates

## Test categories

- C++ unit and integration tests
- C ABI tests
- Python binding tests
- Fortran smoke tests

## Minimum pre-merge gate for binding or ABI work

1. build targets with `-j6`
2. run C ABI tests
3. run Fortran basic test
4. run Python non-MPI/non-CUDA suite

## Recommended command set

```bash
cmake --build <build-dir> -j6 --target _dtl test_c_bindings test_fortran_basic
ctest --test-dir <build-dir> -R '^(CBindingsTests|fortran_basic_test)$' -j6 --output-on-failure
MPI4PY_RC_INITIALIZE=0 PYTHONPATH=<build-dir>/bindings/python \
  python3 -m pytest bindings/python/tests -q -m 'not mpi and not cuda'
```

## Regression triage guidance

- first isolate whether breakage is C++ core, C ABI, or binding adapter
- check version/packaging assumptions before changing runtime code
- keep failing tests and root cause linked in PR description

## Quality bar

- no known crashes, leaks, or invalid-handle paths in modified areas
- docs updated for externally visible changes
- release checklist updated when introducing new requirements
