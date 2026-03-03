# 12. Runtime and Handle Development

## Scope

This chapter defines contributor standards for modifying runtime capability plumbing and handle-based APIs.

## Runtime architecture touchpoints

Primary runtime-related areas:

- environment/runtime registry logic
- backend capability discovery and reporting
- context/domain creation and validation
- runtime library boundaries and initialization order

## Handle categories

- internal C++ RAII handles
- C ABI opaque handles
- binding-layer wrappers (Python/Fortran over C handles)

Each category must preserve deterministic ownership and clear validity rules.

## Required invariants

1. Handle validity is checkable at API boundary.
2. Handle destruction is safe and deterministic.
3. Backend capability reporting is consistent across layers.
4. No operation dereferences invalid or moved-from handle state.

## C ABI-specific requirements

- allocate handles only after argument validation where possible
- ensure partial-initialization failure paths release allocations
- keep error codes specific:
  - invalid argument/handle
  - backend unavailable
  - backend init/operation failure

## Python wrapper requirements

- wrappers own native handles via RAII-like lifetime control
- requests/actions/windows/contexts do not outlive dependencies
- exposed async APIs provide explicit lifecycle operations (`wait/test/destroy` equivalents)

## Fortran wrapper requirements

- signatures remain aligned with C ABI exactly
- wrapper changes track C ABI evolution in same PR
- basic Fortran handle lifecycle tests remain green

## Review checklist for runtime/handle changes

- [ ] capability checks and status returns are consistent
- [ ] no leaked handles on failure branches
- [ ] cross-language parity preserved for changed behavior
- [ ] tests updated across C ABI/Python/Fortran where affected
- [ ] docs updated in both developer and user guides

## Validation commands

```bash
cmake --build <build-dir> -j6 --target _dtl test_c_bindings test_fortran_basic
ctest --test-dir <build-dir> -R '^(CBindingsTests|fortran_basic_test)$' -j6 --output-on-failure
MPI4PY_RC_INITIALIZE=0 PYTHONPATH=<build-dir>/bindings/python \
  python3 -m pytest bindings/python/tests -q -m 'not mpi and not cuda'
```
