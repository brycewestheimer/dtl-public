# 12. Migration and Upgrade Guidance

## Migration goals

Upgrade safely while preserving correctness and performance.

## Recommended migration process

1. lock current baseline behavior with tests
2. upgrade DTL and rebuild with explicit flags
3. run compatibility and regression suites
4. address API/behavior changes incrementally

## Areas to verify during upgrades

- environment/context lifecycle behavior
- policy defaults and compatibility
- collective and remote-operation contracts
- binding behavior across C/Python/Fortran

## Version alignment checks

Ensure consistent version expectations across:

- CMake project version
- package metadata
- binding-exposed version fields
- test assertions

## Risk-managed rollout pattern

- stage in non-MPI/non-CUDA mode first
- enable one backend class at a time
- compare performance/correctness against baseline metrics

## Migration references

- `docs/user_guide/migration_v1_to_v15.md`
- `docs/migration/from_stl.md`

## Final pre-production checklist

- build and test matrix passed for deployed backends
- docs updated for user-facing behavior changes
- known limitations and rollout caveats documented

## Deep-dive reference

- [Legacy Deep-Dive: Migration (V1.0 to V1.5)](migration_v1_to_v15.md)
