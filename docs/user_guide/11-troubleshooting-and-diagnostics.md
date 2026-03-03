# 11. Troubleshooting and Diagnostics

## Build/configuration failures

- verify CMake summary matches intended backend flags
- disable unavailable optional backends explicitly
- reconfigure from clean build directory after major flag changes

## Runtime capability mismatches

Symptoms:

- backend unavailable errors at runtime
- unexpected single-rank behavior

Actions:

1. print capability checks at startup (`has_mpi/has_cuda/...`)
2. verify runtime environment and launch mode
3. verify code path is backend-gated correctly

## Collective hangs or deadlocks

Common causes:

- mismatched collective participation
- diverging control flow by rank
- rank-specific early return before collective

Debug steps:

- add rank-scoped logging around collective boundaries
- minimize test case to smallest reproducer
- verify every rank reaches each collective call

## Binding-specific issues

### C ABI

- validate handles before use
- ensure matching create/destroy pairs

### Python

- ensure correct `PYTHONPATH` for local extension builds
- verify `_dtl` extension matches active Python interpreter

### Fortran

- verify `bind(c)` signatures and type mapping
- confirm explicit destroy calls for handles

## Diagnostics best practices

- include rank, backend, and context info in logs
- keep deterministic reproducer scripts for failures
- separate environment/setup issues from library contract issues

## Additional resources

- `docs/user_guide/troubleshooting.md`
- `docs/process/known_issues_workflow.md`

## Deep-dive reference

- [Legacy Deep-Dive: Troubleshooting](troubleshooting.md)
- [Runtime and Handle Model](13-runtime-and-handle-model.md)
