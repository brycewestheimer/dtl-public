# 10. Release Process and Checklists

## Release goals

A release candidate should satisfy:

- stable C++ core behavior for supported surfaces
- C ABI behavior consistent and tested
- Python and Fortran bindings build and pass expected test gates
- docs site and API references build successfully

## Contributor release checklist

### Build and tests

- [ ] build core/bindings with `-j6`
- [ ] C ABI tests pass
- [ ] Fortran basic test passes
- [ ] Python non-MPI/non-CUDA suite passes

### Documentation

- [ ] Doxygen generation succeeds
- [ ] Sphinx site generation succeeds
- [ ] new contributor-facing pages are in docs navigation
- [ ] `distributed_span` coverage is present in user/developer/bindings guides where non-owning semantics are relevant

### API quality

- [ ] public API comments complete and signature-aligned
- [ ] backend-unavailable paths return explicit status codes
- [ ] ownership/lifetime contracts documented for new handles/wrappers
- [ ] parity matrix rows are updated for any changed container/view semantics (including `distributed_span`)

## Version and packaging notes

- keep package version sources aligned (CMake/package/binding expectations)
- avoid hardcoded test expectations that diverge from project version metadata
