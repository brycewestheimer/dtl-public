# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.1.0-alpha.1] - 2026-02-27

First public alpha pre-release. Signals that the API/ABI may change in non-trivial ways.

### Changed
- Version downgraded from 1.0.0 to 0.1.0-alpha.1 (alpha pre-release)

### Removed
- Stub backends: GASNet-EX, UCX, SYCL (all-stub, not production-ready)
- Distributed map C/Fortran binding surface (C++ core retained)
- Remote RPC C/Fortran/Python binding surface (all stubs returning not-supported)

---

[Unreleased]: https://github.com/brycewestheimer/dtl-public/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/brycewestheimer/dtl-public/releases/tag/v0.1.0-alpha.1
