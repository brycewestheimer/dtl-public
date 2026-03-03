# Developer Guide

This guide is the contributor-facing handbook for DTL internals, workflows, standards, and release gates.

## Who this is for

Use this guide if you are:

- adding or changing C++ library behavior
- changing C, Python, or Fortran bindings
- touching ABI surface areas
- updating test strategy, docs, packaging, or release assets

## How to use this guide

Read chapter 1 through 4 once before major changes. Then use the API- and language-specific chapters as needed.

```{toctree}
:maxdepth: 2

01-development-environment
02-codebase-architecture
03-cpp-contribution-standards
04-memory-management-and-lifetime
05-c-abi-development
06-python-bindings-development
07-fortran-bindings-development
08-testing-and-quality-gates
09-documentation-and-api-comments
10-release-process-and-checklists
11-debugging-and-troubleshooting
12-runtime-and-handle-development
```
