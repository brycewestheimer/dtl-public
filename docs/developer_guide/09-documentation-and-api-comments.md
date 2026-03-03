# 9. Documentation and API Comments

## Toolchain

- Doxygen for C/C++ API docs: `docs/Doxyfile`
- Sphinx + Breathe (RTD theme) for documentation site: `docs/conf.py`

## Local docs commands

```bash
doxygen docs/Doxyfile
sphinx-build -b html docs docs/_build/html
```

Doxygen output is generated under `docs/_generated/doxygen/` and consumed by Breathe.

Or via CMake docs targets:

```bash
bash scripts/generate_docs.sh build-docs -- -j6
```

## Doxygen standards for public APIs

- every public class/function has a brief description
- `@param` names exactly match signature names
- return behavior is explicit for non-void APIs
- lifetime/thread/backend caveats are in `@note` or `@details`
- non-owning surfaces (for example `distributed_span`) explicitly state ownership and invalidation constraints

## Python docs and docstrings

- keep user-facing Python functions/classes documented in docstrings
- ensure docs pages link to binding usage and semantics
- maintain consistency between examples and tested behavior

## Site organization expectations

- include new docs in toctrees
- avoid orphan pages for primary contributor docs
- keep contributor docs discoverable from `docs/index.md`
