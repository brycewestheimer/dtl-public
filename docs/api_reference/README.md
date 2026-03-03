# API Reference

The API reference is generated from source headers with Doxygen and rendered
into the Sphinx site via Breathe.

## Local build

```bash
doxygen docs/Doxyfile
sphinx-build -b html docs docs/_build/html
```

Generated Doxygen artifacts are written under `docs/_generated/doxygen/`.
The rendered API section is available in the Sphinx output under
`docs/_build/html/api_reference/`.
