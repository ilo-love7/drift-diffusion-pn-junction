# Vendored third-party code: Sesame

`reference/sesame/` is a vendored copy of **Sesame 2.0**, the finite-difference
drift-diffusion–Poisson solver developed at the U.S. National Institute of
Standards and Technology (NIST) and the University of Maryland.

| item | value |
| --- | --- |
| Project | Sesame |
| Version | 2.0 (`sesame/_version.py`) |
| Authors | Benoit Gaury (NIST / University of Maryland), Paul M. Haney (NIST) |
| Upstream | `http://github.com/usnistgov/sesame` (as recorded in `INSTALL.rst`) |
| Docs | `http://sesame.readthedocs.io/en/latest/` (as recorded in `README.rst`) |
| License | 3-clause BSD-style, `LICENSE.rst`, "Copyright 2017 University of Maryland" |
| Support | NIST and University of Maryland, CRA 70NANB14H209 (`AUTHORS.rst`) |

## Modifications

**No retained file was edited.** `LICENSE.rst`, `AUTHORS.rst`, `INSTALL.rst`,
`README.rst`, `setup.py`, `setup.cfg`, `npdoc.txt`, `app.py`, `app.spec` and the
whole importable package directory `sesame/` are byte-for-byte the copy that was
archived at intake.

Publishing this repository removed three subtrees that contain no solver code —
documentation, example input scripts and Sesame's own GUI test suite:

| removed subtree | `.py` files | content |
| --- | --- | --- |
| `doc/` | 1 | reST documentation, SVG/PNG figures (11 MB of the original 15 MB) |
| `examples/` | 10 | standalone example scripts, never imported |
| `test_suite/` | 9 | Sesame's own GUI-oriented tests |

The package directory `sesame/` (22 `.py`, including `ui/` and the unbuilt
`mumps/` binding) was kept whole rather than selectively pruned, so the
"nothing we ship was modified" claim stays simple and checkable.

## Integrity digests

```bash
# the tree shipped here: 24 .py files
cd reference/sesame && find . -name "*.py" -type f | sort | xargs cat | sha256sum
# fa1698474ee02d89c632761742c122df5e8f3a13cbe0f68fdcf9f8a481f1ee30

# the complete pre-trim copy (44 .py, i.e. with doc/, examples/, test_suite/ present)
# b1b093cf389a487b8d249b9b782e6343ca17aeb36272c94054ccca851cca9cb0
```

The second digest is carried over as provenance for the complete copy from which
this trimmed one was made; it is reproducible from upstream by running the same
command before the removal.

## How this project uses it

Only the scriptable API is touched, from
[`src/p5dd/devices.py`](../src/p5dd/devices.py) `sesame_system()`:

```python
sys_ = sesame.Builder(x, input_length="cm", T=300.0)
sys_.add_material({...})
sys_.add_donor(ND, lambda pos: pos < x_j)
sys_.add_acceptor(NA, lambda pos: pos >= x_j)
sys_.contact_type("Ohmic", "Ohmic")
sys_.contact_S(sv, sv, sv, sv)
J = sesame.IVcurve(sys_, voltages, out, verbose=False, tol=1e-9)
```

`sesame` is added to `sys.path` at run time by
[`scripts/case5_sesame_crosscheck.py`](../scripts/case5_sesame_crosscheck.py);
it is never installed and is not imported by the solver itself, so the rest of
the project runs without this directory. The optional MUMPS direct-solver binding
(`sesame/mumps/`) is not built and not used — Sesame falls back to SciPy, which
is what the cross-check ran with.

After the trim above the cross-check still runs end to end: `pytest -q` reports
**93 collected / 93 passed / 0 skipped**. Deleting `reference/` entirely turns
case 5 and 4 tests into skips and leaves the other 89 running; skips are never
counted as passes (see [`reports/acceptance-checklist.md`](../reports/acceptance-checklist.md)).

## Redistributing

Sesame is 3-clause BSD: redistribution in source or binary form requires keeping
the copyright notice and the license text, both retained verbatim here. If you
republish this repository, keep `reference/sesame/LICENSE.rst`,
`reference/sesame/AUTHORS.rst` and this note — and do not present Sesame as part
of this project's own contribution. Its role here is an external measuring
stick; the physics it checks is stated in `reports/technical-report.md` §1.3.
