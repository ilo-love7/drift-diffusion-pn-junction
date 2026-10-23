# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-10-23

Reusable verification tooling and cross-repository discoverability. Nothing in this release
changes the solver's physics, its committed products or its scope: the three evidence layers are
now also packaged as an API, and **that API states the same limits the README already states** -
it does not upgrade any claim.

### Added

- **`p5dd.verify`: one structured report for the three evidence layers** (`src/p5dd/verify.py`).
  Each layer is reported as name / criterion / measured value / threshold / pass-fail, together
  with **what that layer cannot show**, so a report can never be read as a stronger claim than
  the study makes:
  * **L1** (analytic invariants) certifies the discretisation, **not** the physics model chosen;
  * **L2** (mesh independence) does **not** show that the converged limit is the right answer;
  * **L3** (cross-check vs NIST Sesame 2.0) is an independent *code*, **not** an independent
    validation chain, and because both codes share the same model set it has zero power over
    model completeness.
  It adds no new metric and no new threshold: it re-runs the case scripts that already exist
  (`scripts/case1_equilibrium.py`, `scripts/case4_mesh_convergence.py`,
  `scripts/case5_sesame_crosscheck.py`, `scripts/case7_sesame_profiles.py`) or reads the products
  they commit under `data/`, and it reads case 7's own declared `GATES` instead of restating
  thresholds. The declared product paths and the existing CLIs are unchanged.
- **A third-party path**: `evaluate_layer("L1" | "L2" | "L3", metrics)` applies this repository's
  declared gates to a foreign metrics mapping, so another solver can be measured against these
  criteria without importing this one. The thresholds always come from this repository, never
  from the submitted report, so a report cannot widen a gate.
- **Test coverage for the API** (`tests/test_verify_api.py`): the report schema, each layer's
  pass/fail, and honest failure - a breached or missing metric fails its check, a layer that
  could not run is reported `not_executed` with `passed is False`, and an L1 failure is not
  compensated by L3. Suite size: **113 -> 118 tests**.
- **README** `## Verifying another solver: the reusable three-layer API`, with a minimal runnable
  snippet; and `## Related repositories` / `## 关联仓库` cross-linking the sibling repositories.
- **CI**: a `windows-latest` leg (`suite-windows`) reusing the same Python matrix, so the test
  matrix grows from 3 jobs to 6. `MPLBACKEND=Agg` is set for the headless runner, and
  `.gitattributes` pins `reference/**`, `data/**` and `figures/**` to `-text`, so the vendored
  Sesame copy and the committed products stay byte-identical on a Windows checkout.

### Changed

- `## Layout` / `## 目录结构` now register the new module (`src/p5dd/verify.py`).
- README test counts synced with the suite (113 -> 118).

### Fixed

- Documentation consistency: the README still claimed 113 tests after the suite had grown, and
  quoted a skip count that no longer matched the pinned `importorskip` behaviour. Both are now
  stated as measured, and skips are still never counted as passes. No solver code, metric,
  threshold or committed product changed in this release.

### Not changed (deliberately)

- The scope boundary. `## Scope boundary: not implemented, not verified` still lists MOS /
  MOSFET structures, pin and heterojunctions, graded-doping junctions, AC small-signal and 2-D
  devices as **not implemented and not verified**, and a passing three-layer report is not a
  claim that a solver is correct.
- The vendored NIST Sesame 2.0 copy under `reference/` is untouched, so the two SHA-256 integrity
  digests quoted in the README still hold.

## [1.0.0] - 2026-10-22

Initial release: a from-scratch 1-D steady-state drift-diffusion-Poisson solver (finite volume,
Scharfetter-Gummel links, analytic sparse Jacobian, damped Newton with adaptive bias
continuation) together with its verification study against closed-form semiconductor theory and
the independently written NIST Sesame 2.0 - the nine case scripts, the committed metrics / CSV /
figure products, and the vendored reference copy. Release notes:
<https://github.com/ilo-love7/drift-diffusion-pn-junction/releases/tag/v1.0.0>.

[Unreleased]: https://github.com/ilo-love7/drift-diffusion-pn-junction/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ilo-love7/drift-diffusion-pn-junction/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ilo-love7/drift-diffusion-pn-junction/releases/tag/v1.0.0