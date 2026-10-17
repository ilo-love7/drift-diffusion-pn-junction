# Drift–Diffusion PN Junction · 一维漂移-扩散求解器与 Si p-n 结全套验证

A from-scratch 1-D steady-state drift-diffusion–Poisson solver (finite volume,
Scharfetter–Gummel fluxes, analytic sparse Jacobian, damped Newton with adaptive bias
continuation), verified against closed-form semiconductor theory **and** against an
independent third-party device simulator (NIST Sesame 2.0) — numpy/scipy only, no GPU, no
build chain, whole verification suite reproducible in about 20 seconds.

> 中文摘要：自研 1D 有限体积 Scharfetter–Gummel 求解器 + Si p-n 结二极管的全套验证
> （I-V / 理想因子 / 内建电势 / 耗尽层宽 / C-V / 网格无关性 / 真实硅物性 / 双码交叉对照），
> 全部产物本地落盘、一条命令复现。交付边界用"已验证 / 外推"两栏写死；
> MOSFET 等结构**未实现、未验证**，本仓库不声明对其具备能力。

**93 collected / 93 passed / 0 skipped** (≈10 s) · 6 verification cases (≈18 s) · MIT（vendored Sesame 部分仍为 3-clause BSD）

---

## Results at a glance

| # | what is checked | measured | verdict |
|---|---|---|---|
| 1 | Thermal equilibrium invariants (V = 0, 1601 nodes) | terminal current 1.6e-26 A/cm², quasi-Fermi flatness 1.8e-15 V, max \|np/n_i²−1\| = 6.4e-14, V_bi matches the analytic value to **0.0** | ✅ |
| 2 | Forward I-V, 0 → 0.70 V (51 points) | **≤ 4 Newton iterations per point**, weighted residual 9.7e-9; ideality 1.58 (low bias) → 1.12 (mid bias); thermal voltage recovered from the ln J – V_j slope within **3.1 %** | ✅ |
| 3 | Reverse branch + C-V, 0 → −0.70 V (71 points) | depletion width from ε/C within **1.67 %** of the analytic value; Mott–Schottky R² = 0.99977 (effective doping 6.5 % off); SRH generation reaches 0.99999 of its theoretical ceiling | ✅ |
| 4 | Mesh independence, 201 → 3201 nodes | observed order **1.95 (J@0.60) / 1.99 (W)**; 1601 vs 3201 differ by 1.8e-5 / 3.3e-5. *Boundary of validity: this device and this mesh template.* The J@0.30 difference has dropped to the 7e-8 solver noise floor, so its order is distorted and is **not** used as a criterion. | ✅ 2nd order |
| 5 | Cross-check vs **NIST Sesame 2.0**, identical inputs, 23 bias points | max relative deviation of terminal current **2.85e-4**, median 1.29e-6; J(0.60 V) = −0.8014472 vs −0.8014469 A/cm²; this solver is **8.5×** faster on this machine. *Covers terminal current only; both codes share the same model set, so this layer says nothing about model completeness — reference truth stays with L1.* | ✅ |
| 6 | Realistic silicon (Arora mobility + Slotboom band-gap narrowing) | V_bi drops **21.3 mV**, local n_i ×1.60, J_s ×1.90, J(0.60 V) ×1.54; equilibrium invariant stays at 8.0e-14 | ✅ |
| 7 | Unit tests | **93 collected / 93 passed / 0 skipped**, incl. analytic-vs-finite-difference Jacobian, a band-gap-narrowing generalised-SG regression test, and an exact-degeneracy assertion when narrowing is absent | ✅ |
| 8 | Reproduction cost | all six cases **≈18 s** (≈8 s with `--fast`); every product is a local CSV/PNG/JSON | ✅ |

Scale check, Si p-n junction (N_D = 1e17, N_A = 1e16 cm⁻³, 300 K):
V_bi = 0.78013 V, W(0) = 0.3331 µm, E_max(0) = 4.68e4 V/cm,
L_n = 5.91 µm, L_p = 3.52 µm, J_s = 7.87e-11 A/cm².

## Figures

| Forward I-V | Ideality factor | Depletion / C-V | Mesh convergence |
|---|---|---|---|
| ![forward](figures/case2_forward_iv.png) | ![ideality](figures/case2_ideality.png) | ![cv](figures/case3_cv_mott_schottky.png) | ![mesh](figures/case4_mesh_convergence.png) |

| Equilibrium profiles | Band diagram | Sesame cross-check | Realistic vs ideal Si |
|---|---|---|---|
| ![profiles](figures/case1_profiles.png) | ![bands](figures/case1_band_diagram.png) | ![cross](figures/case5_sesame_crosscheck.png) | ![realistic](figures/case6_realistic_iv.png) |

---

## Quickstart

```bash
pip install -r requirements.txt

python scripts/run_all.py              # all six verification cases, ≈18 s -> data/ figures/ + summary table
python scripts/run_all.py --fast       # skip the mesh study and the Sesame cross-check, ≈8 s
python scripts/run_all.py --only 2 5   # selected cases only
python -m pytest -q                    # 93 tests, ≈10 s
```

Each case also runs standalone and prints its metrics, e.g. `python scripts/case1_equilibrium.py`.
`pip install -e .` works too (`src/p5dd`), but the scripts and the test suite insert `src/`
into `sys.path` themselves, so a plain clone runs everything with no install step.

Products: `data/metrics_caseN.json` (structured metrics — every number quoted here is read
back from these files), `data/caseN_*.csv` (per-bias / per-node data), `figures/caseN_*.png`
(12 figures), `data/summary.json` (rollup with per-case wall clock).

## Using the library

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from p5dd import DDSolver, reference_device, realistic_device

dev = reference_device(n_nodes=1601)   # constant mobility, no BGN -> comparable to analytics/Sesame
sol = DDSolver(dev, tol=1e-8)

u, rep = sol.solve_point(0.0)          # rep: iterations / residual / converged / backtracks
st = sol.physical_state(u)             # x, psi, n, p, E, Jn, Jp, J, Ec, Ev, Ei, Ef (dimensional)
Jn, Jp = sol.currents(u)               # link currents [A/cm^2]
print(sol.total_current(u))            # terminal current [A/cm^2]

U, J, reps = sol.iv_sweep(np.arange(0.0, 0.71, 0.02))   # adaptive bias continuation
```

Sign convention: the n-side is at x = 0, so **forward bias is a negative terminal current**
(conventional current flows from the p contact to the n contact inside the device).
`theory.shockley()` uses the same convention and can be compared point by point.

Analytic references and post-processing are importable on their own:
`built_in_potential`, `depletion`, `minority_transport`, `saturation_current`, `shockley`,
`ideality_factor`, `peak_field`, `depletion_charge`, `capacitance`, `mott_schottky`,
`expected_capacitance`, `net_recombination_rate`.

---

## How it works, and why it is exact at equilibrium

**Unknowns.** Dual-potential (Slotboom) variables `psi, phi_n, phi_p` in units of V_t, with
`n = r_ni·exp(psi − phi_n)` and `p = r_ni·exp(phi_p − psi)`. Densities are carried in
exponential form, so they are **strictly positive** — no negative-density or underflow
failure at large bias. Ohmic contacts pin `phi_n`, `phi_p` and `psi` together, which is
equivalent to pinning both carrier densities, but every boundary row is O(1) instead of 1e-15.

**Fluxes: Scharfetter–Gummel links.** A SG link is the exact integral of the
drift-diffusion ODE across one link assuming a linear potential drop inside it; its
coefficient is the Bernoulli function. The property that matters: when the quasi-Fermi level
is constant, the discrete flux is **identically zero**, so zero current at equilibrium and
`np = n_i²` are analytic properties of the scheme rather than something tuned. That is why
case 1 reaches 1e-14.

**Band-gap narrowing requires the generalised SG.** If `n_i` varies with position, the
equilibrium density ratio becomes `n_R/n_L = (n_iR/n_iL)·exp(Δpsi)`, so a bare `Δpsi` no
longer reproduces the Boltzmann factor: a naive scheme leaves a spurious flux at the doping
step and breaks `np = n_i(x)²`. This implementation takes the Bernoulli argument as
**Δpsi + Δln n_i for electrons, Δpsi − Δln n_i for holes**
([src/p5dd/solver.py](src/p5dd/solver.py)), and `B(−x)/B(x) = e^x` then proves both link
currents vanish for **any** `n_i(x)`. Regression test in
[tests/test_equilibrium.py](tests/test_equilibrium.py): the realistic device's equilibrium
violation goes from 0.125 to 8.0e-14 and Newton from 5 to 4 iterations; without band-gap
narrowing the generalisation is provably inert (`_dln_ni == 0` node by node), so the
reference device is bit-for-bit unchanged.

**Mesh.** Nodes are placed at equal arc length in the `1/h` metric: 4 nm near the junction,
50 nm in the neutral regions, neighbour ratio < 1.15 (no abrupt grading), and doubling the
node count halves both h_min and h_max — which is what makes case 4's observed order a clean
second order.

**Linear algebra.** Analytic sparse Jacobian (per-link Bernoulli derivatives + SRH chain
rule), `scipy.sparse.linalg.spsolve`; damped Newton with an infinity-norm step cap
`du_cap = 4 V_t` so the first step cannot push the Slotboom exponent out of range, plus
backtracking on the same directional residual.

**Stopping criterion.** Term-scaled weighted infinity norm `max|F_i|/s_i`, where `s_i` is the
sum of the magnitudes of that equation's own terms (floor 1) and the weights are frozen
within a Newton step. This makes the criterion mesh-independent — an absolute residual
amplified by 1/h² cannot express "how well equilibrium is satisfied". The achievable floor
is set by the sparse LU (≈3e-9 at 1601 nodes), i.e. terminal-current error < 1e-6 A/cm².

**Bias sweeps.** Adaptive continuation halves the step on failure (down to 1e-13 V), so every
point from 0 to 0.70 V converges in ≤ 4 Newton iterations with no manual tuning; tightening
`du_cap` to 0.5 still converges to the same state.

---

## Verification design: three evidence layers, and what each cannot show

The reusable output here is the **framework**, not a device result, so every claim states its
layer. Rules: [reports/technical-report.md](reports/technical-report.md) §1.2–§1.5, checked
criterion by criterion in [reports/acceptance-checklist.md](reports/acceptance-checklist.md).

| layer | evidence | status | what it does **not** prove |
|---|---|---|---|
| **L1** | analytic invariants: zero equilibrium current, flat quasi-Fermi levels, `np = n_i²`, V_bi / W / E_max against closed forms | **mandatory floor** | model completeness — it certifies the discretisation, not the physics you chose |
| **L2** | mesh independence over 5 grids, observed order ≈ 2 | **mandatory floor** | that the converged limit is the *right* answer |
| **L3** | terminal current vs NIST Sesame 2.0 on identical inputs, 23 bias points | bonus | any independent ground truth — **independent code ≠ independent validation chain**; and because both codes share the same model set, this layer has zero power over model completeness |

Gate: an L1 or L2 failure means the deliverable fails and L3 cannot compensate; L3 missing
alone means "external consistency unproven", not "rejected". A cross-check is only
meaningful under stated premises (same mesh, doping, material constants, boundary conditions,
tolerances, model set). Here the contact treatment (Dirichlet dual potentials vs `contact_S`
surface recombination velocities) and the residual norm/tolerance are only approximately
equivalent — which is precisely where the observed 2.85e-4 deviation sits (near-zero current,
small reverse bias). Premises P1–P6 are declared item by item in §1.3.

Skips are never counted as passes: without `reference/`, the suite reports 89 passed /
4 skipped and case 5 vanishes; that state is labelled "not executed", not "passed".

### Third-party code: version lock and raw data

So L3 does not depend on one machine, everything needed to re-check it lives in this
repository ([reference/PROVENANCE.md](reference/PROVENANCE.md)):

| item | value |
|---|---|
| reference code | **NIST Sesame 2.0** (`reference/sesame/sesame/_version.py`, and `sesame.__version__` at run time) |
| how obtained | vendored, **not pip-installed** (`pip show sesame` → not found), so no version ambiguity for a reviewer |
| modifications | none in any retained file; `LICENSE.rst` / `AUTHORS.rst` and per-file copyright headers verbatim. `doc/`, `examples/`, `test_suite/` (no solver code) were removed to keep the repository near 4 MB |
| integrity digest | 24 shipped `.py` files → SHA-256 `fa1698474ee02d89c632761742c122df5e8f3a13cbe0f68fdcf9f8a481f1ee30`; complete pre-trim copy (44 `.py`) → `b1b093cf389a487b8d249b9b782e6343ca17aeb36272c94054ccca851cca9cb0` |
| raw point data | [data/case5_sesame_crosscheck.csv](data/case5_sesame_crosscheck.csv) — 23 rows (`V_V`, `J_self_A_cm-2`, `J_sesame_A_cm-2`, `rel_dev`), so the deviation is recomputable without running any solver |
| dependency isolation | the solver core never imports Sesame; only `devices.sesame_system()` and the case-5 script do |

```bash
# check the vendored reference against the digest above
cd reference/sesame && find . -name "*.py" -type f | sort | xargs cat | sha256sum
```

---

## Physics findings from this device (each with its applicability limit)

Wording rule: **verified** = a measured value exists in `data/metrics_caseN.json`;
**extrapolation** = same-family evidence or a mechanism argument only, with the verification
path named. All six items are measurements on *this* Si abrupt junction at *these* parameters
and are not general device claims.

1. **The textbook generation current is an upper bound here, not a prediction.**
   `q·n_i·W/(τ_n+τ_p)` assumes the whole depletion region generates at the limiting rate; only
   the **doubly-depleted window** (where n and p are both below n_i) actually does, because
   Boltzmann tails keep the majority carrier above n_i over most of the region. Measured
   reverse current: 11 % / 23 % / 36 % of that bound at −0.10 / −0.30 / −0.70 V, and the
   effective generation width inferred from `|J|(τ_n+τ_p)/(q·n_i)` exceeds the window width by
   a factor of only 1.06 — the window width, not the depletion width, sets the current.
   *(Verified for τ = 10 ns, W ≈ 0.33 µm, 1e16–1e17 cm⁻³; other lifetimes/dopings untested.)*
2. **The applied bias does not all fall across the junction.** At 0.70 V the neutral-region
   drop is 49.2 mV, within 11 % of the `|J|·R_sheet` estimate (R_sheet = 8.08e-3 Ω·cm²), while
   at ≤ 0.20 V it is < 1e-5 V. Regressing ln J against the **junction** voltage V_j is what
   recovers the thermal voltage (3.1 % error); regressing against V inflates the ideality
   factor at high bias (2.23 here).
3. **The three ideality regimes emerge on their own**: ≈1.58 where SRH recombination dominates
   at low bias, ≈1.12 in the diffusion-dominated middle, > 2 once series resistance takes over.
   No piecewise rule exists anywhere in the solver — one equation set, three behaviours.
4. **Band-gap narrowing pulls the built-in potential down**: ΔE_g ≈ 24 meV on the 1e17 cm⁻³
   side, n_i ×1.60, V_bi 0.78013 → 0.75880 V (−21.3 mV), J_s ×1.90, J(0.60 V) ×1.54. Model
   sensitivity (a physics result), not solver error.
5. **`np = n_i²` holds to 6e-14 even inside the space-charge region** — a direct consequence of
   the zero-flux property, not an imposed constraint.
6. **Photogeneration runs through the same entry point** (`generation=`): J(0) > 0, J_sc
   matches `qG(W + L_n + L_p)`, total current stays continuous. Evidence about the framework,
   deliberately not expanded into an optoelectronic device result.

---

## Scope boundary: not implemented, not verified

Framework-level deliverables and device-level results are counted separately, and this
repository makes no capability claim for the following, none of which was built or tested:

MOS / MOSFET capacitance structures · pin junctions · heterojunctions · graded-doping
junctions · transient and switching · AC small-signal admittance · 2-D devices · high
injection (V approaching V_bi).

`Device.cdot` accepts an arbitrary doping profile, `r_ni` may vary node by node and
`generation(x)` is injectable, so these are **structurally supported** — but "structurally
supported" ≠ "numerically verified". Each extrapolation and its verification path (inversion
charge and interface states for MOS; band offsets and fixed interface charge for
heterojunctions; the doping-gradient effect on the depletion-edge criterion for graded
junctions) is listed in [reports/technical-report.md](reports/technical-report.md) §1.5.2 and
§8.2.

Other declared limits: steady state, 1-D, uniform lifetime, a single mid-gap SRH level, no
Auger / tunnelling / Schottky recombination, no displacement current (so no AC and no C-V
frequency dispersion — case 3's capacitance is a quasi-static dQ/dV); abrupt doping at x_j;
the 0.70 V sweep stays on the weak-injection side; the Sesame comparison covers terminal
current only (comparing fields and densities would require reading Sesame's solved state);
and since the residual floor is ≈3e-9, every *relative* metric is gated on |J| > 1e-3 A/cm² —
below that you are reading solver noise, not physics.

The second route named in the original project brief (FEniCS finite elements) was not used:
1-D finite volume with SG fluxes already delivers provable equilibrium exactness and a clean
second order, with numpy/scipy only and no build chain, which fits the CPU-only,
one-command-reproduction constraint better.

## Layout

```
src/p5dd/          constants · bernoulli · material (Arora / Slotboom / SRH) · device (graded
                   mesh, scaling, initial guess) · devices (reference / realistic / Sesame
                   adapters) · solver (residual + analytic Jacobian + damped Newton +
                   continuation) · theory (V_bi, depletion, Shockley, ideality) · analysis
                   (peak field, depletion charge, C = dQ/dV, Mott-Schottky) · viz (Agg styling)
scripts/           case1..case6 + run_all.py
tests/             93 tests, session-scoped solved fixtures
reports/           technical-report.md · acceptance-checklist.md
data/ figures/     committed metrics, per-point CSV, 12 PNG
reference/         vendored Sesame 2.0 (trimmed; retained files unmodified) + PROVENANCE.md
.github/workflows/ CI: test suite + fast case run on Python 3.10 / 3.12 / 3.13
```

## License

MIT for the code written in this repository (`src/`, `scripts/`, `tests/`, `reports/`,
`data/`, `figures/`). `reference/sesame/` remains under its own 3-clause BSD license
("Copyright 2017 University of Maryland"), redistributed unmodified together with
`LICENSE.rst`, `AUTHORS.rst` and [reference/PROVENANCE.md](reference/PROVENANCE.md).
