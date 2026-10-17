"""Profile-level agreement with the vendored NIST Sesame (extends case 5).

Case 5 compares the terminal current, a single scalar.  These tests compare the
*solved state*: densities, electrostatic potential and field at every shared
mesh node.  Sesame's converged per-bias state is read back from the files its
own ``IVcurve`` writes, so no bias continuation is re-implemented here and the
comparison keeps the same premises as case 5.

The tests skip cleanly when the vendored copy is absent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from p5dd import DDSolver, reference_device, sesame_system

VOLTAGES = np.round(np.concatenate([
    [-0.60, -0.40, -0.20, -0.10, -0.05],
    np.arange(0.0, 0.60, 0.05),
    np.arange(0.60, 0.7001, 0.02),
]), 4)
N_NODES = 801
BIASES = (0.00, 0.60)
AFFINITY = 4.05           # [eV] handed to Sesame by devices.sesame_system()
DENSITY_FLOOR = 1.0e6     # [cm^-3]
FIELD_FLOOR = 1.0e3       # [V/cm]
BULK_EDGE = 5.0e-4        # [cm]


@pytest.fixture(scope="module")
def profiles():
    """Node-wise states from both codes at the tested biases."""
    sesame = pytest.importorskip("sesame", reason="vendored Sesame not available")
    dev = reference_device(n_nodes=N_NODES)
    sol = DDSolver(dev)
    sys_ = sesame_system(dev)

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp, "ses").as_posix()
        J_ses = np.asarray(sesame.IVcurve(sys_, list(VOLTAGES), prefix,
                                          verbose=False, tol=1e-9), dtype=float)
        for bias in BIASES:
            idx = int(np.argmin(np.abs(VOLTAGES - bias)))
            s2, res = sesame.load_sim(f"{prefix}_{idx}.gzip")
            az = sesame.Analyzer(s2, res)
            u, rep = sol.solve_point(bias)
            assert rep.converged
            st = sol.physical_state(u)
            x = np.asarray(dev.x, dtype=float)
            out[bias] = {
                "x": x,
                "n_self": st["n"], "p_self": st["p"], "psi_self": st["psi"], "E_self": st["E"],
                "n_ses": s2.scaling.density * np.asarray(az.electron_density()).ravel(),
                "p_ses": s2.scaling.density * np.asarray(az.hole_density()).ravel(),
                "psi_ses": s2.scaling.energy * np.asarray(res["v"]).ravel(),
                "E_ses": -np.diff(s2.scaling.energy * np.asarray(res["v"]).ravel()) / np.diff(x),
                "J_ses": float(J_ses[idx]) * sys_.scaling.current,
                "J_self": float(sol.total_current(u)),
            }
    out["dev"] = dev
    return out


def _rel(a, b, floor=0.0):
    """Relative deviation of a against b on the full node array; nodes under
    ``floor`` are excluded from the comparison, not silently counted as zero error."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    strong = np.minimum(np.abs(a), np.abs(b)) > floor
    rel = np.zeros(a.shape, dtype=float)
    rel[strong] = np.abs(a[strong] - b[strong]) / np.abs(b[strong])
    return rel, strong


def test_the_two_codes_share_one_mesh(profiles):
    assert np.array_equal(profiles[BIASES[0]]["x"], np.asarray(profiles["dev"].x))


def test_potential_offset_is_the_intrinsic_level_identity(profiles):
    """Sesame references v to the intrinsic level, so psi differs by exactly
    -(affinity + Eg/2 + (Vt/2)ln(Nc/Nv)) - a constant, at every bias."""
    m = profiles["dev"].mat
    expected = -(AFFINITY + m.Eg / 2.0 + 0.5 * m.Vt * np.log(m.Nc / m.Nv))
    offsets = [p["psi_ses"][0] - p["psi_self"][0] for p in
               (profiles[b] for b in BIASES)]
    assert max(abs(o - expected) for o in offsets) < 1e-9
    assert abs(offsets[0] - offsets[1]) < 1e-12          # pure gauge, no bias drift


def test_potential_profiles_agree_after_gauge_removal(profiles):
    for b in BIASES:
        p = profiles[b]
        d = (p["psi_ses"] - p["psi_ses"][0]) - (p["psi_self"] - p["psi_self"][0])
        assert np.max(np.abs(d)) < 1e-6                  # measured ~1e-10 .. 1e-8 V


def test_density_profiles_agree_at_equilibrium(profiles):
    p = profiles[0.00]
    for key in ("n", "p"):
        rel, _ = _rel(p[f"{key}_ses"], p[f"{key}_self"], DENSITY_FLOOR)
        assert rel.max() < 1e-6                          # measured 4.3e-9


def test_density_profiles_agree_in_the_bulk_under_forward_bias(profiles):
    """Forward bias is where the two contact treatments differ most, so the bulk
    band is the part that has to stay in agreement."""
    p = profiles[0.60]
    x = p["x"]
    bulk = (x > BULK_EDGE) & (x < x[-1] - BULK_EDGE)
    for key in ("n", "p"):
        rel, strong = _rel(p[f"{key}_ses"], p[f"{key}_self"], DENSITY_FLOOR)
        assert rel[strong & bulk].max() < 1e-2           # measured 2.2e-5 / 3.0e-6
        assert rel[strong].max() < 1e-2                  # measured 8.5e-4 / 5.1e-6


def test_forward_disagreement_is_contact_localised(profiles):
    """Where the largest density error sits is itself evidence: at +0.60 V the
    worst electron error is at the far contact, not in the bulk."""
    p = profiles[0.60]
    rel_n, strong = _rel(p["n_ses"], p["n_self"], DENSITY_FLOOR)
    x = p["x"]
    bulk = (x > BULK_EDGE) & (x < x[-1] - BULK_EDGE)
    assert rel_n[strong].max() > 10 * rel_n[strong & bulk].max()


def test_field_profiles_agree_where_the_field_is_resolved(profiles):
    for b in BIASES:
        p = profiles[b]
        rel = np.abs(p["E_ses"] - p["E_self"]) / np.maximum(np.abs(p["E_self"]), FIELD_FLOOR)
        assert rel.max() < 1e-2                          # measured <= 3.3e-7
        assert np.max(np.abs(p["E_ses"] - p["E_self"])) < 1e-2   # [V/cm], measured 8.0e-4


def test_terminal_current_still_agrees(profiles):
    """Case 5's scalar result must survive the switch to per-point solving."""
    for b in BIASES:
        p = profiles[b]
        if abs(p["J_self"]) > 1e-3:
            assert abs(p["J_ses"] - p["J_self"]) / abs(p["J_self"]) < 5e-3
