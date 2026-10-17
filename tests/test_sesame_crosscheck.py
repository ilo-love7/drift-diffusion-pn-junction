"""Independent-code agreement: the self-built solver vs the vendored NIST Sesame.

Sesame shares no code with this project.  Handing both the identical mesh,
doping, mobilities, lifetimes, density of states, band gap, temperature and
Ohmic boundary conditions means any difference is a difference of numerics, so
this is the strongest available check on the solver - and the only one that does
not assume the analytic diode theory is right.

The test skips cleanly when the vendored copy is absent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from p5dd import DDSolver, reference_device, sesame_system

# Sesame runs its own Newton continuation along the list it is handed, so the
# sweep must already be monotone and fine enough for a coarse-step divergence -
# the same sequence the case-5 script uses.
VOLTAGES = np.round(np.concatenate([
    [-0.60, -0.40, -0.20, -0.10, -0.05],
    np.arange(0.0, 0.60, 0.05),
    np.arange(0.60, 0.7001, 0.02),
]), 4)
N_NODES = 801


@pytest.fixture(scope="module")
def comparison():
    sesame = pytest.importorskip("sesame", reason="vendored Sesame not available")
    dev = reference_device(n_nodes=N_NODES)
    sol = DDSolver(dev)
    _, J_self, reps = sol.iv_sweep(VOLTAGES)
    assert all(r.converged for r in reps)

    sys_ = sesame_system(dev)
    with tempfile.TemporaryDirectory() as tmp:
        J_ses = sesame.IVcurve(sys_, list(VOLTAGES), Path(tmp, "ses").as_posix(),
                               verbose=False, tol=1e-9)
    J_ses = np.asarray(J_ses, dtype=float) * sys_.scaling.current
    assert np.all(np.isfinite(J_ses))
    return {"J_self": J_self, "J_sesame": J_ses, "dev": dev}


def _strong(comparison):
    """Bias points where a *relative* current comparison is meaningful.

    At V = 0 both codes return their linear-solver noise floor (~1e-20 A/cm^2),
    whose ratio is arbitrary; that point is checked for absolute size instead.
    """
    return np.abs(comparison["J_self"]) > 1e-9


def test_both_zero_at_equilibrium(comparison):
    k = int(np.argmin(np.abs(VOLTAGES)))
    assert abs(comparison["J_self"][k]) < 1e-18
    assert abs(comparison["J_sesame"][k]) < 1e-18


def test_terminal_currents_agree(comparison):
    """Three digits everywhere the current is resolvable."""
    a, b = comparison["J_self"], comparison["J_sesame"]
    strong = _strong(comparison)
    assert strong.sum() >= len(VOLTAGES) - 2
    rel = np.abs(b - a) / np.abs(a)
    assert np.max(rel[strong]) < 5e-3
    assert np.max(rel[np.abs(a) > 1e-2]) < 1e-4     # forward branch: 4 digits
    decades = np.abs(np.log10(np.abs(b[strong])) - np.log10(np.abs(a[strong])))
    assert np.max(decades) < 5e-3


def test_both_follow_the_same_curve_shape(comparison):
    """Same sign convention and the same monotone sweep, not just similar numbers."""
    a, b = comparison["J_self"], comparison["J_sesame"]
    strong = _strong(comparison)
    assert np.all(np.sign(a[strong]) == np.sign(b[strong]))
    assert np.all(np.diff(a) < 0.0) and np.all(np.diff(b) < 0.0)
