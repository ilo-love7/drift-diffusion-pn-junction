"""Forward-bias behaviour: sign, monotonicity, continuity, exponential slope.

These are the checks that a drift-diffusion diode is actually being simulated
rather than a curve fitted to one.  Every threshold is set from what the 801-node
mesh delivers, with at least twice the observed margin, so a real regression (a
flipped sign, a lost factor, a broken link flux) fails immediately while float
noise does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import (DDSolver, built_in_potential, ideality_factor,
                  minority_transport, reference_device, shockley)
from p5dd.constants import Q_E

V_SWEEP = np.round(np.arange(0.0, 0.6001, 0.02), 4)


def _sheet_resistance(dev) -> float:
    """Neutral-region Ohmic sheet resistance [ohm cm^2], majority carriers."""
    mu_n = float(dev.mat.mu_n(np.full(1, dev.Nd))[0])
    mu_p = float(dev.mat.mu_p(np.full(1, dev.Na))[0])
    return float(dev.x_j / (Q_E * mu_n * dev.Nd)
                 + (dev.x[-1] - dev.x_j) / (Q_E * mu_p * dev.Na))


@pytest.fixture(scope="module")
def forward():
    dev = reference_device(n_nodes=801)
    sol = DDSolver(dev)
    U, J, reps = sol.iv_sweep(V_SWEEP)
    states = [sol.physical_state(u) for u in U]
    j = int(np.argmin(np.abs(dev.x - dev.x_j)))
    v_j = np.array([st["phi_p"][j] - st["phi_n"][j] for st in states])
    return {"dev": dev, "sol": sol, "U": U, "J": J, "reps": reps,
            "states": states, "v_j": v_j, "j": j}


def test_every_bias_point_converges(forward):
    assert all(r.converged for r in forward["reps"])
    assert max(r.iterations for r in forward["reps"]) <= 6
    assert max(r.residual for r in forward["reps"]) < 1e-7


def test_forward_current_is_negative_and_monotonic(forward):
    J = forward["J"]
    assert np.all(J[1:] < 0.0)
    assert np.all(np.diff(J) < 0.0)          # |J| grows with bias
    assert abs(J[0]) < 1e-18                 # and is exactly zero at V = 0


def test_current_is_continuous_along_the_device(forward):
    """div(Jn + Jp) = 0: the terminal current equals every internal link sum.

    Only points above 1e-3 A/cm^2 are measured relatively.  Below that the
    absolute spread is already at the sparse-LU floor (a few 1e-9 of the
    dimensionless residual, i.e. ~1e-6 A/cm^2 once scaled by Jscale = 559
    A/cm^2), so a relative measure would only report solver round-off.
    """
    for st, Jt in zip(forward["states"], forward["J"]):
        assert float(np.ptp(st["J"])) < 1e-5
        if abs(Jt) > 1e-3:
            assert float(np.ptp(st["J"])) / abs(Jt) < 1e-4


def test_potential_drop_equals_builtin_minus_bias(forward):
    dev, st = forward["dev"], forward["states"]
    vbi = built_in_potential(dev)
    for V, s in zip(V_SWEEP, st):
        assert s["psi"][0] - s["psi"][-1] == pytest.approx(vbi - V, abs=1e-12)


def test_applied_bias_falls_on_the_junction_at_low_current(forward):
    """Below 1e-4 A/cm^2 the neutral bulk is equipotential to a few microvolts."""
    weak = (V_SWEEP <= 0.20)
    assert np.all(np.abs(V_SWEEP[weak] - forward["v_j"][weak]) < 1e-5)


def test_remaining_drop_is_the_majority_ohmic_voltage(forward):
    """V - V_j must equal |J| times the analytic neutral-region sheet resistance.

    This ties the transport solver to an entirely independent resistance
    calculation and shows the two curves cross over where the ideality factor
    turns back up: at high injection the junction itself, not the bulk, limits.
    """
    dev = forward["dev"]
    rs = _sheet_resistance(dev)
    drop = V_SWEEP - forward["v_j"]
    ohmic = np.abs(forward["J"]) * rs
    sel = (V_SWEEP >= 0.50) & (V_SWEEP <= 0.58)
    rel = np.abs(drop[sel] - ohmic[sel]) / drop[sel]
    assert np.max(rel) < 0.15
    assert rs < 1.0e-2                       # a well-doped, long diode


def test_exponential_slope_is_the_thermal_voltage(forward):
    """ln|J| vs the *junction* voltage is locally a straight line of slope 1/Vt.

    Comparing against the applied V rather than V_j mixes in the series
    resistance; against V_j the residual curvature above 0.55 V is the genuine
    high-injection correction and stays inside a few percent.
    """
    dev = forward["dev"]
    lnJ = np.log(np.abs(forward["J"]))
    sel = (V_SWEEP >= 0.55) & (V_SWEEP <= 0.60)
    slope = np.polyfit(forward["v_j"][sel], lnJ[sel], 1)[0]
    assert 1.0 / slope == pytest.approx(dev.mat.Vt, rel=0.05)


def test_current_stays_within_a_factor_three_of_shockley(forward):
    """Above 0.44 V the diffusion current dominates and the ideal diode law
    describes the magnitude; below it, space-charge recombination adds a term
    the Shockley formula cannot represent."""
    dev = forward["dev"]
    sel = (V_SWEEP >= 0.44) & (V_SWEEP <= 0.60)
    ratio = np.abs(forward["J"][sel] / shockley(dev, V_SWEEP[sel]))
    assert np.all(ratio > 0.5) and np.all(ratio < 2.5)


def test_ideality_factor_spans_the_expected_regimes(forward):
    """Low bias is SRH-dominated (1 < n_id < 2), mid bias diffusion (~1).

    The V = 0 point is dropped: ln|J| is undefined for the numerical zero, so
    its secant slope carries no information.
    """
    dev = forward["dev"]
    v = V_SWEEP[1:]
    n_id = ideality_factor(v, forward["J"][1:], dev.mat.T)
    low = n_id[(v >= 0.08) & (v <= 0.24)]
    mid = n_id[(v >= 0.48) & (v <= 0.56)]
    assert np.all(low > 1.5) and np.all(low < 2.0)     # recombination, not diffusion
    assert np.all(mid > 1.0) and np.all(mid < 1.20)    # diffusion-dominated
    assert n_id.min() > 0.8 and n_id.max() < 3.0


def test_minority_carriers_decay_over_a_diffusion_length(forward):
    """Measured excess-electron tail on the p side vs L_n = sqrt(Dn tau_n)."""
    dev, st = forward["dev"], forward["states"][-1]
    tr = minority_transport(dev)
    x = st["x"]
    ni = dev.r_ni[-1] * dev.Nref
    n_equilibrium = ni ** 2 / dev.Na      # minority electron density on the p side
    # exponential fit of the tail 20 um beyond the space-charge region
    tail = (x > dev.x_j + 20e-4) & (x < dev.x_j + 40e-4)
    xs = x[tail]
    ln_ex = np.log(np.maximum(st["n"][tail] - n_equilibrium, 1e-6))
    slope = np.polyfit(xs, ln_ex, 1)[0]
    measured_L = -1.0 / slope if slope < 0 else np.inf
    assert measured_L == pytest.approx(tr["Ln"], rel=0.25)
