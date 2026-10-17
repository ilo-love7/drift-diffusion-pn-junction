"""Reverse bias and small-signal C-V: depletion theory must fall out.

The solver is never told about capacitance.  Q = eps*E_max comes from Gauss' law
applied to the computed field, C = dQ/dV from the sweep, and only then is the
result compared with the abrupt-junction formulas - so agreement is a statement
about the discretisation, not about a shared helper function.
"""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import (DDSolver, built_in_potential, capacitance, depletion,
                  depletion_charge, expected_capacitance, mott_schottky,
                  net_recombination_rate, peak_field, reference_device)
from p5dd.constants import Q_E

V_SWEEP = -np.round(np.arange(0.0, 0.6001, 0.02), 4)


@pytest.fixture(scope="module")
def reverse():
    dev = reference_device(n_nodes=801)
    sol = DDSolver(dev)
    U, J, reps = sol.iv_sweep(V_SWEEP)
    states = [sol.physical_state(u) for u in U]
    q = np.array([depletion_charge(dev, st) for st in states])
    c = capacitance(V_SWEEP, q)
    return {"dev": dev, "sol": sol, "J": J, "reps": reps, "states": states,
            "charge": q, "cap": c}


def test_sweep_converges_everywhere(reverse):
    assert all(r.converged for r in reverse["reps"])
    assert max(r.residual for r in reverse["reps"]) < 1e-7


def test_reverse_current_is_positive_and_small(reverse):
    J = reverse["J"]
    assert np.all(J[1:] > 0.0)                  # opposite sign to forward bias
    assert J[-1] < 1e-3                          # six orders below the forward current
    assert J[-1] / J[10] < 5.0                   # and only slowly varying: near-saturated


def test_reverse_current_is_far_below_the_forward_current(reverse):
    dev = reverse["dev"]
    sol = DDSolver(dev)
    u, _ = sol.solve_point(0.60)
    fwd = abs(sol.total_current(u))
    assert abs(reverse["J"][-1]) / fwd < 1e-3


def test_depletion_capacitance_matches_the_abrupt_junction_formula(reverse):
    dev = reverse["dev"]
    expected = expected_capacitance(dev, V_SWEEP)
    rel = np.abs(reverse["cap"] - expected) / expected
    assert np.max(rel) < 0.03


def test_capacitance_falls_monotonically_with_reverse_bias(reverse):
    assert np.all(np.diff(reverse["cap"]) < 0.0)


def test_capacitance_derived_depletion_width(reverse):
    """W = eps/C against the analytic width, at zero and deep reverse bias."""
    dev = reverse["dev"]
    for k, V in ((0, 0.0), (len(V_SWEEP) - 1, -0.60)):
        w_num = dev.mat.eps / reverse["cap"][k]
        w_ref = depletion(dev, V)["W"]
        assert w_num == pytest.approx(w_ref, rel=0.03)


def test_mott_schottky_recovers_doping_and_builtin_voltage(reverse):
    dev = reverse["dev"]
    fit = mott_schottky(V_SWEEP, reverse["cap"], dev.mat.eps)
    # 1/C^2 is a straight line in V whose slope encodes the doping *sum*
    n_harmonic = 1.0 / (1.0 / dev.Na + 1.0 / dev.Nd)
    assert fit["doping_eff"] == pytest.approx(n_harmonic, rel=0.10)
    assert fit["Vbi_fit"] == pytest.approx(built_in_potential(dev), abs=0.06)
    assert fit["R2"] > 0.999
    assert fit["slope"] < 0.0                    # C shrinks as V goes reverse


def test_srh_generation_saturates_at_the_theoretical_limit(reverse):
    """max |R| = n_i / (tau_n + tau_p), reached where both carriers are depleted.

    The textbook q*ni*W/(tau_n+tau_p) is an upper bound, because generation only
    runs at the theoretical rate where *both* n and p fall below n_i; the
    measured current is therefore well below it, and the peak rate itself hits
    the bound almost exactly.
    """
    dev = reverse["dev"]
    ni = dev.r_ni * dev.Nref
    rate = net_recombination_rate(dev, reverse["states"][-1])
    limit = ni / (dev.mat.tau_n + dev.mat.tau_p)
    ratio = float(np.max(-rate) / np.max(limit))
    assert 0.98 < ratio <= 1.002

    w_ref = depletion(dev, V_SWEEP[-1])["W"]
    bound = Q_E * float(ni.min()) * w_ref / (dev.mat.tau_n + dev.mat.tau_p)
    assert 0.1 < abs(reverse["J"][-1]) / bound < 0.6


def test_effective_generation_width_is_narrower_than_depletion_width(reverse):
    """The doubly-depleted window (np < ni^2 with both carriers below ni) is the
    region that actually generates, and it is narrower than the depletion width.
    """
    dev, st = reverse["dev"], reverse["states"][-1]
    ni = dev.r_ni * dev.Nref
    doubly = (st["n"] < ni) & (st["p"] < ni)
    x = st["x"]
    w_window = float(x[doubly][-1] - x[doubly][0]) if doubly.sum() > 1 else 0.0
    w_ref = depletion(dev, V_SWEEP[-1])["W"]
    assert 0.2 * w_ref < w_window < 0.8 * w_ref


def test_peak_field_grows_with_reverse_bias(reverse):
    e = np.array([peak_field(st) for st in reverse["states"]])
    assert np.all(np.diff(e) > 0.0)
    # Emax = sqrt(2 q (Vbi - V) NA ND/(NA+ND)) / ... check the sqrt law directly
    dev = reverse["dev"]
    vbi = built_in_potential(dev)
    scaled = e / np.sqrt(np.maximum(vbi - V_SWEEP, 1e-9))
    assert np.ptp(scaled) / scaled.mean() < 0.02
