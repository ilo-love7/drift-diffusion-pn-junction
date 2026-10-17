"""Solver mechanics: continuation, residual weighting and the generation term."""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import DDSolver, depletion, minority_transport, reference_device
from p5dd.constants import Q_E

G_PHOTON = 1.0e19        # uniform generation, pairs cm^-3 s^-1


@pytest.fixture(scope="module")
def dev():
    return reference_device(n_nodes=401)


def test_continuation_and_cold_start_reach_the_same_solution(dev):
    """Warm-started sweep and a from-scratch solve agree to solver precision."""
    sol = DDSolver(dev, tol=1e-7)
    _, J_sweep, reps = sol.iv_sweep(np.round(np.arange(0.0, 0.6001, 0.05), 4))
    assert all(r.converged for r in reps)
    u_cold, rep_cold = sol.solve_point(0.60)
    assert rep_cold.converged
    J_cold = sol.total_current(u_cold)
    assert abs(J_cold - J_sweep[-1]) / abs(J_sweep[-1]) < 1e-3


def test_residual_weights_are_term_scaled(dev):
    """Each equation is measured against the sum of its own term magnitudes."""
    sol = DDSolver(dev)
    u = dev.initial_state(0.35)
    F, _, s = sol.residual(u, 0.35, want_jac=True, want_scales=True)
    assert s.shape == F.shape
    assert np.all(s >= 1.0)                     # the floor keeps it a norm
    assert np.all(np.isfinite(s))
    assert sol.weighted_norm(F, s) == pytest.approx(np.max(np.abs(F) / s))
    # the scaling is not a formality: the raw residual spans many decades
    assert np.max(np.abs(F)) / np.min(np.abs(F)[np.abs(F) > 0]) > 1e6


def test_high_bias_point_is_reachable(dev):
    """0.70 V is 90 % of the built-in voltage; the damped Newton plus bias
    continuation is what makes that point solvable from an analytic guess."""
    sol = DDSolver(dev, tol=1e-7)
    u0 = dev.initial_state(0.0)
    u, rep = sol.solve_point(0.70)
    assert rep.converged and np.all(np.isfinite(u))
    n, p = dev.densities(u)
    assert np.all(n > 0.0) and np.all(p > 0.0)
    assert abs(sol.total_current(u)) > 0.5      # a real forward current [A/cm^2]
    # the equilibrium guess is not itself a solution: it leaves a large residual
    assert np.max(np.abs(sol.residual(u0, 0.70))) > 1e-3


def test_damped_newton_reaches_the_same_state_with_a_tight_step_cap(dev):
    """``du_cap`` is in units of the thermal voltage, since every unknown is a
    potential in those units; capping it changes the path, not the solution."""
    sol = DDSolver(dev, tol=1e-7, du_cap=0.5)
    base = DDSolver(dev, tol=1e-7)
    u, rep = sol.solve_point(0.60)
    u_ref, rep_ref = base.solve_point(0.60)
    assert rep.converged and rep_ref.converged
    assert abs(sol.total_current(u) - base.total_current(u_ref)) \
        < 1e-3 * abs(base.total_current(u_ref))


@pytest.fixture(scope="module")
def photovoltaic(dev):
    sol = DDSolver(dev, generation=lambda x: G_PHOTON, tol=1e-7)
    U, J, reps = sol.iv_sweep(np.array([0.0, -0.30]))
    states = [sol.physical_state(u) for u in U]
    return {"sol": sol, "J": J, "reps": reps, "states": states}


def test_generation_drives_a_positive_current_at_zero_bias(photovoltaic):
    assert all(r.converged for r in photovoltaic["reps"])
    assert photovoltaic["J"][0] > 0.0            # same sign as the reverse current
    assert photovoltaic["J"][1] > photovoltaic["J"][0]


def test_short_circuit_current_matches_the_collection_length(dev, photovoltaic):
    """J_sc = q G (W + L_n + L_p): the SCR width plus one diffusion length each side."""
    tr = minority_transport(dev)
    w = depletion(dev, 0.0)["W"]
    estimate = Q_E * G_PHOTON * (w + tr["Ln"] + tr["Lp"])
    assert photovoltaic["J"][0] / estimate == pytest.approx(1.0, rel=0.4)


def test_total_current_stays_continuous_under_generation(photovoltaic):
    """div(Jn + Jp) = 0 still holds, but each carrier current now varies: the
    generation term enters both continuity equations with opposite signs."""
    J0 = photovoltaic["J"][0]
    st = photovoltaic["states"][0]
    assert float(np.ptp(st["J"])) / abs(J0) < 1e-3
    assert float(np.ptp(st["Jn"])) > 0.2 * abs(J0)
    assert float(np.ptp(st["Jp"])) > 0.2 * abs(J0)


def test_generation_is_absent_by_default(dev):
    """Without an explicit source the solver is the dark device of cases 1-5."""
    dark = DDSolver(dev)
    lit = DDSolver(dev, generation=lambda x: G_PHOTON, tol=1e-7)
    u_dark, _ = dark.solve_point(0.0)
    u_lit, _ = lit.solve_point(0.0)
    assert abs(dark.total_current(u_dark)) < 1e-18
    assert lit.total_current(u_lit) > 1e-4
