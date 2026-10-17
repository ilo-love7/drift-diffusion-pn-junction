"""Case 1 invariants as regression tests: equilibrium must be exact.

Nothing in the solver is told that V = 0 is special.  Zero current, flat
quasi-Fermi levels and ``n p = n_i^2`` are consequences of the discretisation
being right, so they are the strongest cheap checks in the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import DDSolver, built_in_potential, depletion, reference_device
from p5dd.constants import Q_E


def test_newton_converges_at_equilibrium(equilibrium):
    assert equilibrium.rep.converged
    assert equilibrium.rep.iterations <= 6
    assert equilibrium.rep.residual < 1e-7


def test_terminal_current_vanishes(equilibrium):
    # A spurious equilibrium current is the classic Scharfetter-Gummel bug:
    # it appears whenever the flux does not reduce to the Boltzmann ratio.
    assert abs(equilibrium.sol.total_current(equilibrium.u)) < 1e-18


def test_quasi_fermi_levels_are_flat(equilibrium):
    dev, u = equilibrium.dev, equilibrium.u
    N = dev.N
    psi, phin, phip = u[:N], u[N:2 * N], u[2 * N:]
    assert np.ptp(phin) * dev.mat.Vt < 1e-12
    assert np.ptp(phip) * dev.mat.Vt < 1e-12


def test_mass_action_holds_everywhere(equilibrium):
    """n p = n_i^2 inside the space-charge region too, not only in the bulk."""
    dev, u = equilibrium.dev, equilibrium.u
    n, p = dev.densities(u)
    assert np.max(np.abs(n * p / dev.r_ni ** 2 - 1.0)) < 1e-9


def test_band_gap_narrowing_does_not_break_equilibrium(equilibrium_bgn):
    """Regression test for the generalised Scharfetter-Gummel link.

    With Slotboom band-gap narrowing n_i jumps by up to 1.602x across the
    metallurgical step.  A link flux built on the bare potential drop is then no
    longer flux-free at equilibrium and shows up as a ~12 % violation of
    n p = n_i(x)^2; carrying ln(n_i) into the Bernoulli argument removes it.
    """
    dev, u = equilibrium_bgn.dev, equilibrium_bgn.u
    n, p = dev.densities(u)
    assert np.max(np.abs(n * p / dev.r_ni ** 2 - 1.0)) < 1e-9
    assert abs(equilibrium_bgn.sol.total_current(u)) < 1e-18


def test_generalised_link_is_exactly_inert_without_band_gap_narrowing(equilibrium):
    """A constant n_i must make the generalisation inert, not merely harmless.

    d(ln n_i) is identically zero on the reference device, so the per-species
    Bernoulli arguments collapse onto the plain potential drop and the scheme is
    textbook Scharfetter-Gummel: the correction cannot silently perturb the case
    it was not designed for.
    """
    assert np.all(equilibrium.sol._dln_ni == 0.0)
    assert np.ptp(equilibrium.dev.r_ni) == 0.0


def test_built_in_potential_matches_the_analytic_value(equilibrium):
    dev, st = equilibrium.dev, equilibrium.st
    numeric = st["psi"][0] - st["psi"][-1]
    assert numeric == pytest.approx(built_in_potential(dev), rel=1e-9)


@pytest.mark.parametrize("name", ["W", "Emax", "x_n", "x_p"])
def test_depletion_parameters_within_depletion_approximation(equilibrium, name):
    """Gauss-law measures agree with the abrupt-junction formula to a few %.

    The residual carriers outside the sharp analytic edge make the numerical
    space-charge region slightly narrower; 4 % is the observed gap at 801 nodes.
    """
    dev, st = equilibrium.dev, equilibrium.st
    ref = depletion(dev, 0.0)
    e_max = float(np.max(np.abs(st["E"])))
    if name == "Emax":
        num = e_max
    elif name == "W":
        num = dev.mat.eps * e_max / Q_E * (1.0 / dev.Na + 1.0 / dev.Nd)
    elif name == "x_n":
        num = dev.mat.eps * e_max / (Q_E * dev.Nd)
    else:
        num = dev.mat.eps * e_max / (Q_E * dev.Na)
    assert num == pytest.approx(ref[name], rel=0.04)


def test_potential_drop_identity_holds_at_bias():
    """psi(0) - psi(L) = Vbi - V for every bias of a short forward sweep.

    The coarse 401-node mesh is used with a 1e-7 residual target: the sparse LU
    floor, not the discretisation, sets what Newton can reach, and it rises with
    the 1/h^2 Poisson scaling.
    """
    dev = reference_device(n_nodes=401)
    sol = DDSolver(dev, tol=1e-7)
    vbi = built_in_potential(dev)
    for V in (0.0, 0.2, 0.4, 0.6):
        u, rep = sol.solve_point(V)
        assert rep.converged
        psi = sol.physical_state(u)["psi"]
        assert psi[0] - psi[-1] == pytest.approx(vbi - V, abs=1e-12)


def test_carrier_densities_stay_positive_under_strong_forward_bias():
    dev = reference_device(n_nodes=401)
    sol = DDSolver(dev, tol=1e-7)
    u, rep = sol.solve_point(0.70)
    assert rep.converged
    n, p = dev.densities(u)
    assert np.all(n > 0.0) and np.all(p > 0.0)
    st = sol.physical_state(u)
    assert np.all(st["n"] > 0.0) and np.all(st["p"] > 0.0)
