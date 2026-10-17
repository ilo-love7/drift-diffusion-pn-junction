"""Case 4 as a regression test: the scheme must converge at second order.

Richardson comparison of the terminal current at 0.60 V and of the equilibrium
depletion width across three successive grids.  The observed order is the single
best summary of discretisation quality: first order would mean the
Scharfetter-Gummel link degraded to upwind differencing, third would mean the
reference values are being fitted rather than computed.
"""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import DDSolver, depletion_charge, peak_field, reference_device

GRIDS = (401, 801, 1601)
V_PROBE = 0.60


@pytest.fixture(scope="module")
def ladder():
    out = []
    for n in GRIDS:
        dev = reference_device(n_nodes=n)
        # tol=1e-7: the sparse-LU residual floor rises on the coarse grids, and
        # solver noise there would otherwise be mistaken for discretisation
        # error (see case 4, where the 0.30 V probe is dominated by it).
        sol = DDSolver(dev, tol=1e-7)
        u0, _ = sol.solve_point(0.0)
        st0 = sol.physical_state(u0)
        q = depletion_charge(dev, st0)
        width = q * (1.0 / dev.Na + 1.0 / dev.Nd) / 1.602176634e-19
        u, rep = sol.solve_point(V_PROBE)
        assert rep.converged
        out.append({"n": n, "J": sol.total_current(u), "W": float(width),
                    "Emax": peak_field(st0), "h": 1.0 / n})
    return out


def _order(ladder, key):
    d1 = abs(ladder[0][key] - ladder[1][key])
    d2 = abs(ladder[1][key] - ladder[2][key])
    return float(np.log(d1 / d2) / np.log(2.0))


@pytest.mark.parametrize("key", ["J", "W", "Emax"])
def test_second_order_convergence(ladder, key):
    p = _order(ladder, key)
    assert 1.5 < p < 2.5, f"{key}: observed order {p:.2f}"


def test_grid_independence_at_the_production_mesh(ladder):
    """1601 nodes differ from 801 by less than 1e-4: the report mesh is converged."""
    for key in ("J", "W"):
        rel = abs(ladder[1][key] - ladder[2][key]) / abs(ladder[2][key])
        assert rel < 2e-4, f"{key}: {rel:.2e} between 801 and 1601 nodes"


def test_equilibrium_invariant_is_mesh_independent(ladder):
    """np = ni^2 at V = 0 is exact algebra, so it must not drift with h."""
    for entry in ladder:
        dev = reference_device(n_nodes=entry["n"])
        sol = DDSolver(dev, tol=1e-7)
        u, _ = sol.solve_point(0.0)
        n, p = dev.densities(u)
        assert np.max(np.abs(n * p / dev.r_ni ** 2 - 1.0)) < 1e-10
