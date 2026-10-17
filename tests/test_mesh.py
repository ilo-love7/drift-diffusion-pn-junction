"""Graded mesh: resolution concentrated where the physics needs it."""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import graded_mesh

L, X_J, H_FINE, H_MAX, SPREAD = 1.2e-2, 6.0e-3, 4.0e-7, 5.0e-5, 6.0e-4


@pytest.mark.parametrize("n", (51, 201, 801))
def test_mesh_is_strictly_ascending_and_spans_the_device(n):
    x = graded_mesh(L, X_J, H_FINE, H_MAX, SPREAD, n)
    assert x.size == n
    assert x[0] == 0.0 and x[-1] == pytest.approx(L)
    assert np.all(np.diff(x) > 0.0)


def test_refinement_sits_on_the_junction():
    x = graded_mesh(L, X_J, H_FINE, H_MAX, SPREAD, 801)
    h = np.diff(x)
    assert abs(x[int(np.argmin(h))] - X_J) * 1e4 < 0.01     # within 10 nm
    assert h.min() < 20e-7                                  # < 20 nm at the step


def test_bulk_is_much_coarser_than_the_junction():
    x = graded_mesh(L, X_J, H_FINE, H_MAX, SPREAD, 801)
    h = np.diff(x)
    ratio = h.max() / h.min()
    # the template spacing is h_fine -> h_max over the spread, so the design
    # ratio 500 nm / 4 nm is what the arc-length scaling preserves
    assert ratio == pytest.approx(H_MAX / H_FINE, rel=0.01)


def test_spacing_steps_are_smooth():
    """No abrupt size jump between neighbours: they would spoil the 1/h stencil."""
    x = graded_mesh(L, X_J, H_FINE, H_MAX, SPREAD, 801)
    h = np.diff(x)
    assert np.max(h[1:] / h[:-1]) < 1.15
    assert np.min(h[1:] / h[:-1]) > 1.0 / 1.15


@pytest.mark.parametrize("quantity", ["min", "max"])
def test_doubling_the_nodes_halves_every_spacing(quantity):
    """The whole profile refines together, which is what makes a Richardson
    order study with n_nodes legitimate."""
    coarse = graded_mesh(L, X_J, H_FINE, H_MAX, SPREAD, 401)
    fine = graded_mesh(L, X_J, H_FINE, H_MAX, SPREAD, 801)
    hc, hf = np.diff(coarse), np.diff(fine)
    pick = (lambda a: a.min()) if quantity == "min" else (lambda a: a.max())
    # the interval count doubles exactly (400 -> 800), the only slack is the
    # 20001-point arc-length integration inside graded_mesh
    assert pick(hf) == pytest.approx(pick(hc) / 2.0, rel=5e-3)


def test_reference_and_realistic_devices_share_the_geometry():
    from p5dd import NA, ND, X_JUNCTION, reference_device, realistic_device

    a, b = reference_device(n_nodes=401), realistic_device(n_nodes=401)
    np.testing.assert_allclose(a.x, b.x)
    assert (a.Nd, a.Na, a.x_j) == (ND, NA, X_JUNCTION) == (b.Nd, b.Na, b.x_j)
    # an abrupt junction: the net doping changes sign exactly once, at x_j
    sign = np.sign(a.cdot)
    jumps = np.flatnonzero(np.diff(sign) != 0.0)
    assert jumps.size == 1
    assert abs(a.x[jumps[0]] - a.x_j) <= float(np.max(np.diff(a.x)))
    assert a.cdot[:jumps[0] + 1].max() < 0.0 < a.cdot[jumps[0] + 1:].min()
