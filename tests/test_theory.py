"""Analytic helpers: the formulas the cases compare against must be right too."""

from __future__ import annotations

import numpy as np
import pytest
from p5dd.constants import Q_E, einstein_mu, intrinsic_density, thermal_voltage

from p5dd import (Material, built_in_potential, depletion, ideality_factor,
                  minority_transport, saturation_current, shockley)


def test_thermal_voltage_at_room_temperature():
    assert thermal_voltage(300.0) == pytest.approx(0.025852, rel=1e-4)


def test_intrinsic_density_is_silicon_like():
    ni = intrinsic_density(2.8e19, 1.83e19, 1.12, 300.0)
    assert 1e9 < ni < 2e10


def test_einstein_relation_round_trip():
    mu = 1350.0
    assert einstein_mu(mu, 300.0) / mu == pytest.approx(thermal_voltage(300.0))


def test_built_in_potential_equals_textbook_form():
    """Vbi = Vt ln(ND NA / ni^2) when band-gap narrowing is switched off."""
    mat = Material(mobility_model="constant", bgn_model="none")
    nd, na = 1e17, 1e16
    expected = mat.Vt * np.log(nd * na / mat.ni0() ** 2)
    dev = _device(nd, na, mat)
    assert built_in_potential(dev) == pytest.approx(expected, rel=1e-12)


def _device(nd, na, mat):
    from p5dd import Device, graded_mesh

    x = graded_mesh(1.2e-2, 6e-3, 4e-7, 5e-5, 6e-4, 401)
    return Device(x=x, Nd=nd, Na=na, x_j=6e-3, mat=mat)


def test_depletion_is_self_consistent():
    """theory.depletion and Device.depletion_width are independent code paths."""
    from p5dd import reference_device

    dev = reference_device(n_nodes=401)
    for V in (0.0, -0.3, 0.5):
        ana = depletion(dev, V)
        num = dev.depletion_width(V)
        assert ana["W"] == pytest.approx(num[0] + num[1], rel=1e-9)
        assert ana["x_n"] == pytest.approx(num[0], rel=1e-9)
    # Gauss' law: eps*Emax must equal the ionised charge on either side.
    ana = depletion(dev, 0.0)
    assert dev.mat.eps * ana["Emax"] == pytest.approx(ana["Q"], rel=1e-12)
    assert ana["Q"] == pytest.approx(Q_E * dev.Nd * ana["x_n"], rel=1e-12)


def test_depletion_width_expands_under_reverse_bias():
    from p5dd import reference_device

    dev = reference_device(n_nodes=401)
    w = [depletion(dev, V)["W"] for V in (0.3, 0.0, -0.3, -0.6)]
    assert np.all(np.diff(w) > 0)      # W grows as the bias goes more reverse
    # and follows the (Vbi - V)^1/2 law
    vbi = built_in_potential(dev)
    root = np.sqrt(np.maximum(vbi - np.array([0.3, 0.0, -0.3, -0.6]), 0.0))
    assert np.allclose(root / root[0], np.array(w) / w[0], rtol=1e-12)


def test_saturation_current_is_the_minority_diffusion_sum():
    from p5dd import reference_device

    dev = reference_device(n_nodes=401)
    tr = minority_transport(dev)
    ni = dev.mat.ni0()
    expected = Q_E * ni ** 2 * (tr["Dn"] / (tr["Ln"] * dev.Na)
                                + tr["Dp"] / (tr["Lp"] * dev.Nd))
    assert saturation_current(dev) == pytest.approx(expected, rel=1e-12)
    assert tr["Ln"] == pytest.approx(np.sqrt(tr["Dn"] * dev.mat.tau_n), rel=1e-12)


def test_shockley_sign_and_zero():
    from p5dd import reference_device

    dev = reference_device(n_nodes=401)
    assert shockley(dev, 0.0) == pytest.approx(0.0, abs=1e-30)
    assert shockley(dev, 0.3) < 0.0 < shockley(dev, -0.3)
    js = saturation_current(dev)
    assert shockley(dev, -0.3) == pytest.approx(-js * np.expm1(-0.3 / dev.mat.Vt))


@pytest.mark.parametrize("n_id", [1.0, 1.5, 2.0])
def test_ideality_factor_recovers_a_synthetic_diode(n_id):
    """n_id = (1/Vt) dV/d ln|J| must return the exponent it was fed."""
    Vt = thermal_voltage(300.0)
    V = np.linspace(0.05, 0.6, 24)
    J = -1e-9 * np.exp(V / (n_id * Vt))
    got = ideality_factor(V, J)
    assert got == pytest.approx(n_id, rel=1e-3)


def test_ideality_factor_is_positive_for_a_negative_current():
    """The solver reports forward current as negative; log|J| keeps n_id > 0."""
    Vt = thermal_voltage(300.0)
    V = np.linspace(0.1, 0.5, 12)
    J = -np.exp(V / Vt)
    assert np.all(ideality_factor(V, J) > 0.0)
