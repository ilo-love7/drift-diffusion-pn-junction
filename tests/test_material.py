"""Doping-dependent material models and the Sesame input adapter."""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import Material, reference_device, realistic_device, sesame_system
from p5dd.constants import EPS_0, einstein_mu, intrinsic_density


def test_reference_model_is_the_ideal_one():
    m = Material(mobility_model="constant", bgn_model="none")
    n = np.array([1e14, 1e17, 1e19])
    assert np.all(m.bgn(n) == 0.0)
    assert np.all(m.mu_n(n) == m.mu_n0) and np.all(m.mu_p(n) == m.mu_p0)
    assert np.all(m.ni(n) == m.ni0())


def test_slotboom_shift_at_the_reference_doping():
    m = Material(bgn_model="slotboom")
    assert float(m.bgn(1e18)) == pytest.approx(32.5e-3, rel=1e-12)
    assert float(m.bgn(0.0)) == 0.0
    # dEg/dN > 0 and sublinear: doubling the doping adds less than doubling.
    # Sampled on a *linear* doping grid - on a log grid the spacing itself grows
    # faster than the 1/8-power flattens, so the increments would rise.
    shift = m.bgn(np.linspace(0.0, 4e18, 9))
    assert np.all(np.diff(shift) > 0.0)
    assert np.all(np.diff(shift)[1:] < np.diff(shift)[:-1])


def test_band_gap_narrowing_raises_the_intrinsic_density():
    m = Material(bgn_model="slotboom")
    ratio = m.ni(1e18) / m.ni0()
    assert ratio == pytest.approx(
        np.exp(float(m.bgn(1e18)) / (2 * m.Vt)), rel=1e-12)
    assert 1.0 < ratio < 3.0


def test_arora_mobility_decreases_with_doping():
    m = Material(mobility_model="arora")
    n = np.array([1e14, 1e16, 1e17, 1e18, 1e19])
    mu_n, mu_p = m.mu_n(n), m.mu_p(n)
    assert np.all(np.diff(mu_n) < 0.0) and np.all(np.diff(mu_p) < 0.0)
    assert mu_n[0] == pytest.approx(m.mu_n0, rel=0.05)
    assert mu_n[2] == pytest.approx(780.2, rel=1e-3)   # Si 1e17 donor, 300 K
    assert 92.0 < mu_n[-1] < 200.0                     # stays above mu_min
    assert np.all(mu_p < m.mu_p0)


def test_einstein_relation_holds_per_site():
    m = Material(mobility_model="arora")
    n = np.array([1e15, 1e17, 1e19])
    assert m.d_n(n) == pytest.approx(einstein_mu(m.mu_n(n), m.T), rel=1e-15)
    assert m.d_p(n) == pytest.approx(einstein_mu(m.mu_p(n), m.T), rel=1e-15)


def test_srh_recombination_sign_structure():
    m = Material()
    ni = m.ni0()
    assert m.srh(np.array([ni]), np.array([ni])) == pytest.approx(0.0, abs=1e-30)
    excess = m.srh(np.array([10 * ni]), np.array([10 * ni]))
    assert excess > 0.0                                # net recombination
    assert m.srh(np.array([ni / 10]), np.array([ni / 10])) < 0.0   # net generation


def test_device_scaling_is_self_consistent():
    dev = reference_device(n_nodes=201)
    m = dev.mat
    assert m.eps == pytest.approx(m.eps_r * EPS_0, rel=1e-15)
    assert dev.Nref == max(dev.Nd, dev.Na, m.ni0())
    assert np.all(dev.r_ni > 0.0)
    # the reference device has one uniform n_i, the realistic one does not
    assert np.ptp(dev.r_ni) == 0.0
    real = realistic_device(n_nodes=201).r_ni
    assert np.ptp(real) / real.min() > 0.1        # >10 % spread across the junction


@pytest.mark.parametrize("n_nodes", [201])
def test_sesame_adapter_maps_the_device_exactly(n_nodes):
    """The adapter must hand Sesame the identical physical input.

    Sesame stores everything dimensionless, in its own fixed reference units
    (density 1e19 cm^-3, mobility 1 cm^2/Vs), so each quantity is converted back
    out through ``sys_.scaling`` before it is compared with the device.
    """
    pytest.importorskip("sesame", reason="vendored Sesame not available")
    dev = reference_device(n_nodes=n_nodes)
    m = dev.mat
    sys_ = sesame_system(dev)
    sc = sys_.scaling

    np.testing.assert_allclose(np.asarray(sys_.xpts, dtype=float), dev.x)
    assert sys_.Nc[0] * sc.density == pytest.approx(m.Nc)
    assert sys_.Nv[0] * sc.density == pytest.approx(m.Nv)
    assert sys_.Eg[0] * sc.energy == pytest.approx(m.Eg)
    assert sys_.epsilon[0] == pytest.approx(m.eps_r)     # relative, unscaled
    assert sys_.mu_e[0] * sc.mobility == pytest.approx(m.mu_n0)
    assert sys_.mu_h[0] * sc.mobility == pytest.approx(m.mu_p0)
    assert sys_.tau_e[0] * sc.time == pytest.approx(m.tau_n)
    assert sys_.scaling.energy == pytest.approx(m.Vt)    # i.e. T = 300 K

    # net doping: donors positive on the n side, acceptors negative on the p side
    rho = np.asarray(sys_.rho, dtype=float) * sc.density
    n_side, p_side = dev.x < dev.x_j, dev.x >= dev.x_j
    assert np.all(rho[n_side] == pytest.approx(dev.Nd))
    assert np.all(rho[p_side] == pytest.approx(-dev.Na))
    assert sc.current > 0.0


def test_realistic_device_keeps_the_reference_geometry():
    ref, real = reference_device(n_nodes=201), realistic_device(n_nodes=201)
    assert ref.mat.mobility_model == "constant" and ref.mat.bgn_model == "none"
    assert real.mat.mobility_model == "arora" and real.mat.bgn_model == "slotboom"
    np.testing.assert_allclose(real.x, ref.x)
    assert real.Vbi < ref.Vbi              # narrowing lowers the built-in voltage
    assert intrinsic_density(ref.mat.Nc, ref.mat.Nv, ref.mat.Eg, 300.0) > 0
