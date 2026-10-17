"""The Bernoulli functions carry the entire Scharfetter-Gummel flux."""

from __future__ import annotations

import numpy as np
import pytest

from p5dd import bernoulli, bernoulli_prime

XS = np.array([-40.0, -3.0, -0.75, -1e-4, 0.0, 1e-4, 0.5, 2.0, 12.0, 55.0])


def test_b0_is_one():
    assert bernoulli(0.0) == pytest.approx(1.0)
    assert bernoulli_prime(0.0) == pytest.approx(-0.5)


def test_matches_closed_form():
    x = np.concatenate([np.linspace(-30, -1e-6, 120), np.linspace(1e-6, 30, 120)])
    assert bernoulli(x) == pytest.approx(x / np.expm1(x), rel=1e-13)
    assert bernoulli(0.0) == pytest.approx(1.0)


def test_reflection_identity():
    # B(-x) = exp(x) B(x); this is what makes the SG flux vanish at Boltzmann
    # equilibrium, where n_R / n_L = exp(delta psi).
    x = np.linspace(1e-3, 40.0, 97)
    assert bernoulli(-x) == pytest.approx(np.exp(x) * bernoulli(x), rel=1e-12)


def test_derivative_matches_finite_differences():
    h = 1e-6
    for x in XS:
        fd = (bernoulli(x + h) - bernoulli(x - h)) / (2.0 * h)
        assert bernoulli_prime(x) == pytest.approx(fd, rel=2e-5, abs=2e-5)


def test_branch_transitions_are_continuous():
    """No step where the Taylor, exact and asymptotic branches hand over.

    The probe offset is far smaller than the local slope, so a genuine
    discontinuity would show up undiluted; the tolerance only has to clear the
    O(delta) change of a smooth function.
    """
    delta = 1e-12
    for boundary in (1e-8, 1e-3, 60.0, -1e-8, -1e-3, -60.0):
        for fn in (bernoulli, bernoulli_prime):
            lo = fn(boundary * (1.0 - delta))
            hi = fn(boundary * (1.0 + delta))
            assert abs(hi - lo) < 1e-8 * max(abs(lo), 1.0)


def test_no_overflow_far_outside_the_range():
    for x in (-1e6, -400.0, 400.0, 1e6):
        assert np.isfinite(bernoulli(x))
        assert np.isfinite(bernoulli_prime(x))
    # In the strongly reversed limit B(-x) -> -x, which is the pure-drift flux.
    assert bernoulli(-400.0) == pytest.approx(400.0, rel=1e-12)


def test_scalar_and_array_agree():
    arr = bernoulli(XS)
    assert np.asarray([bernoulli(float(x)) for x in XS]) == pytest.approx(arr)
