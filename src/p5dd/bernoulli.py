"""Bernoulli functions used by the Scharfetter-Gummel flux, overflow-safe."""

from __future__ import annotations

import numpy as np

# Beyond this magnitude exp() overflows in float64 and the asymptotic forms
# are accurate to better than 1e-25 relative error.
_LIMIT = 60.0


def _as_array(x):
    scalar = np.ndim(x) == 0
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    return arr, scalar


def bernoulli(x, clip: float = _LIMIT):
    """B(x) = x / (exp(x) - 1), with B(0) = 1."""
    arr, scalar = _as_array(x)
    out = np.empty_like(arr)
    small = np.abs(arr) < 1e-8
    big_pos = arr > clip
    big_neg = arr < -clip
    mid = ~(small | big_pos | big_neg)

    xs = arr[small]
    out[small] = 1.0 - xs / 2.0 + xs * xs / 12.0

    bm = arr[mid]
    out[mid] = bm / np.expm1(bm)

    bp = arr[big_pos]
    out[big_pos] = bp * np.exp(-bp)

    bn = arr[big_neg]
    out[big_neg] = -bn

    return float(out[0]) if scalar else out


def bernoulli_prime(x, clip: float = _LIMIT):
    """dB/dx = [exp(x)(1 - x) - 1] / (exp(x) - 1)**2.

    Branches: Taylor series near zero (the closed form is 0/0 there and loses
    precision once |x| < 1e-3), the asymptotic ``x exp(-x)`` / ``-1`` limits
    beyond ``clip``, and the exact expression everywhere in between.
    """
    arr, scalar = _as_array(x)
    out = np.empty_like(arr)
    small = np.abs(arr) < 1e-3
    big_pos = arr > clip
    big_neg = arr < -clip
    mid = ~(small | big_pos | big_neg)

    xs = arr[small]
    out[small] = -0.5 + xs / 6.0 - xs ** 3 / 180.0

    bm = arr[mid]
    em1 = np.expm1(bm)
    out[mid] = (em1 - bm * (em1 + 1.0)) / (em1 * em1)

    bp = arr[big_pos]
    out[big_pos] = np.exp(-bp) * (1.0 - bp)

    bn = arr[big_neg]
    out[big_neg] = -1.0 + bn * np.exp(bn)

    return float(out[0]) if scalar else out
