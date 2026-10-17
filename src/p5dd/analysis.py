"""Post-processing helpers: depletion charge, capacitance, Mott-Schottky fit."""

from __future__ import annotations

import numpy as np

from .constants import Q_E
from .theory import built_in_potential, depletion


def peak_field(state: dict) -> float:
    """Maximum electric-field magnitude [V/cm] inside the device."""
    return float(np.max(np.abs(state["E"])))


def net_recombination_rate(device, state: dict) -> np.ndarray:
    """SRH net recombination rate R = (np - ni^2)/(tau_p(n+ni) + tau_n(p+ni)).

    Returned in cm^-3 s^-1 per node; negative values are net generation, which
    is what a reverse-biased junction shows throughout the space-charge region.
    """
    ni = device.r_ni * device.Nref
    n, p = state["n"], state["p"]
    den = device.mat.tau_p * (n + ni) + device.mat.tau_n * (p + ni)
    return (n * p - ni ** 2) / den


def integrate_on_mesh(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of a nodal quantity over a graded mesh."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def depletion_charge(device, state: dict) -> float:
    """Space charge per unit area on either side of the junction [C/cm^2].

    Gauss' law over a pillbox that covers the whole n-side depletion region
    gives ``Q = eps * E_max``, because the field vanishes in both neutral
    regions of an abrupt junction.
    """
    return float(device.mat.eps * peak_field(state))


def capacitance(voltages, charge) -> np.ndarray:
    """Small-signal depletion capacitance per area [F/cm^2] = dQ/dV."""
    v = np.asarray(voltages, dtype=float)
    q = np.asarray(charge, dtype=float)
    return np.abs(np.gradient(q, v))


def mott_schottky(voltages, cap, eps: float) -> dict:
    """Linear fit of 1/C^2 vs V.

    For an abrupt junction 1/C^2 = 2 (Vbi - V) (1/Na + 1/Nd) / (q eps), so
    ``N_eff = 2 / (q eps |slope|)`` recovers the doping sum and the voltage
    intercept recovers Vbi.
    """
    v = np.asarray(voltages, dtype=float)
    c = np.asarray(cap, dtype=float)
    y = 1.0 / c ** 2
    a, b = np.polyfit(v, y, 1)
    return {"slope": float(a), "intercept": float(b),
            "doping_eff": float(2.0 / (Q_E * eps * abs(a))),
            "Vbi_fit": float(-b / a),
            "R2": float(np.corrcoef(v, y)[0, 1] ** 2)}


def expected_capacitance(device, V) -> np.ndarray:
    """Analytic abrupt-junction depletion capacitance [F/cm^2]."""
    v = np.atleast_1d(np.asarray(V, dtype=float))
    vbi = built_in_potential(device)
    w = np.array([depletion(device, float(vv))["W"] for vv in v])
    return device.mat.eps / w
