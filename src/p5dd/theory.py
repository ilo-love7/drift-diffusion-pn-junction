"""Analytic reference formulas used to validate the numerical solver."""

from __future__ import annotations

import numpy as np

from .constants import Q_E, einstein_mu, thermal_voltage


def ni_local(device, side: str) -> float:
    """Effective intrinsic density on one side of the junction [cm^-3]."""
    return float(device.r_ni[0 if side == "n" else -1] * device.Nref)


def built_in_potential(device) -> float:
    """Vbi = Vt [ln(ND/ni_n) + ln(NA/ni_p)] [V]."""
    return (np.log(device.Nd / ni_local(device, "n"))
            + np.log(device.Na / ni_local(device, "p"))) * device.mat.Vt


def depletion(device, V: float) -> dict:
    """Abrupt-junction depletion approximation at bias ``V`` [V]."""
    eps, Vt = device.mat.eps, device.mat.Vt
    vbi = built_in_potential(device)
    vj = max(vbi - V, 1e-9)
    W = float(np.sqrt(2.0 * eps * vj / Q_E * (1.0 / device.Na + 1.0 / device.Nd)))
    x_n = W * device.Na / (device.Na + device.Nd)
    Emax = float(np.sqrt(2.0 * Q_E * vj / eps * device.Na * device.Nd
                         / (device.Na + device.Nd)))
    return {"Vbi": vbi, "W": W, "x_n": x_n, "x_p": W - x_n, "Emax": Emax,
            "Q": Q_E * device.Nd * x_n}          # |space charge| per area [C/cm^2]


def minority_transport(device) -> dict:
    """Diffusion coefficients / lengths of the injected minority carriers."""
    T = device.mat.T
    Dn = einstein_mu(float(device.mat.mu_n(np.full(1, device.Na))[0]), T)   # in p side
    Dp = einstein_mu(float(device.mat.mu_p(np.full(1, device.Nd))[0]), T)   # in n side
    return {"Dn": Dn, "Dp": Dp,
            "Ln": float(np.sqrt(Dn * device.mat.tau_n)),
            "Lp": float(np.sqrt(Dp * device.mat.tau_p))}


def saturation_current(device) -> float:
    """Long-diode Shockley prefactor J_s [A/cm^2]."""
    tr = minority_transport(device)
    jn = Q_E * tr["Dn"] * ni_local(device, "p") ** 2 / (tr["Ln"] * device.Na)
    jp = Q_E * tr["Dp"] * ni_local(device, "n") ** 2 / (tr["Lp"] * device.Nd)
    return float(jn + jp)


def shockley(device, V):
    """Ideal-diode current density [A/cm^2].

    Sign convention of the solver: the n region sits at x=0, so forward bias
    drives a negative conventional current (it flows from the p contact to the
    n contact inside the device).
    """
    Vt = device.mat.Vt
    x = np.clip(np.asarray(V, dtype=float) / Vt, -60.0, 60.0)
    return -saturation_current(device) * np.expm1(x)


def ideality_factor(V, J, T: float = 300.0) -> np.ndarray:
    """n_id = (1/Vt) dV/d ln|J|, secant slopes on the sweep grid.

    For J = J0 exp(V / (n_id Vt)) this returns n_id exactly; a pure diffusion
    current gives 1, space-charge recombination gives 2.
    """
    V = np.asarray(V, dtype=float)
    lnJ = np.log(np.maximum(np.abs(np.asarray(J, dtype=float)), 1e-300))
    return np.gradient(V, lnJ) / thermal_voltage(T)
