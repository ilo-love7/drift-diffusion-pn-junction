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


# --------------------------------------------------------------- slab modes
# The two references below belong to the uniformly doped slab
# (``devices.homogeneous_device``), which is the abrupt-junction container with
# the metallurgical junction moved outside the mesh.  A sin(k x) perturbation
# that vanishes at both pinned contacts is then an eigenmode of the *discrete*
# system as well, provided the continuum wavenumber is replaced by the
# eigenvalue of the finite-volume second difference.

def discrete_wavenumber(device, k: float) -> float:
    """``k_h = 2 sin(k h / 2) / h`` of the uniform-mesh second difference [cm^-1].

    The discrete Laplacian of ``sin(k x)`` is ``-k_h^2 sin(k x)``, which is the
    eigenvalue the solver actually applies, so using ``k_h`` instead of ``k``
    removes the leading O(h^2) mesh error from a mode-based comparison.
    """
    h = float(np.mean(device.h))
    return float(2.0 * np.sin(k * h / 2.0) / h)


def dielectric_relaxation(device, k: float) -> dict:
    """Decay rate of a charge perturbation ``sin(k x)`` in a doped slab [s^-1].

    Linearising the electron continuity pair about the flat equilibrium and
    eliminating the potential with Poisson gives
    ``d(dn)/dt = -(sigma/eps) dn + Dn d2(dn)/dx2`` with
    ``sigma = q (mu_n n0 + mu_p p0)``: the excess space charge drains at the
    dielectric relaxation rate, slightly accelerated by diffusion.  Both terms
    are evaluated with :func:`discrete_wavenumber`.
    """
    m = device.mat
    ni2 = (device.r_ni[0] * device.Nref) ** 2
    n0 = device.Nd + ni2 / device.Nd
    p0 = ni2 / device.Nd
    sigma = Q_E * (m.mu_n0 * n0 + m.mu_p0 * p0)
    Dn = einstein_mu(m.mu_n0, m.T)
    kh = discrete_wavenumber(device, k)
    return {"sigma": sigma, "Dn": Dn, "k_h": kh,
            "lambda": sigma / m.eps + Dn * kh ** 2}


def ambipolar_decay(device, k: float) -> dict:
    """Decay rate of a quasi-neutral electron-hole plasma ``sin(k x)`` [s^-1].

    With dn = dp the perturbation carries no net charge, so Poisson keeps the
    potential flat and SRH linearises to ``R = dn / tau``: ``np - ni^2`` becomes
    ``dn (n0 + p0)`` and the denominator stays ``~ tau (n0 + p0 + 2 ni)``.  The
    amplitude that statement needs is measured, not assumed - at finite
    injection the same expression reads ``R = dn (n0 + p0 + dn) /
    (tau (n0 + p0 + 2 ni + dn))``, and case 8 finds the extracted rate
    amplitude-independent, and the trace single-exponential, only while ``dn``
    stays below about a tenth of the *minority* density ``p0``.
    The plasma itself leaves at 1/tau but diffuses with the ambipolar
    coefficient ``D_a = (n + p) Dn Dp / (n Dn + p Dp)``, which is set by the
    slow carrier as soon as ``n0 >> dp``.
    """
    m = device.mat
    ni2 = (device.r_ni[0] * device.Nref) ** 2
    n0 = device.Nd + ni2 / device.Nd
    p0 = ni2 / device.Nd
    Dn = einstein_mu(m.mu_n0, m.T)
    Dp = einstein_mu(m.mu_p0, m.T)
    kh = discrete_wavenumber(device, k)
    D_a = (n0 + p0) * Dn * Dp / (n0 * Dn + p0 * Dp)
    return {"D_a": D_a, "k_h": kh, "tau": m.tau_n,
            "lambda": 1.0 / m.tau_n + D_a * kh ** 2}
