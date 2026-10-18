"""Modal analysis of a transient run and the exact time-discretisation maps.

A perturbation of the form ``sin(pi x / L)`` in a uniformly doped slab (and, to
the extent the equations stay linear, the corresponding eigenmode of a biased
junction) decays as a single exponential, so a whole transient field can be
reduced to one number: the projection of the state onto that mode.  Fitting the
logarithm of its amplitude then measures the decay rate directly, which can be
compared with an analytic rate -- something an integral metric such as a
terminal current cannot do.

The comparison has to account for the time integrator itself.  For the scalar
test equation ``y' = -lambda y`` the theta scheme reproduces the amplitude by a
constant factor per step,

    theta = 1  :  G = 1 / (1 + x)            x = lambda dt
    theta = 1/2:  G = (1 - x/2) / (1 + x/2)

so the *measured* rate is a known function of the true one, invertible in closed
form (see :func:`true_rate`).  Correcting a measured rate with these maps
separates the temporal error of the integrator from the physics, and the
residual dependence on ``dt`` is the order-of-convergence evidence: the lag is
O(dt) for backward Euler and O(dt^2) for Crank-Nicolson.
"""

from __future__ import annotations

import numpy as np

_COMPONENTS = ("psi", "phi_n", "phi_p", "n", "p")


def discrete_laplacian(device, P) -> np.ndarray:
    """Second difference of a nodal field on the interior nodes [1/Lambda^2]."""
    return ((P[2:] - P[1:-1]) / device.hd[1:]
            - (P[1:-1] - P[:-2]) / device.hd[:-1]) / device.hb[1:-1]


def mode_wavenumber(device, mode: int = 1) -> float:
    """Continuum wavenumber of the ``mode``-th half-sine on the slab [cm^-1]."""
    return mode * np.pi / (device.x[-1] - device.x[0])


def charge_mode_state(device, u_eq, amp: float, k: float | None = None) -> np.ndarray:
    """Slab state whose potential deviates by ``amp sin(k (x - x0))`` [in Vt units].

    The density perturbation is whatever Poisson demands,
    ``dn = [lap(psi) - lap(psi_eq)] / alpha``, applied to the interior nodes only
    so that the pinned contact densities stay at their equilibrium values, and
    the electron quasi-Fermi level is rewritten to carry it.  The hole rows are
    left untouched, which is consistent because a doped slab holds negligible
    equilibrium holes: this is the pure space-charge (dielectric relaxation)
    mode.  Any other start would break Poisson and spend the first Newton
    iterate on the resulting field shock instead of on the mode.
    """
    d = device
    N = d.N
    k = mode_wavenumber(d) if k is None else k
    s = np.sin(k * (d.x - d.x[0]))
    u_eq = np.asarray(u_eq, dtype=float)
    psi = u_eq[:N] + amp * s
    n, p = d.densities(u_eq)
    n[1:-1] += (discrete_laplacian(d, psi) - discrete_laplacian(d, u_eq[:N])) / d.alpha
    u = u_eq.copy()
    u[:N] = psi
    u[N:2 * N] = psi - np.log(n / d.r_ni)
    return u


def plasma_mode_state(device, u_eq, delta, k: float | None = None) -> np.ndarray:
    """Slab state with ``dn = dp = delta sin(k (x - x0))`` and a flat potential.

    Equal electron and hole perturbations carry no net charge, so Poisson is
    satisfied by leaving the potential at its equilibrium profile -- this is the
    quasi-neutral excess-carrier mode, which decays at recombination speed while
    diffusing with the ambipolar coefficient.  ``delta`` is dimensionless
    (relative to ``Nref``) and must stay far below ``Nd`` for the mode to remain
    linear, yet far above the equilibrium minority density so that SRH is in its
    high-injection form.
    """
    d = device
    N = d.N
    k = mode_wavenumber(d) if k is None else k
    s = np.sin(k * (d.x - d.x[0]))
    u_eq = np.asarray(u_eq, dtype=float)
    n, p = d.densities(u_eq)
    n = n + delta * s
    p = p + delta * s
    u = u_eq.copy()
    u[N:2 * N] = u_eq[:N] - np.log(n / d.r_ni)
    u[2 * N:] = u_eq[:N] + np.log(p / d.r_ni)
    return u


def mode_amplitude(device, u, u_ref, component: str = "psi",
                   k: float | None = None) -> float:
    """Least-squares coefficient of ``sin(k (x - x0))`` in ``u - u_ref``.

    ``component`` selects one of ``psi``, ``phi_n``, ``phi_p``, ``n``, ``p``.
    The projection uses the finite-volume weights ``hb``, i.e. the inner product
    the discrete Laplacian is self-adjoint with, and a mode that vanishes at
    both contacts is orthogonal to the Dirichlet rows.
    """
    d = device
    N = d.N
    k = mode_wavenumber(d) if k is None else k
    s = np.sin(k * (d.x - d.x[0]))
    u = np.asarray(u, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    if component in ("psi", "phi_n", "phi_p"):
        b = {"psi": 0, "phi_n": N, "phi_p": 2 * N}[component]
        v = u[b:b + N] - u_ref[b:b + N]
    elif component in ("n", "p"):
        i = 0 if component == "n" else 1
        v = (d.densities(u)[i] - d.densities(u_ref)[i])
    else:
        raise ValueError(f"unknown component {component!r}, expected {_COMPONENTS}")
    return float(np.sum(s * v * d.hb) / np.sum(s * s * d.hb))


def fit_decay_rate(t, a) -> dict:
    """Least-squares logarithmic slope of a decaying amplitude.

    Besides the rate (positive for decay) the fit rms relative to the amplitude
    is returned, so "this is a single exponential" is a measured statement
    rather than an assumption.
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(a, dtype=float)
    keep = np.abs(a) > 0.0
    t, a = t[keep], a[keep]
    w, c = np.polyfit(t, np.log(np.abs(a)), 1)
    model = np.sign(a[0]) * np.exp(c + w * t)
    return {"rate": float(-w),
            "rms_rel": float(np.max(np.abs(a - model) / np.abs(a))),
            "a0": float(a[0]), "a1": float(a[-1]),
            "efolds": float(-w * (t[-1] - t[0]))}


def true_rate(measured: float, dt: float, theta: float) -> float:
    """Invert the scalar amplification map: measured rate -> rate of the ODE.

    Both rates and ``dt`` are physical (s^-1 and s); the map acts on the
    dimensionless product ``x = lambda dt``, which is the same number in the
    solver's own time unit because ``lambda dt = lambda_hat dt_hat``.  Writing
    ``m = measured * dt`` for the per-step logarithmic decay,

        theta = 1   : x = expm1(m)
        theta = 1/2 : x = 2 tanh(m / 2)
    """
    m = float(measured) * float(dt)
    if theta == 1.0:
        x = np.expm1(m)
    elif theta == 0.5:
        x = 2.0 * np.tanh(0.5 * m)
    else:
        raise ValueError("closed-form maps exist for theta = 1 and theta = 0.5 only")
    return float(x / dt)


def measured_rate(lam: float, dt: float, theta: float) -> float:
    """Forward map: rate of the ODE -> rate that a finite ``dt`` measures."""
    x = float(lam) * float(dt)
    if theta == 1.0:
        g = 1.0 / (1.0 + x)
    elif theta == 0.5:
        g = (1.0 - 0.5 * x) / (1.0 + 0.5 * x)
    else:
        raise ValueError("closed-form maps exist for theta = 1 and theta = 0.5 only")
    return float(-np.log(g) / dt)


def convergence_order(h, err) -> float:
    """Log-log slope of ``err`` against step size ``h``, from the two finest ``h``."""
    h = np.asarray(h, dtype=float)
    e = np.abs(np.asarray(err, dtype=float))
    i = np.argsort(h)
    h, e = h[i], e[i]
    ok = e > 0.0
    h, e = h[ok], e[ok]
    if len(h) < 2:
        return float("nan")
    return float(np.log(e[-1] / e[-2]) / np.log(h[-1] / h[-2]))


def first_harmonic(t, y, omega: float) -> complex:
    """Complex amplitude of the ``cos(omega t + arg)`` component of ``y``.

    This is the discrete lock-in: the mean of ``2 y exp(-j omega t)`` over the
    samples.  It is exact rather than quadrature-accurate when the samples span
    a whole number of periods with the final duplicate excluded, because the DFT
    basis vectors are then orthogonal, and it rejects any DC offset.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    return complex(2.0 * np.mean(y * np.exp(-1j * omega * t)))
