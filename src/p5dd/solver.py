"""Steady-state 1D drift-diffusion-Poisson solver.

Formulation
-----------
Unknowns per site (dual-potential / slotboom form), dimensionless:
    psi, phi_n, phi_p      in units of the thermal voltage Vt
    n = r_i exp(psi - phi_n),  p = r_i exp(phi_p - psi),  r_i = ni_i / Nref
The exponential representation keeps the carrier densities strictly positive.

Discretisation
  * Poisson: finite-volume second difference on the graded mesh.
  * Currents: Scharfetter-Gummel link fluxes, i.e. the exact integral of the
    drift-diffusion ODE across one link assuming a linear potential drop.  The
    discrete flux therefore vanishes identically at any Boltzmann equilibrium.
  * SRH recombination with mid-gap traps, plus optional external generation.
  * Ohmic contacts: majority density pinned to the doping, minority density
    pinned to its equilibrium (Boltzmann) value, electrostatic potential pinned
    so that the applied bias drops entirely across the junction.

Solution: damped Newton on the sparse block system with analytic Jacobians,
driven by adaptive bias continuation for an I-V sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .bernoulli import bernoulli, bernoulli_prime

_EXP_CLIP = 60.0


@dataclass
class SolveReport:
    """Convergence record of one Newton solve.

    ``residual`` is the weighted infinity-norm: each equation divided by the
    sum of its own term magnitudes, hence dimensionless and mesh-independent.
    """

    iterations: int = 0
    residual: float = np.inf
    converged: bool = False
    backtracks: int = 0
    history: list = field(default_factory=list)


class DDSolver:
    """Assembles and solves the discrete system for one :class:`Device`.

    ``tol`` applies to the weighted residual (see :meth:`SolveReport`).  Its
    achievable floor is set by the sparse LU solve, not by the discretisation:
    about 3e-9 for a 1601-site junction, which corresponds to a terminal
    current error below 1e-6 A/cm^2 anywhere on the I-V curve.
    """

    def __init__(self, device, generation=None, tol: float = 1e-8,
                 max_newton: int = 80, du_cap: float = 4.0):
        self.dev = device
        self.tol = tol
        self.max_newton = max_newton
        self.du_cap = du_cap
        self.N = device.N
        gen = np.zeros(device.N) if generation is None else np.asarray(
            generation(device.x), dtype=float)
        self.gen_dim = gen / device.Rscale
        # Fixed (bias independent) per-link jump of ln(n_i).  Zero whenever the
        # material model has no band-gap narrowing.
        self._dln_ni = np.log(device.r_ni[1:] / device.r_ni[:-1])

    # ------------------------------------------------------------------ core
    def _fields(self, u, d):
        N = self.N
        psi, phin, phip = u[:N], u[N:2 * N], u[2 * N:]
        n = d.r_ni * np.exp(np.clip(psi - phin, -_EXP_CLIP, _EXP_CLIP))
        p = d.r_ni * np.exp(np.clip(phip - psi, -_EXP_CLIP, _EXP_CLIP))
        return psi, phin, phip, n, p

    def residual(self, u: np.ndarray, V: float, want_jac: bool = False,
                 want_scales: bool = False):
        d = self.dev
        N = self.N
        psi, phin, phip, n, p = self._fields(u, d)

        de = psi[1:] - psi[:-1]
        # Band-gap narrowing (or a hetero-junction) makes n_i vary from node to
        # node, and then the plain potential drop is no longer the quantity
        # whose Boltzmann factor reproduces the density ratio: n_R/n_L =
        # (n_iR/n_iL) exp(de - dphi_n).  Feeding the link Bernoulli functions
        # the shifted drops de +/- d ln(n_i) makes both discrete fluxes vanish
        # identically whenever the quasi-Fermi levels are flat, for any n_i(x).
        dln_ni = self._dln_ni
        de_n, de_p = de + dln_ni, de - dln_ni
        g_n, gm_n = bernoulli(de_n), bernoulli(-de_n)
        g_p, gm_p = bernoulli(de_p), bernoulli(-de_p)
        gp_n, gpm_n = bernoulli_prime(de_n), bernoulli_prime(-de_n)
        gp_p, gpm_p = bernoulli_prime(de_p), bernoulli_prime(-de_p)
        a = d.dn_link / d.hd
        b = d.dp_link / d.hd
        Jn = a * (g_n * n[1:] - gm_n * n[:-1])
        Jp = -b * (gm_p * p[1:] - g_p * p[:-1])

        # SRH in physical units, normalised afterwards.  U is the net
        # generation rate density (G - R) in dimensionless form; the steady
        # continuity pair is  div Jn + qU = 0  and  div Jp - qU = 0.
        n_phys, p_phys = n * d.Nref, p * d.Nref
        ni_phys = d.r_ni * d.Nref
        den = np.maximum(d.mat.tau_n * (p_phys + ni_phys)
                         + d.mat.tau_p * (n_phys + ni_phys), 1e-30)
        num = n_phys * p_phys - ni_phys ** 2
        dU_dn = -(p_phys * den - num * d.mat.tau_p) / den ** 2 * d.Nref / d.Rscale
        dU_dp = -(n_phys * den - num * d.mat.tau_n) / den ** 2 * d.Nref / d.Rscale
        U = self.gen_dim - num / den / d.Rscale
        # chain rule through the slotboom densities: partials of U
        T_psi = dU_dn * n - dU_dp * p
        T_phin = -dU_dn * n
        T_phip = dU_dp * p

        F = np.zeros(3 * N)
        vt = V / d.mat.Vt

        # ---- Poisson -------------------------------------------------------
        F[1:N - 1] = (((psi[2:] - psi[1:-1]) / d.hd[1:]
                       - (psi[1:-1] - psi[:-2]) / d.hd[:-1]) / d.hb[1:-1]) / d.alpha \
            - (n[1:-1] - p[1:-1] + d.cdot[1:-1])
        F[0] = psi[0] - d.psi_left
        F[N - 1] = psi[N - 1] - (d.psi_right0 + vt)

        # ---- electron continuity:  div Jn + U = 0 --------------------------
        F[N + 1:2 * N - 1] = (Jn[1:] - Jn[:-1]) / d.hb[1:-1] + U[1:-1]
        # ---- hole continuity:  div Jp - U = 0 ------------------------------
        F[2 * N + 1:3 * N - 1] = (Jp[1:] - Jp[:-1]) / d.hb[1:-1] - U[1:-1]
        # Ohmic contacts, written on the dual potentials: pinning phi_n and
        # phi_p together with psi is exactly equivalent to pinning both carrier
        # densities to their equilibrium (Boltzmann) values, but keeps every
        # boundary row O(1) instead of ~1e-15.
        F[N] = phin[0]
        F[2 * N] = phip[0]
        F[2 * N - 1] = phin[N - 1] - vt
        F[3 * N - 1] = phip[N - 1] - vt

        # Weights for a mesh-independent stopping measure: the sum of the
        # absolute term magnitudes of each equation.  Dividing the residual by
        # them measures how completely every balance is satisfied relative to
        # its own terms, which an absolute norm cannot express once 1/h^2
        # amplifies the round-off floor.
        s = np.ones(3 * N)
        s[1:N - 1] = ((np.abs(psi[2:] - psi[1:-1]) / d.hd[1:]
                       + np.abs(psi[1:-1] - psi[:-2]) / d.hd[:-1]) / d.hb[1:-1]
                      / d.alpha
                      + n[1:-1] + p[1:-1] + np.abs(d.cdot[1:-1]))
        s[N + 1:2 * N - 1] = ((np.abs(Jn[1:]) + np.abs(Jn[:-1])) / d.hb[1:-1]
                              + np.abs(U[1:-1]))
        s[2 * N + 1:3 * N - 1] = ((np.abs(Jp[1:]) + np.abs(Jp[:-1])) / d.hb[1:-1]
                                  + np.abs(U[1:-1]))
        np.maximum(s, 1.0, out=s)
        self._s = s

        if not want_jac:
            return (F, s) if want_scales else F
        if want_scales:
            pass

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.atleast_1d(np.asarray(r, dtype=int)))
            cols.append(np.atleast_1d(np.asarray(c, dtype=int)))
            vals.append(np.atleast_1d(np.asarray(v, dtype=float)))

        # ---- Poisson rows --------------------------------------------------
        i = np.arange(1, N - 1)
        cL = 1.0 / (d.hd[:-1] * d.hb[1:-1] * d.alpha)
        cR = 1.0 / (d.hd[1:] * d.hb[1:-1] * d.alpha)
        add(i, i + 1, cR)
        add(i, i - 1, cL)
        add(i, i, -(cL + cR) - (n[1:-1] + p[1:-1]))
        add(i, N + i, n[1:-1])
        add(i, 2 * N + i, p[1:-1])
        add(0, 0, 1.0)
        add(N - 1, N - 1, 1.0)

        # ---- link flux derivatives (L = left node, R = right node) ---------
        # dln_ni is a constant, so d(de +/- dln_ni)/dpsi has the same sign
        # pattern as before; only the Bernoulli coefficients become per-species.
        dJn_dpsi_L = a * (-gp_n * n[1:] - (gpm_n + gm_n) * n[:-1])
        dJn_dpsi_R = a * ((gp_n + g_n) * n[1:] + gpm_n * n[:-1])
        dJn_dphin_L = a * gm_n * n[:-1]
        dJn_dphin_R = -a * g_n * n[1:]
        dJp_dpsi_L = -b * (gpm_p * p[1:] + (gp_p + g_p) * p[:-1])
        dJp_dpsi_R = b * ((gpm_p + gm_p) * p[1:] + gp_p * p[:-1])
        dJp_dphi_L = b * g_p * p[:-1]
        dJp_dphi_R = -b * gm_p * p[1:]

        # ---- electron continuity rows (interior) ---------------------------
        i = np.arange(1, N - 1)
        r = N + i
        inv = 1.0 / d.hb[1:-1]
        add(r, i, dJn_dpsi_L[1:] * inv - dJn_dpsi_R[:-1] * inv + T_psi[1:-1])
        add(r, i + 1, dJn_dpsi_R[1:] * inv)
        add(r, i - 1, -dJn_dpsi_L[:-1] * inv)
        add(r, N + i, dJn_dphin_L[1:] * inv - dJn_dphin_R[:-1] * inv + T_phin[1:-1])
        add(r, N + i + 1, dJn_dphin_R[1:] * inv)
        add(r, N + i - 1, -dJn_dphin_L[:-1] * inv)
        add(r, 2 * N + i, T_phip[1:-1])
        # Ohmic contact rows: d(phi)/d(phi) = 1.
        add(N, N, 1.0)
        add(2 * N - 1, 2 * N - 1, 1.0)

        # ---- hole continuity rows (interior) ------------------------------
        i = np.arange(1, N - 1)
        r = 2 * N + i
        add(r, i, dJp_dpsi_L[1:] * inv - dJp_dpsi_R[:-1] * inv - T_psi[1:-1])
        add(r, i + 1, dJp_dpsi_R[1:] * inv)
        add(r, i - 1, -dJp_dpsi_L[:-1] * inv)
        add(r, 2 * N + i, dJp_dphi_L[1:] * inv - dJp_dphi_R[:-1] * inv - T_phip[1:-1])
        add(r, 2 * N + i + 1, dJp_dphi_R[1:] * inv)
        add(r, 2 * N + i - 1, -dJp_dphi_L[:-1] * inv)
        add(r, N + i, -T_phin[1:-1])
        # Ohmic contact rows: d(phi)/d(phi) = 1.
        add(2 * N, 2 * N, 1.0)
        add(3 * N - 1, 3 * N - 1, 1.0)

        J = coo_matrix((np.concatenate(vals),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(3 * N, 3 * N)).tocsr()
        J.sum_duplicates()
        return (F, J, s) if want_scales else (F, J)

    def residual_only(self, u: np.ndarray, V: float) -> np.ndarray:
        return self.residual(u, V, want_jac=False)

    @staticmethod
    def weighted_norm(F: np.ndarray, s: np.ndarray) -> float:
        return float(np.max(np.abs(F) / s))

    # ---------------------------------------------------------------- Newton
    def newton(self, V: float, u0: np.ndarray) -> tuple[np.ndarray, SolveReport]:
        """Damped Newton on the weighted residual.

        The weights are frozen for the duration of one Newton step, so the
        iteration is mathematically identical to plain Newton (a positive
        diagonal scaling cancels in ``J du = -F``) while the acceptance test
        measures every equation against its own term magnitudes.
        """
        rep = SolveReport()
        u = u0.copy()
        F, J, s = self.residual(u, V, want_jac=True, want_scales=True)
        nrm = self.weighted_norm(F, s)
        rep.history.append(nrm)
        for _ in range(self.max_newton):
            if nrm < self.tol:
                break
            du = spsolve(J.tocsc(), -F)
            if not np.all(np.isfinite(du)):
                break
            # Every unknown is a potential expressed in units of Vt, so a single
            # infinity-norm cap is a natural step limiter: without it the first
            # Newton iterate of a strongly-biased step drives the Slotboom
            # exponents (and with them the carrier densities) far out of range.
            dmax = float(np.max(np.abs(du)))
            if dmax > self.du_cap:
                du *= self.du_cap / dmax
            alpha, ok = 1.0, False
            for _ in range(30):
                trial = u + alpha * du
                Fn = self.residual_only(trial, V)
                nn = self.weighted_norm(Fn, s)
                if np.isfinite(nn) and nn < nrm:
                    u, F, nrm, ok = trial, Fn, nn, True
                    break
                alpha *= 0.5
                rep.backtracks += 1
            rep.history.append(nrm)
            if not ok or nrm < self.tol:
                break
            F, J, s = self.residual(u, V, want_jac=True, want_scales=True)
            nrm = self.weighted_norm(F, s)
        rep.iterations = len(rep.history) - 1
        rep.residual = nrm
        rep.converged = nrm < self.tol
        return u, rep

    def solve_point(self, V: float, u0: np.ndarray | None = None):
        """Solve one bias point, ramping the bias from equilibrium if needed.

        ``u0=None`` starts from the analytic depletion-approximation guess and
        uses continuation from ``V=0`` when a direct Newton solve does not
        converge — the guess is only accurate near equilibrium.
        """
        if u0 is not None:
            return self.newton(V, u0)
        u, rep = self.newton(V, self.dev.initial_state(V))
        if rep.converged:
            return u, rep
        u0, rep0 = self.newton(0.0, self.dev.initial_state(0.0))
        if not rep0.converged:
            return u, rep
        u_prev, _, _ = self._advance(u0, 0.0, V)
        return self.newton(V, u_prev)

    # ------------------------------------------------------------------ sweep
    def _advance(self, u, V, target, min_step=1e-13, max_fail=40):
        last = None
        while abs(target - V) > 1e-13:
            step = target - V
            solved = False
            for _ in range(max_fail):
                if abs(step) < min_step:
                    break
                trial_V = V + step
                cand, rep = self.newton(trial_V, u)
                last = rep
                if rep.converged:
                    u, V, solved = cand, trial_V, True
                    break
                step *= 0.5
            if not solved:
                raise RuntimeError(
                    f"bias continuation failed: stuck at V={V:.6f}, target {target:.6f}")
        return u, V, last

    def iv_sweep(self, voltages, u0=None, V0=None):
        """Solve at every voltage of a monotone sequence by continuation.

        Returns ``(states, terminal current density [A/cm^2], reports)``.
        """
        voltages = np.asarray(voltages, dtype=float)
        reports = []
        if u0 is None:
            V = float(voltages[0])
            u, rep = self.solve_point(V)
            if not rep.converged:
                raise RuntimeError(f"first solve failed (residual {rep.residual:.3e})")
            sols, next_idx, reports = [u], 1, [rep]
        else:
            u, V, sols, next_idx = u0.copy(), float(V0), [], 0
        for k in range(next_idx, len(voltages)):
            u, V, rep = self._advance(u, V, float(voltages[k]))
            sols.append(u)
            reports.append(rep)
        U = np.array(sols)
        Jt = np.array([self.total_current(u) for u in U])
        return U, Jt, reports

    # -------------------------------------------------------------- extraction
    def currents(self, u: np.ndarray):
        """Electron / hole current density on links [A/cm^2]."""
        d = self.dev
        psi, phin, phip, n, p = self._fields(u, d)
        de = psi[1:] - psi[:-1]
        de_n, de_p = de + self._dln_ni, de - self._dln_ni
        g_n, gm_n = bernoulli(de_n), bernoulli(-de_n)
        g_p, gm_p = bernoulli(de_p), bernoulli(-de_p)
        Jn = d.dn_link / d.hd * (g_n * n[1:] - gm_n * n[:-1]) * d.Jscale
        Jp = -d.dp_link / d.hd * (gm_p * p[1:] - g_p * p[:-1]) * d.Jscale
        return Jn, Jp

    def total_current(self, u: np.ndarray) -> float:
        Jn, Jp = self.currents(u)
        return float(Jn[0] + Jp[0])

    def physical_state(self, u: np.ndarray) -> dict:
        """Dimensional fields for post-processing."""
        d = self.dev
        psi, phin, phip, n, p = self._fields(u, d)
        Vt = d.mat.Vt
        n_phys, p_phys = n * d.Nref, p * d.Nref
        E = -(psi[1:] - psi[:-1]) * Vt / d.h
        Jn, Jp = self.currents(u)
        Ef = -phin * Vt
        Ec = Ef + Vt * np.log(d.mat.Nc / np.maximum(n_phys, 1e-6))
        Ev = Ef - Vt * np.log(np.maximum(p_phys, 1e-6) / d.mat.Nv)
        return {
            "x": d.x, "um": d.x * 1e4,
            "psi": psi * Vt, "phi_n": phin * Vt, "phi_p": phip * Vt,
            "n": n_phys, "p": p_phys,
            "xm": 0.5 * (d.x[1:] + d.x[:-1]),
            "xm_um": 0.5 * (d.x[1:] + d.x[:-1]) * 1e4,
            "E": E, "Jn": Jn, "Jp": Jp, "J": Jn + Jp,
            "Ec": Ec, "Ev": Ev, "Ei": 0.5 * (Ec + Ev), "Ef": Ef,
        }
