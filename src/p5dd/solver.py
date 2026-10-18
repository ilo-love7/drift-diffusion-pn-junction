"""Steady-state and transient 1D drift-diffusion-Poisson solver.

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

Transients
----------
Time is measured in units of ``Device.Tscale = Lambda^2 / Dref``, the diffusive
scale that makes the dimensionless continuity pair read exactly
``d n / d t = div J + U``.  A step of length ``dt`` then uses the theta scheme
``(x - x_old) / dt = theta R(new) + (1 - theta) R(old)`` (``theta = 1``:
backward Euler, first order and L-stable; ``theta = 0.5``: Crank-Nicolson,
second order but not L-stable, so it can ring under a bias step) with Poisson
still imposed at the new time, and only the two continuity rows acquire a time
term.  Because the sum of the discrete continuity pair reproduces the discrete
Poisson row exactly, the link quantity

    K = Jn + Jp + eps * dE/dt          (particle + displacement current)

is spatially uniform up to the Newton residual; ``K`` is the current an external
ammeter sees, and :meth:`DDSolver.total_current_profile` returns it per link.
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

    Steady mode is the default; :meth:`snapshot` plus :meth:`step` (or the
    driving loop :meth:`transient`) switch on the time terms.
    """

    def __init__(self, device, generation=None, tol: float = 1e-8,
                 max_newton: int = 80, du_cap: float = 4.0, theta: float = 1.0):
        self.dev = device
        self.tol = tol
        self.max_newton = max_newton
        self.du_cap = du_cap
        self.N = device.N
        gen = None if generation is None else (
            generation(device.x) if callable(generation) else generation)
        self.set_generation(gen)
        # Fixed (bias independent) per-link jump of ln(n_i).  Zero whenever the
        # material model has no band-gap narrowing.
        self._dln_ni = np.log(device.r_ni[1:] / device.r_ni[:-1])
        # --- transient state (theta scheme) ---------------------------------
        self.theta = float(theta)     # 1 = backward Euler, 0.5 = Crank-Nicolson
        self.time = 0.0               # [s] simulated time of the last accepted state
        self.V_prev = 0.0             # [V] bias the last accepted state solves
        self._dt = 0.0                # dimensionless step; > 0 means transient mode
        self._u_prev = None           # previous state vector
        self._n_prev = None           # its densities and the two old continuity
        self._p_prev = None           # rows, cached so a step needs no re-evaluation
        self._cn_prev = None
        self._cp_prev = None

    # ------------------------------------------------------------- generation
    def set_generation(self, generation=None) -> None:
        """Install the external generation field [cm^-3 s^-1] (scalar or per node).

        Changing it between transient steps is how a light pulse is switched off.
        """
        if generation is None:
            gen = np.zeros(self.dev.N)
        else:
            gen = np.asarray(generation, dtype=float)
            if gen.ndim == 0:
                gen = np.full(self.dev.N, float(gen))
        self.gen_dim = gen / self.dev.Rscale

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

        # ---- carrier continuity --------------------------------------------
        transient = self._dt > 0.0
        th = self.theta if transient else 1.0
        Cn = (Jn[1:] - Jn[:-1]) / d.hb[1:-1] + U[1:-1]
        Cp = (Jp[1:] - Jp[:-1]) / d.hb[1:-1] - U[1:-1]
        if transient:
            # The theta scheme  (x - x_old)/dt = theta R(new) + (1-theta) R(old)
            # with R the steady spatial balance.  The electron row carries +div Jn
            # and the hole row -div Jp, so the two time terms enter with opposite
            # signs.  Neither interior row mentions V or the generation field
            # explicitly, which is what lets the rows cached at the previous state
            # serve as R(old) for whatever bias this step ramps to.
            inv_dt = 1.0 / self._dt
            tn = (n[1:-1] - self._n_prev[1:-1]) * inv_dt
            tp = (p[1:-1] - self._p_prev[1:-1]) * inv_dt
            Cn = th * Cn + (1.0 - th) * self._cn_prev - tn
            Cp = th * Cp + (1.0 - th) * self._cp_prev + tp
        F[N + 1:2 * N - 1] = Cn
        F[2 * N + 1:3 * N - 1] = Cp
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
        wn = ((np.abs(Jn[1:]) + np.abs(Jn[:-1])) / d.hb[1:-1] + np.abs(U[1:-1]))
        wp = ((np.abs(Jp[1:]) + np.abs(Jp[:-1])) / d.hb[1:-1] + np.abs(U[1:-1]))
        if transient:
            wn = th * wn + (1.0 - th) * np.abs(self._cn_prev) + np.abs(tn)
            wp = th * wp + (1.0 - th) * np.abs(self._cp_prev) + np.abs(tp)
        s[N + 1:2 * N - 1] = wn
        s[2 * N + 1:3 * N - 1] = wp
        # The floor is absolute - one reference rate - and that is what makes the
        # steady study robust: the rows that should vanish there vanish exactly
        # (Scharfetter-Gummel carries no flux at flat quasi-Fermi levels, and
        # there np - ni^2 is zero too), while the depletion-region rows whose
        # content is ~1e-11 carry no physics at all.  Making the floor relative
        # instead asks Newton for a 30 %-of-content cancellation on those rows,
        # which double precision cannot deliver, and the line search stalls; it
        # also does not help a slow mode, whose own content vanishes wherever its
        # shape does.
        #
        # The cost is that a transient balanced entirely by recombination at 1/tau
        # has a whole content of ~1e-9 in these units, so tol = 1e-8 accepts it
        # untouched and the carriers never move.  That is a tolerance question,
        # not a weighting one: the stopping measure is absolute, so the tolerance
        # must come from the rate the chain is meant to resolve.  Case 8 sets
        # tol = min(1e-8, 1e-3 * lambda_hat * amplitude) per chain and reports the
        # achieved residual, and the physical acceptance of a transient is carried
        # by the invariants (uniform total current, exponential fit, dt order),
        # none of which a loose Newton tolerance can fake.
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

        if transient:
            # Only the spatial operator is weighted by theta; R(old) is a constant
            # of the step, and the Poisson / Dirichlet rows are imposed at the new
            # time untouched.
            for arr in (dJn_dpsi_L, dJn_dpsi_R, dJn_dphin_L, dJn_dphin_R,
                        dJp_dpsi_L, dJp_dpsi_R, dJp_dphi_L, dJp_dphi_R,
                        T_psi, T_phin, T_phip):
                arr *= th

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
        if transient:
            # d(-n/dt)/dpsi = -n/dt and d(-n/dt)/dphi_n = +n/dt (Slotboom chain rule).
            add(r, i, -n[1:-1] * inv_dt)
            add(r, N + i, n[1:-1] * inv_dt)
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
        if transient:
            add(r, i, -p[1:-1] * inv_dt)
            add(r, 2 * N + i, p[1:-1] * inv_dt)
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

    # -------------------------------------------------------------- transient
    def snapshot(self, u: np.ndarray, V: float | None = None) -> np.ndarray:
        """Make ``u`` the previous state of the theta scheme (at ``self.time``).

        Caches the densities and the two steady continuity rows, which are all a
        step needs from the past.  ``_dt`` is masked out while reading the rows
        back, so a snapshot taken in the middle of a schedule is still a *steady*
        evaluation of the state it stores.
        """
        u = np.asarray(u, dtype=float)
        N = self.N
        self._u_prev = u.copy()
        _, _, _, n, p = self._fields(u, self.dev)
        self._n_prev, self._p_prev = n, p
        dt_keep, self._dt = self._dt, 0.0
        try:
            F = self.residual(u, self.V_prev if V is None else V, want_jac=False)
        finally:
            self._dt = dt_keep
        self._cn_prev = F[N + 1:2 * N - 1].copy()
        self._cp_prev = F[2 * N + 1:3 * N - 1].copy()
        self.V_prev = float(self.V_prev if V is None else V)
        return u

    def step(self, V: float, dt: float, gen=None, u_guess=None,
             theta: float | None = None):
        """Advance one theta-scheme step of length ``dt`` [s], ending at bias ``V``.

        Returns ``(u, SolveReport)``; a converged state becomes the new previous
        state and moves ``self.time`` forward.  ``gen`` (cm^-3 s^-1, scalar or per
        node) is installed for the *end* of the step, which is what a switched
        light pulse needs: the Crank-Nicolson offset still reads the field that
        was active when the previous state was stored.
        """
        if self._u_prev is None:
            raise RuntimeError("no previous state: call snapshot() or transient() first")
        if dt <= 0.0:
            raise ValueError("a time step must be positive")
        if theta is not None:
            self.theta = float(theta)
        t_new = self.time + float(dt)
        self._dt = float(dt) / self.dev.Tscale
        guess = self._u_prev if u_guess is None else np.asarray(u_guess, dtype=float)
        if gen is not None:
            self.set_generation(gen)
        u, rep = self.newton(float(V), guess)
        self._dt = 0.0
        if rep.converged:
            self.time = t_new
            self.snapshot(u, V)
        return u, rep

    def transient(self, u0, times, bias, gen=None, theta: float | None = None):
        """Integrate over ``times`` [s] from the state ``u0`` at ``times[0]``.

        ``bias`` is a callable ``V(t)`` or an array of the same length as
        ``times``; ``gen`` is a callable ``G(t)``, an array, or ``None`` to keep
        whatever field the solver holds.  Returns
        ``(states, terminal current density [A/cm^2], reports)``.  The current is
        the displacement-aware one, i.e. what an ammeter in the external circuit
        reads, not the particle current at one contact.
        """
        times = np.asarray(times, dtype=float)
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        b = (np.array([float(bias(float(t))) for t in times], dtype=float)
             if callable(bias) else np.asarray(bias, dtype=float))
        if b.shape != times.shape:
            raise ValueError("bias must match times in length")
        if theta is not None:
            self.theta = float(theta)
        self.snapshot(u0, float(b[0]))
        states = [np.array(u0, dtype=float, copy=True)]
        currents = [self.total_current(states[0])]
        reports = []
        for k in range(1, len(times)):
            dt = float(times[k] - times[k - 1])
            g = None
            if gen is not None:
                g = float(gen(float(times[k]))) if callable(gen) else gen[k]
            u, rep = self.step(float(b[k]), dt, gen=g)
            if not rep.converged:
                raise RuntimeError(
                    f"transient step {k} failed (t={times[k]:.3e} s, V={b[k]:.4f}, "
                    f"dt={dt:.3e} s, residual {rep.residual:.3e})")
            states.append(u)
            currents.append(self.terminal_current(u, states[-2], dt))
            reports.append(rep)
        return np.array(states), np.array(currents), reports

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

    def displacement_current(self, u: np.ndarray, u_prev: np.ndarray,
                             dt: float) -> np.ndarray:
        """``eps dE/dt`` on every link [A/cm^2].

        The time derivative is the same backward difference the continuity rows
        use, which is what makes the sum of the two rows cancel against it.
        """
        d = self.dev
        p_new = np.asarray(u, dtype=float)[:d.N]
        p_old = np.asarray(u_prev, dtype=float)[:d.N]
        E_new = -(p_new[1:] - p_new[:-1]) * d.mat.Vt / d.h
        E_old = -(p_old[1:] - p_old[:-1]) * d.mat.Vt / d.h
        return d.mat.eps * (E_new - E_old) / float(dt)

    def total_current_profile(self, u, u_prev=None, dt=None, theta=None) -> np.ndarray:
        """Particle plus displacement current on every link [A/cm^2].

        ``Jn + Jp`` alone is not spatially uniform while the field changes.  The
        theta-weighted sum returned here is uniform up to the Newton residual,
        because adding the two discrete continuity rows eliminates SRH and leaves
        exactly the discrete time derivative of the Poisson row.
        """
        Jn, Jp = self.currents(u)
        if u_prev is None or dt is None:
            return Jn + Jp
        th = self.theta if theta is None else float(theta)
        Jn_o, Jp_o = self.currents(u_prev)
        return (th * (Jn + Jp) + (1.0 - th) * (Jn_o + Jp_o)
                + self.displacement_current(u, u_prev, dt))

    def terminal_current(self, u, u_prev=None, dt=None, theta=None) -> float:
        """Uniform total current [A/cm^2], averaged over the links.

        The average is meaningful only because the profile is flat; the transient
        checks measure that spread rather than assume it.
        """
        return float(np.mean(self.total_current_profile(u, u_prev, dt, theta)))

    def current_uniformity(self, u, u_prev=None, dt=None, theta=None) -> float:
        """(max - min) of the total-current profile over the terms it cancels.

        This is the discrete charge-conservation invariant.  It sits at the
        Newton-residual level when the displacement term is present and blows up
        to O(1) as soon as that term is dropped during a field change.

        The denominator is the sum of the particle and displacement magnitudes,
        not ``max|K|``: while the field changes those two terms are individually
        large and their uniform sum is the external current, which for a
        mirror-symmetric mode (a slab between identical contacts) is exactly
        zero by symmetry.  Normalising a round-off-sized residual by it would
        report a correct solve as a failure.  The residual error that Newton
        leaves in the Poisson row enters the profile amplified by
        ``hb / dt_hat``, so this measure degrades as ``1 / dt`` -- expected, and
        the transient checks quote it together with the step size.
        """
        K = self.total_current_profile(u, u_prev, dt, theta)
        Jn, Jp = self.currents(u)
        scale = float(np.max(np.abs(Jn + Jp)))
        if u_prev is not None and dt is not None:
            th = self.theta if theta is None else float(theta)
            Jn_o, Jp_o = self.currents(u_prev)
            scale = (th * scale + (1.0 - th) * float(np.max(np.abs(Jn_o + Jp_o)))
                     + float(np.max(np.abs(self.displacement_current(u, u_prev, dt)))))
        return float((K.max() - K.min()) / max(scale, 1e-300))

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
