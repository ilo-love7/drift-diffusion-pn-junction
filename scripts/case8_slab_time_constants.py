"""Case 8 - transient slab: two analytic time constants and the integrator order.

The solver gained a time term, so the study now has to prove something it never
had to prove before: that a *time* integration is right.  A uniformly doped slab
between two identical Ohmic contacts is the one structure whose transient
eigenmodes have closed-form rates, because the mode shape is a half-sine that
vanishes at both pinned contacts and the discrete Laplacian of a half-sine is
itself (times its eigenvalue).  Two of them are used here.

  * **Space-charge (dielectric) mode** - the potential is perturbed by
    ``A sin(k x)`` and the density perturbation is whatever Poisson demands.
    Linearising the electron continuity pair and eliminating the field gives
    ``d(dn)/dt = -(sigma/eps) dn + Dn dn''``, i.e. the excess space charge drains
    at the dielectric relaxation rate ``sigma/eps`` with a small diffusion
    acceleration.  SRH cannot touch it: separating charge by ``dn`` costs
    ``dn/tau`` only through the minority density, which is ten orders of
    magnitude smaller.
  * **Quasi-neutral plasma mode** - ``dn = dp``, so Poisson is satisfied by an
    unperturbed potential and the pair recombines.  For a perturbation small
    against *both* equilibrium densities SRH linearises to ``R = dn / tau``
    (``np - ni^2 ~ dn (n0 + p0)`` while the denominator stays ``~ tau (n0 + p0)``),
    and the plasma diffuses with the ambipolar coefficient ``D_a``, giving
    ``1/tau + D_a k^2``.  How small is "small" is itself measured here, by an
    injection-level scan.

The two rates differ by five orders of magnitude (48 fs against 10 ns) while the
solver's own time unit ``Lambda^2 / Dref`` is 0.29 ns, so a single unit covers
both and each mode is measured with its own step size.

The stopping test is one number - the largest residual divided by the reference
rate - so it has to be chosen per mode, not per case: the whole content of the
plasma mode is ~1e-9 of the reference at low injection, which a tolerance of 1e-8
would accept without moving a single carrier, while the dielectric mode at the
tiny step sizes its stiffness demands leaves a round-off floor of ~1e-8 behind.
Both tolerances are therefore taken from a scan rather than from taste, and the
case reports the achieved residual next to the tolerance it used.  What makes the
result trustworthy is not the tolerance but the invariants measured on the state:
the amplitude must decay as a single exponential (fit rms), the extracted rate
must not depend on the fit window, on the perturbation amplitude (linear
response), or - within its own scan - on the tolerance.

Time-integration error is separated from physics with the exact amplification
factor of the theta scheme on ``y' = -lambda y``: a measured rate is a known
function of the true one, invertible in closed form, so the check is made both
raw (which exposes the O(dt) lag of backward Euler and the O(dt^2) lag of
Crank-Nicolson as convergence *orders*) and corrected (which must land on the
analytic rate).

The third quantity is the one the transient machinery was really built for:
while the field moves, the particle current alone is *not* conserved, and adding
``eps dE/dt`` makes it uniform.  In this slab the uniform total current is also
exactly zero by mirror symmetry, so the invariant can be stated as "both terms
are large, their sum vanishes" - a much sharper test than any bound on a
non-zero current.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p5dd import (DDSolver, ambipolar_decay, charge_mode_state,  # noqa: E402
                  convergence_order, dielectric_relaxation, fit_decay_rate,
                  homogeneous_device, mode_amplitude, mode_wavenumber,
                  plasma_mode_state, true_rate, viz)

# L1 = an analytic quantity the solved state must reproduce; L2 = a refinement
# limit the answer must sit in.  Each threshold is set a decade or two above what
# the measured value actually is, so a real regression fails without the gates
# being loose enough to hide a wrong answer.
GATES = {
    "equilibrium_np_over_ni2_max_dev": 1e-12,
    "equilibrium_residual": 1e-8,
    "charge_mode_poisson_row_max": 1e-10,
    "plasma_mode_poisson_row_max": 1e-10,
    "dielectric_rate_rel_error_max": 1e-4,
    "dielectric_order_dev_theta1.0": 0.1,
    "dielectric_order_dev_theta0.5": 0.1,
    "dielectric_fit_rms_max": 1e-4,
    "dielectric_rate_spread_over_amplitude": 1e-4,
    "dielectric_rate_window_spread": 1e-4,
    "plasma_rate_rel_error_max": 1e-3,
    "plasma_rate_tol_converged": 1e-4,
    "plasma_low_injection_rate_spread": 1e-4,
    "uniformity_with_displacement_max": 1e-3,
    "terminal_current_over_terms_max": 1e-6,
    "atom_max_state_dev_V": 1e-9,
    "atom_current_rel_dev": 1e-6,
    "resistor_J_rel_dev": 1e-6,
    "resistor_potential_linearity_dev_V": 1e-9,
    "resistor_density_uniformity_rel": 1e-6,
    "chains_with_zero_newton_iterations": 0.5,
}
# Quantities a working solver must make *large*: they say a small term was
# resolved rather than ignored.
GATES_MIN = {
    "plasma_error_if_diffusion_ignored_max": 5e-3,
    "uniformity_ratio_displacement_gain": 1e3,
}



def mode_fit(t, a, dt, theta, lam, lo=0.3, hi=3.0):
    """Fit a decay trace over the ``[lo, hi]`` e-fold window and undo the ``dt`` lag.

    Returns ``(rate, rel_error, rms, steps_in_window, measured_rate)``.  The
    window is stated in
    e-folds rather than in steps so that the same physical part of the trace is
    fitted at every step size; the first 0.3 e-folds are dropped because the
    initial state is built as a pure shape, and the linearised system's own fast
    component (the dielectric relaxation, 2e5 times quicker here) settles inside
    it.  Late times are dropped because the row content decays with the mode, so
    the fixed Newton tolerance eventually dominates it.
    """
    i0 = int(np.ceil(lo / (lam * dt)))
    i1 = min(int(np.ceil(hi / (lam * dt))), len(t) - 1)
    fit = fit_decay_rate(t[i0:i1 + 1], a[i0:i1 + 1])
    rate = true_rate(fit["rate"], dt, theta)
    return rate, abs(rate / lam - 1.0), fit["rms_rel"], i1 - i0, fit["rate"]


def run_chain(dev, u_eq, u0, dt, n_steps, theta, component="psi", k=None,
              tol=1e-8):
    """Sample one mode amplitude through ``n_steps`` theta-scheme steps at ``V = 0``.

    Also records, per step, the total-current profile statistics that decide the
    displacement-current verdict: the spread with and without the displacement
    term, both normalised by the magnitudes of the terms that cancel in it.

    ``tol`` is passed per mode (see the module docstring) and the achieved
    residual is reported beside it, so a tolerance that was never approached is
    visible rather than assumed.
    """
    sol = DDSolver(dev, tol=tol, theta=theta)
    u = sol.snapshot(np.asarray(u0, dtype=float), 0.0)
    times = [0.0]
    amps = [mode_amplitude(dev, u, u_eq, component, k)]
    stats = []
    iters, residuals = [], []
    for _ in range(n_steps):
        u_prev = u
        u, rep = sol.step(0.0, dt)
        if not rep.converged:
            raise RuntimeError(f"transient step failed at t={sol.time:.3e} s "
                               f"(residual {rep.residual:.3e})")
        iters.append(rep.iterations)
        residuals.append(rep.residual)
        times.append(sol.time)
        amps.append(mode_amplitude(dev, u, u_eq, component, k))
        Jn, Jp = sol.currents(u)
        particle = Jn + Jp
        disp = sol.displacement_current(u, u_prev, dt)
        scale = float(np.max(np.abs(particle)) + np.max(np.abs(disp)))
        K = sol.total_current_profile(u, u_prev, dt)
        stats.append({"t": sol.time, "scale": scale,
                      "uniformity_disp": float(np.ptp(K) / scale),
                      "uniformity_nodisp": float(np.ptp(particle) / scale),
                      "K_over_scale": float(np.max(np.abs(K)) / scale),
                      "J_terminal": float(np.mean(K))})
    return (np.array(times), np.array(amps), stats,
            {"steps": len(stats), "min_iterations": min(iters),
             "max_iterations": max(iters),
             "max_weighted_residual": max(residuals)})


def run() -> dict:
    dev = homogeneous_device(n_doping=1.0e17, n_nodes=401)
    sol = DDSolver(dev, tol=1e-9)
    u_eq = sol.solve_point(0.0)[0]
    n, p = dev.densities(u_eq)
    ni = dev.r_ni[0] * dev.Nref
    k = mode_wavenumber(dev)
    metrics = {"case": 8, "name": "slab_time_constants",
               "n_nodes": dev.N, "length_um": float(dev.x[-1] * 1e4),
               "Nd_cm-3": float(dev.Nd), "Tscale_s": dev.Tscale,
               "alpha": dev.alpha}

    # --- slab equilibrium, the zeroth-order check ----------------------------
    # Mass action in physical units: `densities` is dimensionless (n0 = 1 here is
    # Nd = 1e17 cm^-3), so both carriers must be scaled back before dividing by
    # ni^2 in cm^-6.
    metrics["equilibrium_np_over_ni2_max_dev"] = float(
        np.max(np.abs(n * p * dev.Nref ** 2 / ni ** 2 - 1.0)))
    Jn, Jp = sol.currents(u_eq)
    metrics["equilibrium_max_current_A_cm2"] = float(np.max(np.abs(Jn + Jp)))
    metrics["equilibrium_potential_flatness_V"] = float(
        np.max(np.abs(u_eq[:dev.N] - dev.psi_left)) * dev.mat.Vt)
    metrics["equilibrium_residual"] = float(sol.newton(
        0.0, u_eq)[1].residual)

    # --- analytic rates ------------------------------------------------------
    dl = dielectric_relaxation(dev, k)
    am = ambipolar_decay(dev, k)
    tau_d = 1.0 / dl["lambda"]
    tau_pl = 1.0 / am["lambda"]
    metrics.update({"sigma_S_cm-1": dl["sigma"], "Dn_cm2_s-1": dl["Dn"],
                    "D_a_cm2_s-1": am["D_a"], "k_cm-1": k, "k_h_cm-1": dl["k_h"],
                    "tau_dielectric_s": tau_d,
                    "lambda_dielectric_s-1": dl["lambda"],
                    "tau_plasma_s": tau_pl,
                    "lambda_plasma_s-1": am["lambda"],
                    "plasma_diffusion_share": am["lambda"] * am["tau"] - 1.0})

    # --- A. dielectric relaxation, both integrators, four step sizes ---------
    # The amplitude is a potential perturbation in units of Vt.  It has to be
    # large enough that the mode's own row content (~lambda_hat * dn) stays well
    # above the stopping tolerance for the whole trace: measured, an amplitude of
    # 1e-5 leaves a content of ~7e-9, which a tolerance of 1e-8 accepts untouched
    # and the carriers stand still, while 1e-3 and above decays at the analytic
    # rate.  5e-2 (= 1.3 mV) is used, and the rate is then checked to be
    # amplitude-independent, which is what makes the linear theory applicable.
    amp_d = 5.0e-2
    u_charge = charge_mode_state(dev, u_eq, amp_d, k)
    metrics["charge_mode_poisson_row_max"] = float(np.max(np.abs(
        sol.residual(u_charge, 0.0)[:dev.N - 1])))
    raw, corr, best = {}, {}, None
    for theta in (1.0, 0.5):
        dts, errs = [], []
        for f in (0.006, 0.012, 0.025, 0.05):
            dt = f / dl["lambda"]
            t, a, stats, diag = run_chain(dev, u_eq, u_charge, dt,
                                          int(np.ceil(3.0 / f)), theta)
            for key, value in diag.items():
                metrics[f"dielectric_theta{theta}_f{f:g}_{key}"] = value
            rate, err, rms, _, measured = mode_fit(t, a, dt, theta, dl["lambda"])
            raw[round(f, 4)] = measured
            corr[round(f, 4)] = rate
            dts.append(dt)
            errs.append(abs(measured / dl["lambda"] - 1.0))
            if theta == 1.0 and f == 0.05:
                best = stats
            metrics[f"dielectric_lag_theta{theta}_f{f:g}"] = rate / dl["lambda"] - 1.0
            metrics[f"dielectric_rate_rel_error_theta{theta}_f{f:g}"] = err
            metrics[f"dielectric_fit_rms_theta{theta}_f{f:g}"] = rms
        metrics[f"dielectric_order_theta{theta}"] = convergence_order(dts, errs)
    metrics["dielectric_rate_rel_error_max"] = max(
        abs(v / dl["lambda"] - 1.0) for v in corr.values())
    metrics["dielectric_rate_rel_error_coarsest"] = abs(
        corr[0.05] / dl["lambda"] - 1.0)
    metrics["dielectric_raw_ratio_coarsest"] = raw[0.05] / dl["lambda"]
    # Linear response: a rate that is a property of the *system* must not move
    # when the perturbation that probes it does.  Two decades of amplitude.
    rates_amp = []
    for amp in (1.0e-3, 1.0e-2, 5.0e-2):
        t, a, _, _ = run_chain(dev, u_eq, charge_mode_state(dev, u_eq, amp, k),
                              0.05 / dl["lambda"], 60, 1.0)
        rates_amp.append(mode_fit(t, a, 0.05 / dl["lambda"], 1.0, dl["lambda"])[0])
    metrics["dielectric_rate_spread_over_amplitude"] = float(
        max(rates_amp) / min(rates_amp) - 1.0)
    # Fit-window stability: the same trace fitted from 0.3 and from 1 e-fold.
    t_w, a_w, _, _ = run_chain(dev, u_eq, u_charge, 0.05 / dl["lambda"], 110, 1.0)
    metrics["dielectric_rate_window_spread"] = float(abs(
        mode_fit(t_w, a_w, 0.05 / dl["lambda"], 1.0, dl["lambda"], 0.3, 3.0)[0]
        - mode_fit(t_w, a_w, 0.05 / dl["lambda"], 1.0, dl["lambda"], 1.0, 3.0)[0])
        / dl["lambda"])

    # --- B. quasi-neutral plasma mode ----------------------------------------
    # delta is the density perturbation amplitude in cm^-3, converted to the
    # solver's units.  1e13 is used for the rate gate: it is 1e-4 of the majority
    # density and 0.1 of the equilibrium minority density, and the injection scan
    # below shows that is where the single-exponential, amplitude-independent
    # answer lives.  The tolerance is 100 times tighter than the dielectric one
    # because this mode's entire row content is ~3e-6 * delta/delta_ref.
    delta_ref = 1.0e13 / dev.Nref
    u_plasma = plasma_mode_state(dev, u_eq, delta_ref, k)
    metrics["plasma_mode_poisson_row_max"] = float(np.max(np.abs(
        sol.residual(u_plasma, 0.0)[:dev.N - 1])))
    pl_corr, pl_err_vs_tau = [], []
    for f in (0.01, 0.02, 0.05):
        dt = f / am["lambda"]
        t, a, _, diag = run_chain(dev, u_eq, u_plasma, dt,
                                  int(np.ceil(3.0 / f)), 1.0, component="n",
                                  tol=1e-12)
        for key, value in diag.items():
            metrics[f"plasma_f{f:g}_{key}"] = value
        rate, err, rms, _, _ = mode_fit(t, a, dt, 1.0, am["lambda"])
        pl_corr.append(rate)
        pl_err_vs_tau.append(abs(rate * am["tau"] - 1.0))
        metrics[f"plasma_rate_f{f:g}_s-1"] = rate
        metrics[f"plasma_rate_rel_error_f{f:g}"] = err
        metrics[f"plasma_fit_rms_f{f:g}"] = rms
    metrics["plasma_rate_rel_error_max"] = max(
        abs(v / am["lambda"] - 1.0) for v in pl_corr)
    metrics["plasma_error_if_diffusion_ignored_max"] = max(pl_err_vs_tau)
    # Injection level: how far below n0 does "small" have to be for R = dn/tau?
    inj = {}
    for delta_cm3 in (1.0e12, 1.0e13, 1.0e14, 1.0e15):
        dt = 0.05 / am["lambda"]
        t, a, _, _ = run_chain(dev, u_eq,
                               plasma_mode_state(dev, u_eq, delta_cm3 / dev.Nref, k),
                               dt, 110, 1.0, component="n", tol=1e-12)
        rate, err, rms, _, _ = mode_fit(t, a, dt, 1.0, am["lambda"])
        inj[delta_cm3] = (rate / am["lambda"], rms)
        metrics[f"plasma_injection{delta_cm3:.0e}_rate_over_lambda"] = rate / am["lambda"]
        metrics[f"plasma_injection{delta_cm3:.0e}_fit_rms"] = rms
    metrics["plasma_low_injection_rate_spread"] = float(
        max(v[0] for k_, v in inj.items() if k_ <= 1.0e13)
        - min(v[0] for k_, v in inj.items() if k_ <= 1.0e13))
    for tol in (1.0e-10, 1.0e-12):
        t, a, _, _ = run_chain(dev, u_eq, u_plasma, 0.05 / am["lambda"], 110, 1.0,
                               component="n", tol=tol)
        metrics[f"plasma_rate_tol{tol:.0e}_s-1"] = mode_fit(
            t, a, 0.05 / am["lambda"], 1.0, am["lambda"])[0]
    metrics["plasma_rate_tol_converged"] = float(abs(
        metrics["plasma_rate_tol1e-12_s-1"]
        / metrics["plasma_rate_tol1e-10_s-1"] - 1.0))


    # --- C. displacement current: the conserved quantity ---------------------
    scale = best[-1]["scale"]
    metrics["uniformity_with_displacement_max"] = max(s["uniformity_disp"] for s in best)
    metrics["uniformity_without_displacement_max"] = max(s["uniformity_nodisp"] for s in best)
    metrics["total_current_over_terms_max"] = max(s["K_over_scale"] for s in best)
    metrics["terminal_current_over_terms_max"] = max(
        abs(s["J_terminal"]) / s["scale"] for s in best)
    metrics["uniformity_ratio_displacement_gain"] = (
        metrics["uniformity_without_displacement_max"]
        / metrics["uniformity_with_displacement_max"])

    # --- D. one huge step must reproduce the steady solve (the atom test) ----
    V_probe = 0.05
    u_ref_steady, rep_steady = sol.solve_point(V_probe)
    big = 1.0e6 * tau_d
    sol_big = DDSolver(dev, tol=1e-9, theta=1.0)
    sol_big.snapshot(u_eq, 0.0)
    u_big, rep_big = sol_big.step(V_probe, big)
    n_st, p_st = dev.densities(u_ref_steady)
    metrics["atom_step_converged"] = bool(rep_big.converged)
    metrics["atom_max_state_dev_V"] = float(np.max(np.abs(
        (u_big[:dev.N] - u_ref_steady[:dev.N]))) * dev.mat.Vt)
    metrics["atom_density_rel_dev"] = float(
        np.max(np.abs(dev.densities(u_big)[0] / n_st - 1.0)))
    metrics["atom_current_rel_dev"] = abs(
        sol.terminal_current(u_big) / sol.total_current(u_ref_steady) - 1.0)

    # --- E. the biased slab is an exact resistor -----------------------------
    sigma = dl["sigma"]
    L = dev.x[-1] - dev.x[0]
    # A uniformly doped slab under bias is a resistor: Poisson is solved by a
    # linear potential with both densities left at their equilibrium values, so
    # the drift current q (mu_n n + mu_p p) V / L is exact.  The minus sign is
    # the solver's convention -- the same one that makes forward diode bias
    # negative (see theory.shockley) -- because the right contact is the one the
    # bias is applied to.
    J_analytic = -sigma * V_probe / L
    J_solver = sol.total_current(u_ref_steady)
    metrics["resistor_J_analytic_A_cm2"] = float(J_analytic)
    metrics["resistor_J_solver_A_cm2"] = float(J_solver)
    metrics["resistor_J_rel_dev"] = float(abs(J_solver / J_analytic - 1.0))
    metrics["resistor_passive_sign_ok"] = bool(J_solver * V_probe < 0.0)
    psi = u_ref_steady[:dev.N]
    linear = psi[0] + (psi[-1] - psi[0]) * (dev.x - dev.x[0]) / L
    metrics["resistor_potential_linearity_dev_V"] = float(
        np.max(np.abs(psi - linear)) * dev.mat.Vt)
    metrics["resistor_density_uniformity_rel"] = float(
        np.max(np.abs(n_st / n_st[0] - 1.0)))

    # --- figures and raw data ------------------------------------------------
    t1, a1, s1, _ = run_chain(dev, u_eq, u_charge, 0.05 / dl["lambda"], 60, 1.0)
    t2, a2, _, _ = run_chain(dev, u_eq, u_plasma, 0.05 / am["lambda"], 60, 1.0,
                             component="n", tol=1e-12)
    fig, axes = viz.figure(1, 3, figsize=(13.4, 4.0))
    a = axes[0]
    a.semilogy(t1 / tau_d, np.abs(a1) / abs(a1[0]), "o-", ms=3,
               label="solver $|\\psi_1(t)/\\psi_1(0)|$")
    a.semilogy(t1 / tau_d, np.exp(-t1 / tau_d), "--",
               label="$e^{-t/\\tau_d}$, $\\tau_d=48$ fs")
    a.set_xlabel("$t/\\tau_d$"); a.set_ylabel("mode amplitude")
    a.set_title("Dielectric relaxation"); a.legend(fontsize=8)
    c = axes[1]
    c.semilogy(t2 * am["lambda"], np.abs(a2) / abs(a2[0]), "s-", ms=3,
               label="solver $|\\delta n_1(t)|$")
    c.semilogy(t2 * am["lambda"], np.exp(-t2 * am["lambda"]), "--",
               label="$e^{-t/\\tau_{\\mathrm{eff}}}$, 1/tau+D$_a$k$^2$")
    c.set_xlabel("$t/\\tau_{\\mathrm{eff}}$"); c.set_title("Plasma decay")
    c.legend(fontsize=8)
    b = axes[2]
    ts = np.array([s["t"] for s in s1]) / tau_d
    b.semilogy(ts, [s["uniformity_disp"] for s in s1], "o-", ms=3,
               label="particle + displacement")
    b.semilogy(ts, [s["uniformity_nodisp"] for s in s1], "s--", ms=3,
               label="particle current only")
    b.semilogy(ts, [s["K_over_scale"] for s in s1], ":",
               label="$|J_{\\mathrm{tot}}|$/terms (must vanish)")
    b.set_xlabel("$t/\\tau_d$"); b.set_ylabel("spread / term magnitudes")
    b.set_title("Displacement current restores conservation")
    b.legend(fontsize=8); b.set_ylim(1e-9, 4)
    metrics["fig_slab"] = viz.save(fig, "case8_slab_transients.png").name

    state_at = [0, 1, 3, 10, 30, 60]
    rows = []
    for i in state_at:
        st = s1[min(i, len(s1) - 1)] if i else None
        rows.append((i, t1[min(i, len(t1) - 1)], a1[min(i, len(a1) - 1)],
                     st["scale"] if st else "", st["uniformity_disp"] if st else "",
                     st["uniformity_nodisp"] if st else "",
                     st["K_over_scale"] if st else "",
                     st["J_terminal"] if st else ""))
    viz.write_csv(rows, "case8_slab_transients.csv",
                  ["step", "t_s", "psi_mode_amplitude_Vt", "terms_A_cm-2",
                   "uniformity_with_disp", "uniformity_without_disp",
                   "total_over_terms", "terminal_current_A_cm-2"])

    # --- gates ---------------------------------------------------------------
    metrics["dielectric_order_dev_theta1.0"] = abs(
        metrics["dielectric_order_theta1.0"] - 1.0)
    metrics["dielectric_order_dev_theta0.5"] = abs(
        metrics["dielectric_order_theta0.5"] - 2.0)
    metrics["dielectric_fit_rms_max"] = max(
        v for k, v in metrics.items()
        if k.startswith("dielectric_fit_rms_"))
    # A chain Newton accepted without a single iteration is a chain that never
    # integrated anything - the failure mode the per-mode tolerance exists to
    # avoid, so it is counted rather than trusted.
    metrics["chains_with_zero_newton_iterations"] = int(sum(
        1 for k, v in metrics.items()
        if k.endswith("_min_iterations") and v == 0))
    gates_passed = {k: bool(metrics[k] < GATES[k]) for k in GATES}
    gates_passed.update({k: bool(metrics[k] > GATES_MIN[k]) for k in GATES_MIN})
    metrics["gates"] = GATES
    metrics["gates_min"] = GATES_MIN
    metrics["gates_passed"] = gates_passed
    metrics["gates_failed"] = sorted(k for k, v in gates_passed.items() if not v)
    metrics["all_gates_passed"] = all(gates_passed.values())

    out = Path(__file__).resolve().parents[1] / "data" / "metrics_case8.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    for key, value in run().items():
        print(f"{key:44s} {value}")
