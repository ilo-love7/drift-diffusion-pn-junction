"""Case 3 - reverse bias: SRH generation current and Mott-Schottky C-V.

Below a few tenths of a volt of reverse bias the diffusion (Shockley) current
has already saturated at ``-J_s``, so whatever the solver carries there is
space-charge generation.  Two references are used:

  * the exact discrete balance ``J = -q \\int R dx`` of the solved SRH rate,
    which tests the solver against itself and must hold to solver accuracy,
  * the textbook bound ``q n_i W /(tau_n + tau_p)``, which assumes the whole
    depletion width generates at its limiting rate ``n_i/(tau_n+tau_p)``.

The second one turns out to be an over-estimate by 3-9x, and the case shows
why: the SRH rate only reaches its limit where *both* carriers fall below
``n_i``, while the Boltzmann tails keep the majority carrier above ``n_i``
through most of the depletion region.  The width of that doubly-depleted
window, not the depletion width, sets the generation current.

The depletion capacitance is obtained from the solved charge,
``C = |dQ/dV|`` with ``Q = eps E_max`` (Gauss), and analysed as a
Mott-Schottky plot: 1/C^2 is linear in V, its slope gives the effective doping
``2/(q eps Na Nd/(Na+Nd))`` and its voltage intercept gives Vbi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p5dd import (DDSolver, capacitance, depletion_charge,  # noqa: E402
                  expected_capacitance, integrate_on_mesh, mott_schottky,
                  net_recombination_rate, peak_field, reference_device,
                  saturation_current, viz)
from p5dd.constants import Q_E  # noqa: E402
from p5dd.theory import depletion, ni_local  # noqa: E402


def generation_current(dev, V):
    """Textbook SRH generation reference q ni W /(tau_n + tau_p) [A/cm^2]."""
    ni = 0.5 * (ni_local(dev, "n") + ni_local(dev, "p"))
    out = [Q_E * ni * depletion(dev, float(vv))["W"]
           / (dev.mat.tau_n + dev.mat.tau_p) for vv in np.atleast_1d(V)]
    return np.asarray(out)


def run() -> dict:
    dev = reference_device()
    sol = DDSolver(dev)
    V = np.round(np.arange(0.0, -0.7001, -0.01), 4)
    U, J, reps = sol.iv_sweep(V)

    states = [sol.physical_state(u) for u in U]
    emax = np.array([peak_field(s) for s in states])
    Q = np.array([depletion_charge(dev, s) for s in states])
    C = capacitance(V, Q)
    W_cv = dev.mat.eps / C
    W_th = np.array([depletion(dev, v)["W"] for v in V])
    Jgen = generation_current(dev, V)
    Js = saturation_current(dev)
    ms = mott_schottky(V, C, dev.mat.eps)
    n_eff_true = dev.Na * dev.Nd / (dev.Na + dev.Nd)

    # --- where the generation actually happens -------------------------------
    # R = (np - ni^2)/(tau_p(n+ni) + tau_n(p+ni)) reaches its limiting value
    # -ni/(tau_n+tau_p) only where *both* carriers are below ni.  Because the
    # Boltzmann tails keep the majority carrier above ni deep into the
    # depletion region, the naive "whole DA width generates" reference is an
    # upper bound, so the solved current is compared with the integral of the
    # actual rate and with the width of the doubly-depleted window.
    ni_node = dev.r_ni * dev.Nref
    r_max = ni_node[0] / (dev.mat.tau_n + dev.mat.tau_p)      # cm^-3 s^-1
    rates = [net_recombination_rate(dev, s) for s in states]
    gen_int = np.array([-Q_E * integrate_on_mesh(r, dev.x) for r in rates])
    both = [(s["n"] < ni_node) & (s["p"] < ni_node) for s in states]
    w_both = np.array([integrate_on_mesh(b.astype(float), dev.x) for b in both])
    w_gen = np.abs(J) * (dev.mat.tau_n + dev.mat.tau_p) / (Q_E * ni_node[0])

    k0 = int(np.argmin(abs(V)))                    # V = 0 (first sweep point)
    at = lambda v: int(np.argmin(abs(V - v)))      # noqa: E731
    metrics = {
        "case": 3, "name": "reverse_cv", "n_nodes": dev.N,
        "sweep_points": int(len(V)), "V_min_V": float(V[-1]),
        "max_newton_iterations": max(r.iterations for r in reps),
        "max_weighted_residual": max(r.residual for r in reps),
        "Js_shockley_A_cm2": Js,
        "J_rev_at_-0p10_A_cm2": float(J[at(-0.10)]),
        "J_rev_at_-0p30_A_cm2": float(J[at(-0.30)]),
        "J_rev_at_-0p70_A_cm2": float(J[-1]),
        "Jgen_textbook_at_-0p70": float(Jgen[-1]),
        "generation_ratio_at_-0p10": float(J[at(-0.10)] / Jgen[at(-0.10)]),
        "generation_ratio_at_-0p30": float(J[at(-0.30)] / Jgen[at(-0.30)]),
        "generation_ratio_at_-0p70": float(J[-1] / Jgen[-1]),
        "srh_peak_rate_over_limit_at_-0p70": float(
            np.max(-rates[-1]) / r_max),
        "generation_balance_max_rel_dev": float(np.max(
            np.abs(J[1:] - gen_int[1:]) / np.abs(J[1:]))),
        "W_doubly_depleted_at_-0p70_um": float(w_both[-1] * 1e4),
        "W_gen_effective_at_-0p70_um": float(w_gen[-1] * 1e4),
        "W_gen_over_doubly_depleted": float(w_gen[-1] / w_both[-1]),
        "W_gen_over_DA_at_-0p70": float(w_gen[-1] / W_th[-1]),
        "reverse_over_saturation_at_-0p70": float(abs(J[-1] / Js)),
        "Emax_at_-0p70_V_cm": float(emax[-1]),
        "Emax_at_0_V_cm": float(emax[k0]),
        "W_cv_at_0_um": float(W_cv[k0] * 1e4),
        "W_theory_at_0_um": float(W_th[k0] * 1e4),
        "W_cv_at_-0p70_um": float(W_cv[-1] * 1e4),
        "W_theory_at_-0p70_um": float(W_th[-1] * 1e4),
        "W_rel_dev_max": float(np.max(np.abs(W_cv - W_th) / W_th)),
        "C_at_0_F_cm-2": float(C[k0]),
        "C_expected_at_0_F_cm-2": float(expected_capacitance(dev, 0.0)[0]),
        "mott_schottky_doping_eff_cm-3": ms["doping_eff"],
        "mott_schottky_doping_true_cm-3": float(n_eff_true),
        "mott_schottky_doping_rel_error": abs(ms["doping_eff"] - n_eff_true)
        / n_eff_true,
        "mott_schottky_Vbi_fit_V": ms["Vbi_fit"],
        "mott_schottky_Vbi_true_V": float(dev.Vbi),
        "mott_schottky_Vbi_abs_error_V": abs(ms["Vbi_fit"] - dev.Vbi),
        "mott_schottky_R2": ms["R2"],
    }

    fig, axes = viz.figure(1, 2, figsize=(10.6, 4.4))
    a = axes[0]
    a.semilogy(-V, np.abs(J), "o-", ms=3, label="solver $|J|$")
    a.semilogy(-V, np.abs(gen_int), "x", ms=4, label="$-q\\int R\\,dx$ of solved $R$")
    a.semilogy(-V, Jgen, "--", label="$q n_i W/(\\tau_n+\\tau_p)$ (DA bound)")
    a.semilogy(-V, np.full_like(V, abs(Js)), ":", color="#8c564b",
               label="$J_s$ (Shockley saturation)")
    a.set_xlabel("$-V$ [V]"); a.set_ylabel("$|J|$ [A/cm$^2$]")
    a.set_title("Reverse current is generation current")
    a.legend(fontsize=8); a.set_xlim(0, 0.71)
    c = axes[1]
    c.plot(-V, w_gen * 1e4, "o-", ms=3, label="$|J|(\\tau_n+\\tau_p)/q n_i$")
    c.plot(-V, w_both * 1e4, "--", label="doubly-depleted window")
    c.plot(-V, W_th * 1e4, ":", color="#8c564b", label="$W_{DA}$")
    c.set_xlabel("$-V$ [V]"); c.set_ylabel("width [$\\mu$m]")
    c.set_title("Only the doubly-depleted window generates")
    c.legend(fontsize=8); c.set_xlim(0, 0.71)
    metrics["fig_reverse"] = viz.save(fig, "case3_reverse_iv.png").name

    fig, axes = viz.figure(1, 2, figsize=(10.6, 4.2))
    a = axes[0]
    a.plot(-V, C * 1e6, "o-", ms=3, label="$|dQ/dV|$ from solver")
    a.plot(-V, expected_capacitance(dev, V) * 1e6, "--", label="$\\varepsilon/W_{DA}$")
    a.set_xlabel("$-V$ [V]"); a.set_ylabel("$C$ [$\\mu$F/cm$^2$]")
    a.set_title("Depletion capacitance"); a.legend(fontsize=8)
    c = axes[1]
    y = 1.0 / C ** 2
    c.plot(-V, y * 1e-8, "o-", ms=3, label="$1/C^2$")
    fit = ms["slope"] * (-V) + ms["intercept"]
    c.plot(-V, fit * 1e-8, "--", label=f"linear fit, $R^2$={ms['R2']:.4f}")
    c.annotate(f"$V_{{bi}}$ fit = {ms['Vbi_fit']:.3f} V\ntrue {dev.Vbi:.3f} V",
               xy=(0.02, y.min() * 1e-8), xytext=(0.25, y.min() * 1.35e-8), fontsize=8)
    c.set_xlabel("$-V$ [V]"); c.set_ylabel("$1/C^2$ [$10^{8}$ cm$^4$ F$^{-2}$]")
    c.set_title("Mott-Schottky plot"); c.legend(fontsize=8)
    metrics["fig_cv"] = viz.save(fig, "case3_cv_mott_schottky.png").name

    viz.write_csv(zip(V, J, emax, Q, C, W_cv, W_th),
                  "case3_reverse_cv.csv",
                  ["V_V", "J_A_cm-2", "Emax_V_cm-1", "Q_C_cm-2", "C_F_cm-2",
                   "W_cv_cm", "W_theory_cm"])

    out = Path(__file__).resolve().parents[1] / "data" / "metrics_case3.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:42s} {v}")
