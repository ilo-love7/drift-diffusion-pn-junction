"""Case 2 - forward I-V characteristic, ideality factor and voltage partition.

The long-diode Shockley law is only one of three current channels in this
device, so the case decomposes the applied bias as

    V = V_j + V_neutral

where ``V_j = phi_p(x_j) - phi_n(x_j)`` is the separation of the two
quasi-Fermi levels at the metallurgical junction - i.e. the voltage that
actually acts on the junction - and ``V_neutral`` is the remainder, the ohmic
drop carried by the majority currents in the two quasi-neutral regions.  The
electrostatic drop ``psi(0) - psi(L) = Vbi - V`` is reported as a consistency
check.  The three regimes follow directly:

  * low bias   - SRH recombination in the space-charge region dominates,
                 n_id -> 2 and |J| sits far above the diffusion reference,
  * mid bias   - minority diffusion current dominates, n_id -> 1, V_j -> V and
                 the Shockley curve applies,
  * high bias  - the neutral-region resistance takes an increasing share of the
                 bias (V - V_j ~ |J| R_sheet), n_id rises again and |J| falls
                 below Shockley.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p5dd import (DDSolver, ideality_factor, minority_transport,  # noqa: E402
                  reference_device, saturation_current, shockley, viz)
from p5dd.constants import Q_E  # noqa: E402

FIELD_EDGE = 1e-2          # fraction of Emax that delimits the SCR


def voltage_partition(sol, u, dev, V):
    """Junction voltage, neutral-region drop and electrostatic checks.

    ``V_j`` is read from the quasi-Fermi splitting at the junction, which is
    well defined at every bias; the field-threshold SCR edges only give the
    *electrostatic* drop, which becomes ambiguous once the neutral regions
    carry a comparable field.
    """
    st = sol.physical_state(u)
    v_j = float(np.interp(dev.x_j, st["x"], st["phi_p"])
                - np.interp(dev.x_j, st["x"], st["phi_n"]))
    psi_drop = float(st["psi"][0] - st["psi"][-1])

    emag = np.abs(st["E"])
    k = int(np.argmax(emag))
    lim = FIELD_EDGE * emag[k]
    i = k
    while i > 0 and emag[i] > lim:
        i -= 1
    j = k
    while j < len(emag) - 1 and emag[j] > lim:
        j += 1
    left_x = st["xm"][max(i - 1, 0)]
    right_x = st["xm"][min(j + 1, len(emag) - 1)]
    at = lambda xx: float(np.interp(xx, st["x"], st["psi"]))  # noqa: E731
    v_scr = at(left_x) - at(right_x)
    return {"V_j": v_j, "V_neutral": V - v_j, "V_scr_field": v_scr,
            "psi_drop": psi_drop, "scr_centre_um": 0.5 * (left_x + right_x) * 1e4,
            "Emax": float(emag[k])}


def sheet_resistance(dev):
    """Low-level-injection resistance of the two neutral regions [Ohm cm^2].

    Conduction is by the *majority* carriers, so each side uses its own doping.
    """
    mu_n = float(dev.mat.mu_n(np.full(1, dev.Nd))[0])
    mu_p = float(dev.mat.mu_p(np.full(1, dev.Na))[0])
    return float(dev.x_j / (Q_E * mu_n * dev.Nd)
                 + (dev.x[-1] - dev.x_j) / (Q_E * mu_p * dev.Na))


def link_current_spread(sol, u):
    """Absolute spread of the total current along the links [A/cm^2]."""
    Jn, Jp = sol.currents(u)
    return float(np.ptp(Jn + Jp))


def run() -> dict:
    dev = reference_device()
    sol = DDSolver(dev)
    V = np.round(np.concatenate([np.arange(0.0, 0.60, 0.02),
                                 np.arange(0.60, 0.705, 0.005)]), 4)
    U, J, reps = sol.iv_sweep(V)

    Jsh = shockley(dev, V)
    nid = ideality_factor(V, J, dev.mat.T)
    rd = np.gradient(V, J)
    part = [voltage_partition(sol, u, dev, v) for u, v in zip(U, V)]
    v_j = np.array([d["V_j"] for d in part])
    v_neu = np.array([d["V_neutral"] for d in part])
    v_scr = np.array([d["V_scr_field"] for d in part])

    tr = minority_transport(dev)
    Js = saturation_current(dev)
    Rs = sheet_resistance(dev)

    low = (V >= 0.03) & (V <= 0.12)        # SRH-dominated
    mid = (V >= 0.44) & (V <= 0.56)        # diffusion-dominated
    high = (V >= 0.64) & (V <= 0.70)       # series-resistance-dominated
    fit = np.polyfit(v_j[high], np.log(np.abs(J[high])), 1)
    vt_fit = 1.0 / fit[0]
    k50 = int(np.argmin(abs(V - 0.50)))

    metrics = {
        "case": 2, "name": "forward_iv", "n_nodes": dev.N,
        "sweep_points": int(len(V)), "V_max_V": float(V[-1]),
        "Vbi_V": float(dev.Vbi), "V_max_over_Vbi": float(V[-1] / dev.Vbi),
        "max_newton_iterations": max(r.iterations for r in reps),
        "max_weighted_residual": max(r.residual for r in reps),
        "Js_shockley_A_cm2": Js,
        "Ln_um": tr["Ln"] * 1e4, "Lp_um": tr["Lp"] * 1e4,
        "neutral_sheet_resistance_ohm_cm2": Rs,
        "J_at_0p25_A_cm2": float(J[np.argmin(abs(V - 0.25))]),
        "J_at_0p50_A_cm2": float(J[k50]),
        "J_at_0p70_A_cm2": float(J[-1]),
        "ideality_low_bias": float(np.nanmedian(nid[low])),
        "ideality_mid_bias": float(np.nanmedian(nid[mid])),
        "ideality_high_bias": float(np.nanmedian(nid[high])),
        "ideality_min": float(np.nanmin(nid[low | mid | high])),
        "ideality_max": float(np.nanmax(nid[low | mid | high])),
        "lnJ_vs_Vj_slope_perV": float(fit[0]),
        "lnJ_vs_Vj_effective_Vt": float(vt_fit),
        "lnJ_vs_Vj_thermal_Vt": float(dev.mat.Vt),
        "lnJ_vs_Vj_effective_Vt_rel_error": abs(vt_fit - dev.mat.Vt) / dev.mat.Vt,
        "shockley_ratio_at_0p40": float(J[np.argmin(abs(V - 0.4))]
                                        / Jsh[np.argmin(abs(V - 0.4))]),
        "shockley_ratio_at_0p50": float(J[k50] / Jsh[k50]),
        "shockley_ratio_at_0p70": float(J[-1] / Jsh[-1]),
        "neutral_drop_at_0p70_V": float(v_neu[-1]),
        "ohmic_estimate_at_0p70_V": float(abs(J[-1]) * Rs),
        "neutral_drop_vs_ohmic_rel_error": abs(v_neu[-1] - abs(J[-1]) * Rs)
        / abs(v_neu[-1]),
        "psi_drop_vs_Vbi_minus_V_max_dev": max(
            abs(d["psi_drop"] - (dev.Vbi - v)) for d, v in zip(part, V)),
        "junction_voltage_tracks_bias_max_dev_V": float(np.max(np.abs(v_neu[low]))),
        "current_continuity_max_rel_dev": max(
            link_current_spread(sol, u) / abs(j) for u, j in zip(U[1:], J[1:])
            if abs(j) > 1e-3),
        "current_continuity_max_abs_dev_A_cm2": max(
            link_current_spread(sol, u) for u in U[1:]),
    }

    fig, ax = viz.figure(figsize=(7.0, 5.0))
    ax.semilogy(V, np.abs(J), "o-", ms=3, label="solver $|J|$")
    ax.semilogy(V, np.abs(Jsh), "--", label="Shockley long diode")
    ax.axhline(abs(Js), color="#8c564b", lw=1, ls="-.", label="$J_s$ (diffusion prefactor)")
    ax.set_xlabel("V [V]"); ax.set_ylabel("$|J|$ [A/cm$^2$]")
    ax.set_title("Forward I-V of the Si p-n diode")
    ax.legend(fontsize=8); ax.set_xlim(0, 0.71)
    metrics["fig_iv"] = viz.save(fig, "case2_forward_iv.png").name

    fig, ax = viz.figure(figsize=(7.0, 4.4))
    ax.plot(V[1:], nid[1:], lw=2, label="ideality factor $n_{id}$")
    ax.axhline(1.0, color="#888", ls="--", lw=0.8)
    ax.axhline(2.0, color="#888", ls="--", lw=0.8)
    ax.text(0.04, 2.55, "SRH $\\approx$ 2", fontsize=8)
    ax.text(0.30, 1.03, "diffusion $\\approx$ 1", fontsize=8)
    ax.text(0.58, 2.45, "series resistance", fontsize=8)
    ax.set_xlabel("V [V]"); ax.set_ylabel("$n_{id}=V_t^{-1}\\,dV/d\\ln|J|$")
    ax.set_ylim(0, 3.0); ax.set_xlim(0.015, 0.71)
    ax.set_title("Ideality factor: three current regimes")
    b = ax.twinx()
    b.semilogy(V[1:], np.abs(rd[1:]), color="#2ca02c", alpha=0.6, label="$dV/dJ$")
    b.set_ylabel("differential resistance [$\\Omega$ cm$^2$]", color="#2ca02c")
    b.tick_params(axis="y", colors="#2ca02c")
    ax.legend(fontsize=8, loc="upper right"); b.legend(fontsize=8, loc="center right")
    metrics["fig_ideality"] = viz.save(fig, "case2_ideality.png").name

    fig, axes = viz.figure(1, 2, figsize=(10.6, 4.2))
    a = axes[0]
    a.plot(V, v_j, label="$V_j=\\varphi_p-\\varphi_n$ at $x_j$")
    a.plot(V, v_neu, label="$V_{neutral}=V-V_j$")
    a.plot(V, v_scr, ":", color="#17becf", label="$V_{SCR}$ (field edges)")
    a.plot(V, V, ":", color="#888", label="$V$ applied")
    a.set_xlabel("V [V]"); a.set_ylabel("potential drop [V]")
    a.set_title("Bias partition"); a.legend(fontsize=8)
    c = axes[1]
    c.plot(np.abs(J), v_neu, "o-", ms=3, label="$V-V_j$ from solver")
    c.plot(np.abs(J), np.abs(J) * Rs, "--", label="$|J|\\,R_{sheet}$")
    c.set_xlabel("$|J|$ [A/cm$^2$]"); c.set_ylabel("neutral-region drop [V]")
    c.set_title("Ohmic drop of the quasi-neutral regions")
    c.set_xlim(0, 7); c.set_ylim(0, 0.06)
    c.legend(fontsize=8)
    metrics["fig_partition"] = viz.save(fig, "case2_bias_partition.png").name

    fig, axes = viz.figure(1, 2, figsize=(10.6, 4.2))
    a = axes[0]
    for Vk in (0.3, 0.5, 0.7):
        k = int(np.argmin(abs(V - Vk)))
        st = sol.physical_state(U[k])
        eq_n = dev.Nref * dev.r_ni ** 2 / dev.Na
        m = (st["x"] > dev.x_j) & (st["x"] < dev.x_j + 20e-4)
        a.semilogy(st["x"][m] * 1e4, np.maximum(st["n"][m] - eq_n[m], 1e2),
                   label=f"excess n, V={Vk}")
    a.set_xlabel("x [$\\mu$m] (p side)"); a.set_ylabel("$\\Delta n$ [cm$^{-3}$]")
    a.set_title("Injected minority profile decays over $L_n$"); a.legend(fontsize=8)
    c = axes[1]
    k = int(np.argmin(abs(V - 0.6)))
    st = sol.physical_state(U[k])
    c.plot(st["um"], st["phi_n"], label="$\\varphi_n$")
    c.plot(st["um"], st["phi_p"], label="$\\varphi_p$")
    c.plot(st["um"], st["psi"], color="#888", label="$\\varphi$")
    c.axvline(dev.x_j * 1e4, color="k", lw=0.6, alpha=0.4)
    c.set_xlabel("x [$\\mu$m]"); c.set_ylabel("potential [V]")
    c.set_title("Quasi-Fermi levels at V = 0.6 V"); c.legend(fontsize=8)
    metrics["fig_profiles"] = viz.save(fig, "case2_profiles.png").name

    viz.write_csv(zip(V, J, Jsh, nid, rd, v_j, v_neu, v_scr),
                  "case2_forward_iv.csv",
                  ["V_V", "J_A_cm-2", "J_shockley_A_cm-2", "ideality",
                   "diff_resistance_ohm_cm2", "V_junction_V", "V_neutral_V",
                   "V_scr_field_V"])

    out = Path(__file__).resolve().parents[1] / "data" / "metrics_case2.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:42s} {v}")
