"""Case 6 - physically realistic silicon: doping-dependent mobility and BGN.

The reference device switches both models off so that the analytic and Sesame
comparisons are exact.  A real Si p-n diode at these dopings has

  * Arora mobility:      mu_n(1e17) ~ 780 instead of 1350 cm2/(V s),
  * Slotboom BGN:        the gap shrinks by ~24 meV, raising n_i locally,

and both feed back into Vbi, the depletion width and the current.  This case
quantifies that model sensitivity - it is a physics result, not a solver
error - and re-checks the equilibrium invariant with the *local* intrinsic
density, which is what the solver must satisfy once the gap varies with x.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p5dd import (DDSolver, built_in_potential, depletion,  # noqa: E402
                  ideality_factor, minority_transport, reference_device,
                  realistic_device, saturation_current, shockley, viz)

PROBE_BIAS = (0.30, 0.45, 0.60)


def sweep(dev, V):
    sol = DDSolver(dev)
    U, J, reps = sol.iv_sweep(V)
    st0 = sol.physical_state(U[0])
    ni_local = dev.r_ni * dev.Nref
    return {
        "sol": sol, "V": V, "J": J, "U": U, "reps": reps,
        "Vbi": float(built_in_potential(dev)),
        "invariant": float(np.max(np.abs(st0["n"] * st0["p"] / ni_local ** 2 - 1.0))),
        "Js": float(saturation_current(dev)),
        "transport": minority_transport(dev),
        "Emax": float(np.max(np.abs(st0["E"]))),
        "W": float(depletion(dev, 0.0)["W"]),
        "ni_enh": float(np.max(ni_local) / dev.mat.ni0()),
        "mu_n_bulk": float(dev.mat.mu_n(np.full(1, dev.Nd))[0]),
        "mu_p_bulk": float(dev.mat.mu_p(np.full(1, dev.Na))[0]),
    }


def run() -> dict:
    V = np.round(np.arange(0.0, 0.6001, 0.02), 4)
    ref = sweep(reference_device(), V)
    rsl = sweep(realistic_device(), V)

    ratio = np.abs(rsl["J"] / ref["J"])
    nid_ref = ideality_factor(V, ref["J"], 300.0)
    nid_rsl = ideality_factor(V, rsl["J"], 300.0)
    mid = (V >= 0.30) & (V <= 0.50)

    metrics = {
        "case": 6, "name": "realistic_device", "n_nodes": ref["sol"].N,
        "sweep_points": int(len(V)),
        "Vbi_reference_V": ref["Vbi"], "Vbi_realistic_V": rsl["Vbi"],
        "Vbi_shift_V": rsl["Vbi"] - ref["Vbi"],
        "ni_enhancement_max": rsl["ni_enh"],
        "mu_n_at_Nd_ref": ref["mu_n_bulk"], "mu_n_at_Nd_real": rsl["mu_n_bulk"],
        "mu_p_at_Na_ref": ref["mu_p_bulk"], "mu_p_at_Na_real": rsl["mu_p_bulk"],
        "Ln_ref_um": ref["transport"]["Ln"] * 1e4,
        "Ln_real_um": rsl["transport"]["Ln"] * 1e4,
        "Lp_ref_um": ref["transport"]["Lp"] * 1e4,
        "Lp_real_um": rsl["transport"]["Lp"] * 1e4,
        "Js_ref_A_cm2": ref["Js"], "Js_real_A_cm2": rsl["Js"],
        "Js_ratio": rsl["Js"] / ref["Js"],
        "Emax_ref_V_cm": ref["Emax"], "Emax_real_V_cm": rsl["Emax"],
        "Emax_rel_change": rsl["Emax"] / ref["Emax"] - 1.0,
        "W_ref_um": ref["W"] * 1e4, "W_real_um": rsl["W"] * 1e4,
        "W_rel_change": rsl["W"] / ref["W"] - 1.0,
        "equilibrium_local_ni_invariant_ref": ref["invariant"],
        "equilibrium_local_ni_invariant_real": rsl["invariant"],
        "max_newton_iters_ref": max(r.iterations for r in ref["reps"]),
        "max_newton_iters_real": max(r.iterations for r in rsl["reps"]),
        "ideality_mid_ref": float(np.nanmedian(nid_ref[mid])),
        "ideality_mid_real": float(np.nanmedian(nid_rsl[mid])),
        "shockley_ratio_ref_at_0p60": float(ref["J"][-1]
                                            / shockley(ref["sol"].dev, 0.60)),
        "shockley_ratio_real_at_0p60": float(rsl["J"][-1]
                                             / shockley(rsl["sol"].dev, 0.60)),
    }
    for v in PROBE_BIAS:
        k = int(np.argmin(abs(V - v)))
        metrics[f"J_ref_{v:.2f}"] = float(ref["J"][k])
        metrics[f"J_real_{v:.2f}"] = float(rsl["J"][k])
        metrics[f"J_ratio_{v:.2f}"] = float(ratio[k])

    dev_r = rsl["sol"].dev
    st = rsl["sol"].physical_state(rsl["U"][0])
    fig, axes = viz.figure(1, 3, figsize=(12.4, 3.8))
    doping = dev_r.doping_tot
    axes[0].plot(st["um"], dev_r.mat.mu_n(doping), label="$\\mu_n(N)$")
    axes[0].plot(st["um"], dev_r.mat.mu_p(doping), label="$\\mu_p(N)$")
    axes[0].axhline(dev_r.mat.mu_n0, color="#888", ls=":", lw=0.8)
    axes[0].axhline(dev_r.mat.mu_p0, color="#888", ls=":", lw=0.8)
    axes[0].set_ylabel("$\\mu$ [cm$^2$/(V s)]")
    axes[1].plot(st["um"], dev_r.mat.ni(doping), label="$n_i(N)$")
    axes[1].axhline(dev_r.mat.ni0(), color="#888", ls=":", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("$n_i$ [cm$^{-3}$]")
    axes[2].plot(st["um"], st["Ec"] - st["Ei"], label="$E_c-E_i$")
    axes[2].plot(st["um"], st["Ei"] - st["Ev"], label="$E_i-E_v$")
    axes[2].axhline(dev_r.mat.Eg / 2.0, color="#888", ls=":", lw=0.8)
    axes[2].set_ylabel("band edge to intrinsic level [eV]")
    for ax in axes:
        ax.set_xlabel("x [$\\mu$m]")
        ax.axvline(dev_r.x_j * 1e4, color="k", lw=0.6, alpha=0.4)
        ax.set_xlim(58, 62)
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); axes[2].legend(fontsize=7)
    fig.suptitle("Material profiles across the junction (Arora + Slotboom BGN)", y=1.02)
    metrics["fig_models"] = viz.save(fig, "case6_material_models.png").name

    fig, axes = viz.figure(1, 2, figsize=(10.6, 4.2))
    a = axes[0]
    a.semilogy(V[1:], np.abs(ref["J"][1:]), "o-", ms=3, label="reference (constant $\\mu$, no BGN)")
    a.semilogy(V[1:], np.abs(rsl["J"][1:]), "s--", ms=3, label="realistic (Arora + BGN)")
    a.set_xlabel("V [V]"); a.set_ylabel("$|J|$ [A/cm$^2$]")
    a.set_title("Model sensitivity of the forward characteristic")
    a.legend(fontsize=8)
    c = axes[1]
    c.plot(V[1:], ratio[1:], "^-", ms=3, color="#8c564b")
    c.set_xlabel("V [V]"); c.set_ylabel("$|J_{real}|/|J_{ref}|$")
    c.set_yscale("log")
    c.set_title("Current ratio"); c.axhline(1.0, color="#888", ls="--", lw=0.8)
    metrics["fig_iv"] = viz.save(fig, "case6_realistic_iv.png").name

    viz.write_csv(zip(V, ref["J"], rsl["J"], ratio, nid_ref, nid_rsl),
                  "case6_realistic_device.csv",
                  ["V_V", "J_reference_A_cm-2", "J_realistic_A_cm-2",
                   "ratio", "ideality_reference", "ideality_realistic"])

    out = Path(__file__).resolve().parents[1] / "data" / "metrics_case6.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:42s} {v}")
