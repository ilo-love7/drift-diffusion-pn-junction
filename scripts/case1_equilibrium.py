"""Case 1 - thermal equilibrium of the abrupt silicon p-n junction (V = 0).

Checks the four equilibrium invariants that any correct drift-diffusion solver
must reproduce without being told to:

  * zero terminal current,
  * flat quasi-Fermi levels,
  * n*p = ni^2 everywhere (including inside the space-charge region),
  * built-in potential, depletion width and peak field of the analytic
    abrupt-junction (depletion approximation) solution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p5dd import DDSolver, built_in_potential, depletion, reference_device, viz  # noqa: E402
from p5dd.constants import Q_E  # noqa: E402


def run() -> dict:
    dev = reference_device()
    sol = DDSolver(dev)
    u, rep = sol.solve_point(0.0)
    st = sol.physical_state(u)
    F = sol.residual(u, 0.0)
    abs_res = float(np.max(np.abs(F)))

    psi, phin, phip, n, p = sol._fields(u, dev)
    Vt = dev.mat.Vt

    # ---- equilibrium invariants -------------------------------------------
    J = sol.total_current(u)
    Ef_flat = max(float(np.ptp(phin)), float(np.ptp(phip))) * Vt
    np_prod = n * p
    np_err = float(np.max(np.abs(np_prod / dev.r_ni ** 2 - 1.0)))

    # ---- analytic comparison ----------------------------------------------
    ref = depletion(dev, 0.0)
    vbi_num = float(st["psi"][0] - st["psi"][-1])
    vbi_ana = built_in_potential(dev)

    emag = np.abs(st["E"])
    inside = emag > 1e-3 * emag.max()
    links = st["xm_um"]
    w_extent = float(links[inside][-1] - links[inside][0])
    e_max_num = float(emag.max())

    # Charge-based depletion edges: Gauss' law gives the ionized charge on each
    # side of the junction as eps*Emax = q*N*x, which is mesh-robust, unlike a
    # field threshold (the field tail beyond the abrupt edge is exponential).
    x_n_num = dev.mat.eps * e_max_num / (Q_E * dev.Nd) * 1e4
    x_p_num = dev.mat.eps * e_max_num / (Q_E * dev.Na) * 1e4
    w_num = x_n_num + x_p_num

    # net charge density on sites, used to locate the space-charge edges
    rho = Q_E * (st["p"] - st["n"] + dev.nd * dev.Nref - dev.na * dev.Nref)

    metrics = {
        "case": 1, "name": "equilibrium", "n_nodes": dev.N,
        "residual": rep.residual, "residual_absolute": abs_res,
        "newton_iterations": rep.iterations,
        "current_abs_A_cm2": abs(J),
        "quasi_fermi_flatness_V": Ef_flat,
        "max_np_over_ni2_deviation": np_err,
        "Vbi_numeric_V": vbi_num, "Vbi_analytic_V": vbi_ana,
        "Vbi_rel_error": abs(vbi_num - vbi_ana) / vbi_ana,
        "W_numeric_um": w_num, "W_analytic_um": ref["W"] * 1e4,
        "W_rel_error": abs(w_num - ref["W"] * 1e4) / (ref["W"] * 1e4),
        "W_field_extent_um": w_extent,
        "x_n_numeric_um": x_n_num, "x_p_numeric_um": x_p_num,
        "Emax_numeric_V_cm": e_max_num, "Emax_analytic_V_cm": ref["Emax"],
        "Emax_rel_error": abs(e_max_num - ref["Emax"]) / ref["Emax"],
        "x_n_analytic_um": ref["x_n"] * 1e4, "x_p_analytic_um": ref["x_p"] * 1e4,
    }

    # ---- figures -----------------------------------------------------------
    x = st["um"]
    fig, ax = viz.figure(figsize=(7.2, 5.0))
    ax.plot(x, st["Ec"], label="$E_c$")
    ax.plot(x, st["Ev"], label="$E_v$")
    ax.plot(x, st["Ei"], "--", label="$E_i$")
    ax.axhline(st["Ef"][0], color="#888", lw=1, label="$E_F$ (flat)")
    ax.axvline(dev.x_j * 1e4, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("x [$\\mu$m]"); ax.set_ylabel("Energy [eV]")
    ax.set_title(f"Band diagram at equilibrium ($V_{{bi}}$ = {vbi_num:.3f} V)")
    ax.legend(ncol=2, fontsize=8)
    metrics["fig_band"] = viz.save(fig, "case1_band_diagram.png").name

    fig, axes = viz.figure(1, 3, figsize=(12.6, 4.0))
    a = axes[0]
    a.semilogy(x, st["n"], label="n")
    a.semilogy(x, st["p"], label="p")
    a.semilogy(x, np.where(dev.nd > 0, dev.nd * dev.Nref, np.nan), ":",
               alpha=0.7, label="doping")
    a.semilogy(x, np.where(dev.na > 0, dev.na * dev.Nref, np.nan), ":", alpha=0.7)
    a.axvspan(dev.x_j * 1e4 - ref["x_n"] * 1e4, dev.x_j * 1e4 + ref["x_p"] * 1e4,
              color="#ff7f0e", alpha=0.15, label="depletion (analytic)")
    a.set_xlim(58.0, 62.0); a.set_xlabel("x [$\\mu$m]"); a.set_ylabel("[cm$^{-3}$]")
    a.set_title("Carrier densities"); a.legend(fontsize=8)

    b = axes[1]
    b.plot(st["xm_um"], st["E"], label="E")
    b.set_xlabel("x [$\\mu$m]"); b.set_ylabel("E [V/cm]")
    b.set_xlim(59.6, 60.6)
    b.axvline(dev.x_j * 1e4, color="k", lw=0.6, alpha=0.4)
    bax = b.twinx()
    bax.plot(x, np.abs(rho) / max(abs(rho.max()), abs(rho.min())), "--",
             color="#2ca02c", label="|net charge|")
    bax.set_ylim(0, 1.05); bax.set_ylabel(r"|ρ| / |ρ|$_{\max}$", color="#2ca02c")
    bax.tick_params(axis="y", colors="#2ca02c")
    b.set_title("Field and space charge in the SCR")
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = bax.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    c = axes[2]
    c.plot(x, st["psi"], label=r"$\varphi$")
    c.set_xlabel("x [$\\mu$m]"); c.set_ylabel(r"$\varphi$ [V]")
    c.set_title("Electrostatic potential (full device)")
    c.legend(fontsize=8)
    metrics["fig_profiles"] = viz.save(fig, "case1_profiles.png").name

    viz.write_csv(
        zip(x, st["psi"], st["Ec"], st["Ev"], st["Ei"], st["n"], st["p"],
            np.append(st["E"], np.nan), rho),
        "case1_equilibrium.csv",
        ["x_um", "psi_V", "Ec_eV", "Ev_eV", "Ei_eV", "n_cm-3", "p_cm-3",
         "E_V_cm", "rho_C_cm-3"])

    out = Path(__file__).resolve().parents[1] / "data" / "metrics_case1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    m = run()
    for k, v in m.items():
        print(f"{k:28s} {v}")
