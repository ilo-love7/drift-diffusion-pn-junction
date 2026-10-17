"""Case 4 - mesh independence of the Scharfetter-Gummel discretisation.

The whole node profile is refined by a factor of two at a time (``graded_mesh``
distributes the nodes by equal arc length in the target-spacing metric, so this
refines the junction and the bulk together).  For each mesh the same three
quantities are extracted - equilibrium depletion width, peak field and the
terminal current at two forward biases - and the successive differences give
the observed order of convergence, which must approach the formal second order
of the finite-volume / Scharfetter-Gummel scheme.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p5dd import DDSolver, depletion_charge, peak_field, reference_device, viz  # noqa: E402
from p5dd.constants import Q_E  # noqa: E402
from p5dd.theory import depletion  # noqa: E402

GRIDS = (201, 401, 801, 1601, 3201)
PROBE_BIAS = (0.30, 0.60)


def measure(n_nodes: int) -> dict:
    dev = reference_device(n_nodes=n_nodes)
    # The weighted-residual floor is set by the sparse LU round-off, which
    # worsens on the coarse meshes; 1e-7 keeps the solver error two orders
    # below the discretisation error being measured at every grid.
    sol = DDSolver(dev, tol=1e-7)
    V = np.array([0.0, *PROBE_BIAS])
    t0 = time.perf_counter()
    U, J, reps = sol.iv_sweep(V)
    dt = (time.perf_counter() - t0) / len(V)

    st0 = sol.physical_state(U[0])
    # Gauss: Q = eps*Emax = q*ND*x_n = q*NA*x_p gives the charge-based edges.
    q_scr = depletion_charge(dev, st0)
    x_n = q_scr / (Q_E * dev.Nd)
    x_p = q_scr / (Q_E * dev.Na)
    ni_node = dev.r_ni * dev.Nref
    inv = float(np.max(np.abs(st0["n"] * st0["p"] / ni_node ** 2 - 1.0)))
    return {
        "n_nodes": dev.N,
        # The profile is graded, so quote both edges of the spacing range: the
        # junction step (which controls the space-charge resolution) and the
        # bulk step.  Both halve with every grid, hence the factor-2 ratios used
        # for the observed order below.
        "h_min_um": float(np.min(dev.h) * 1e4),
        "h_max_um": float(np.max(dev.h) * 1e4),
        "W_um": (x_n + x_p) * 1e4, "Emax": peak_field(st0),
        "np_ni2_dev": inv, "seconds_per_bias": dt,
        "newton_iters": max(r.iterations for r in reps),
        "residual": max(r.residual for r in reps),
        **{f"J_{v:.2f}": float(J[i + 1]) for i, v in enumerate(PROBE_BIAS)},
    }


def run() -> dict:
    rows = [measure(n) for n in GRIDS]
    ref = rows[-1]
    for r in rows:
        r["W_rel_dev"] = abs(r["W_um"] - ref["W_um"]) / ref["W_um"]
        r["Emax_rel_dev"] = abs(r["Emax"] - ref["Emax"]) / ref["Emax"]
        for v in PROBE_BIAS:
            key = f"J_{v:.2f}"
            r[key + "_rel_dev"] = abs(r[key] - ref[key]) / abs(ref[key])

    # Observed order from the three finest meshes: p = ln(d_c/d_f)/ln(h_c/h_f)
    orders = {}
    for v in PROBE_BIAS:
        key = f"J_{v:.2f}"
        d = [abs(rows[i][key] - rows[i + 1][key]) for i in range(len(rows) - 1)]
        p = [np.log(d[i] / d[i + 1]) / np.log(rows[i]["h_min_um"] / rows[i + 1]["h_min_um"])
             for i in range(len(d) - 1) if d[i] > 0 and d[i + 1] > 0]
        orders[f"J_{v:.2f}"] = [float(x) for x in p]
    d_w = [abs(rows[i]["W_um"] - rows[i + 1]["W_um"]) for i in range(len(rows) - 1)]
    orders["W"] = [float(np.log(d_w[i] / d_w[i + 1])
                         / np.log(rows[i]["h_min_um"] / rows[i + 1]["h_min_um"]))
                   for i in range(len(d_w) - 1) if d_w[i] > 0 and d_w[i + 1] > 0]

    metrics = {
        "case": 4, "name": "mesh_convergence", "grids": list(GRIDS),
        "W_at_1601_um": rows[-2]["W_um"], "W_at_3201_um": ref["W_um"],
        "W_1601_vs_3201_rel_dev": rows[-2]["W_rel_dev"],
        "J0p60_1601_vs_3201_rel_dev": rows[-2]["J_0.60_rel_dev"],
        "J0p30_1601_vs_3201_rel_dev": rows[-2]["J_0.30_rel_dev"],
        "J0p60_801_vs_3201_rel_dev": rows[-3]["J_0.60_rel_dev"],
        "Emax_1601_vs_3201_rel_dev": rows[-2]["Emax_rel_dev"],
        "equilibrium_np_ni2_max_dev_at_3201": ref["np_ni2_dev"],
        "observed_orders": orders,
        "order_J_0p60_last": orders["J_0.60"][-1] if orders["J_0.60"] else None,
        "order_W_last": orders["W"][-1] if orders["W"] else None,
        "seconds_per_bias_at_3201": ref["seconds_per_bias"],
        "speedup_201_vs_3201": rows[-1]["seconds_per_bias"] / rows[0]["seconds_per_bias"],
        "max_newton_iters_over_grids": max(r["newton_iters"] for r in rows),
        "W_theory_um": float(depletion(reference_device(), 0.0)["W"] * 1e4),
    }

    h = np.array([r["h_min_um"] for r in rows])
    dev_j = np.array([abs(r["J_0.60"] - ref["J_0.60"]) / abs(ref["J_0.60"])
                      for r in rows])
    dev_w = np.array([abs(r["W_um"] - ref["W_um"]) / ref["W_um"] for r in rows])
    fig, axes = viz.figure(1, 2, figsize=(10.6, 4.3))
    a = axes[0]
    a.loglog(h[:-1], dev_j[:-1], "o-", label="$|J_{0.60}-J_{ref}|/J_{ref}$")
    a.loglog(h[:-1], dev_w[:-1], "s--", label="$|W-W_{ref}|/W_{ref}$")
    guide = h[:-1]
    a.loglog(guide, guide ** 2 * (dev_j[0] / guide[0] ** 2), ":", color="#888",
             label="$\\propto h^2$")
    a.set_xlabel("junction mesh spacing $h$ [$\\mu$m]")
    a.set_ylabel("relative deviation from the finest mesh")
    a.set_title("Mesh convergence"); a.legend(fontsize=8)
    c = axes[1]
    c.plot([r["n_nodes"] for r in rows], [r["newton_iters"] for r in rows],
           "o-", label="Newton iterations (3 bias points)")
    c.plot([r["n_nodes"] for r in rows], [r["seconds_per_bias"] for r in rows],
           "s--", color="#2ca02c", label="s / bias point")
    c.set_yscale("log"); c.set_xlabel("nodes"); c.set_ylabel("iterations / seconds")
    c.set_title("Cost does not degrade the iteration count"); c.legend(fontsize=8)
    metrics["fig_conv"] = viz.save(fig, "case4_mesh_convergence.png").name

    viz.write_csv(zip(*[[r[k] for k in ("n_nodes", "h_min_um", "h_max_um", "W_um",
                                         "Emax", "J_0.30", "J_0.60", "np_ni2_dev",
                                         "newton_iters", "seconds_per_bias")]
                        for r in rows]),
                  "case4_mesh_convergence.csv",
                  ["n_nodes", "h_min_um", "h_max_um", "W_um", "Emax_V_cm-1",
                   "J_0p30_A_cm-2", "J_0p60_A_cm-2", "np_ni2_max_dev",
                   "newton_iters", "s_per_bias"])

    out = Path(__file__).resolve().parents[1] / "data" / "metrics_case4.json"
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2),
                   encoding="utf-8")
    return metrics


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:42s} {v}")
