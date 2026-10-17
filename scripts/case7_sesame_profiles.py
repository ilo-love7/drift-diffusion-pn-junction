"""Case 7 - cross-check of the *internal* profiles against NIST Sesame.

Case 5 compares one scalar (the terminal current).  This case compares the
solved state itself: electron density, hole density, electrostatic potential
and electric field at every mesh node, at four bias points, against the same
NIST Sesame 2.0 system.  It needs no change to Sesame and no re-implementation
of its bias continuation: ``IVcurve`` already pickles the converged state of
every sweep point (``sesame/solvers.py:399-406``) and ``load_sim`` reads it back.

Two premises are measured rather than assumed:

* **Gauge.** Sesame's ``v`` is referenced to the intrinsic level
  (``-affinity - Eg/2``), so the two potentials differ by a constant offset.
  The offset and its spread across biases are reported; every potential
  comparison is made on the left-contact-referenced difference.
* **Contacts.** Sesame pins densities through ``contact_S`` surface velocities,
  this solver through Dirichlet rows.  Deviations are therefore split into a
  bulk band (>= 5 um from either contact) and the contact-adjacent regions, so
  the table says *where* the difference lives instead of hiding it in one maximum.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference" / "sesame"))

from p5dd import DDSolver, reference_device, sesame_system, viz  # noqa: E402

N_NODES = 801
VOLTAGES = np.round(np.concatenate([
    [-0.60, -0.40, -0.20, -0.10, -0.05],
    np.arange(0.0, 0.60, 0.05),
    np.arange(0.60, 0.7001, 0.02),
]), 4)
PROFILE_BIAS = (-0.60, 0.00, 0.30, 0.60)
MESH_CHECK_BIAS = (0.30, 0.60)

DENSITY_FLOOR = 1.0e6        # [cm^-3] below this a relative density error is solver noise
FIELD_FLOOR = 1.0e3          # [V/cm] the field changes sign inside the device
BULK_EDGE = 5.0e-4           # [cm] nodes closer than this to a contact are contact-adjacent
AFFINITY = 4.05              # [eV] value handed to Sesame by devices.sesame_system()

#: thresholds declared here, before the run below reads any number off them
GATES = {
    "psi_gauge_offset_spread_V": 1e-9,
    "psi_gauge_expected_abs_error_V": 1e-9,
    "psi_max_abs_diff_V": 1e-6,
    "density_rel_dev_eq_reverse_max": 1e-6,
    "density_rel_dev_forward_bulk_max": 1e-2,
    "field_rel_dev_bulk_max": 1e-2,
}


def _expected_gauge(dev) -> float:
    """Sesame's potential zero: the intrinsic level of a homogeneous neutral region,
    ``-(affinity + Eg/2 + (Vt/2)*ln(Nc/Nv))`` [V]."""
    m = dev.mat
    return -(AFFINITY + m.Eg / 2.0 + 0.5 * m.Vt * np.log(m.Nc / m.Nv))


def _restore_logging() -> None:
    """Undo the root DEBUG handler Sesame installs at import time."""
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _sesame_profiles(sys_, voltages, tol):
    """Sesame's own sweep, returning per-bias (n, p, v) read back from its saved states."""
    import sesame

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp, "ses").as_posix()
        J = np.asarray(sesame.IVcurve(sys_, list(voltages), prefix,
                                      verbose=False, tol=tol), dtype=float)
        for idx, v in enumerate(voltages):
            s2, res = sesame.load_sim(f"{prefix}_{idx}.gzip")
            az = sesame.Analyzer(s2, res)
            out[round(float(v), 4)] = {
                "n": s2.scaling.density * np.asarray(az.electron_density()).ravel(),
                "p": s2.scaling.density * np.asarray(az.hole_density()).ravel(),
                "v": s2.scaling.energy * np.asarray(res["v"]).ravel(),
                "J": float(J[idx]) * sys_.scaling.current,
            }
    return out


def _compare(dev, sol, ses, bias):
    """Node-by-node comparison of one bias point; returns metrics plus plot series."""
    u, rep = sol.solve_point(bias)
    st = sol.physical_state(u)
    s = ses[round(bias, 4)]
    x, psi_s, psi_n = st["x"], s["v"], st["psi"]
    e_ses = -np.diff(psi_s) / np.diff(x)                 # same edge stencil as st["E"]
    bulk_node = (x > BULK_EDGE) & (x < x[-1] - BULK_EDGE)
    bulk_edge = bulk_node[1:] & bulk_node[:-1]           # an edge is bulk if both nodes are

    m = {"bias_V": float(bias),
         "newton_iterations": int(rep.iterations),
         "gauge_offset_V": float(psi_s[0] - psi_n[0]),
         "psi_max_abs_diff_V": float(np.max(np.abs((psi_s - psi_s[0]) - (psi_n - psi_n[0]))))}
    for name, a, b in (("n", s["n"], st["n"]), ("p", s["p"], st["p"])):
        strong = np.minimum(a, b) > DENSITY_FLOOR
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-30)
        m[f"{name}_rel_dev_max"] = float(np.max(rel[strong]))
        m[f"{name}_rel_dev_bulk_max"] = float(np.max(rel[strong & bulk_node]))
        m[f"{name}_rel_dev_rms"] = float(np.sqrt(np.mean(rel[strong] ** 2)))
        m[f"{name}_argmax_x_um"] = float(x[strong][np.argmax(rel[strong])] * 1e4)
    rel_e = np.abs(e_ses - st["E"]) / np.maximum(np.abs(st["E"]), FIELD_FLOOR)
    m["E_rel_dev_max"] = float(np.max(rel_e))
    m["E_rel_dev_bulk_max"] = float(np.max(rel_e[bulk_edge]))
    m["E_max_abs_diff_V_cm"] = float(np.max(np.abs(e_ses - st["E"])))

    series = {"x": x, "xm": st["xm"], "st": st, "ses": s, "e_ses": e_ses,
              "bulk_node": bulk_node, "bulk_edge": bulk_edge}
    return m, series


def _headline(per_bias):
    eq_rev = [c for c in per_bias if c["bias_V"] <= 0.0]
    fwd = [c for c in per_bias if c["bias_V"] > 0.0]
    pick = lambda cs, key: max(c[key] for c in cs)        # noqa: E731
    return {
        "psi_max_abs_diff_V": pick(per_bias, "psi_max_abs_diff_V"),
        "psi_gauge_offset_spread_V": float(np.ptp([c["gauge_offset_V"] for c in per_bias])),
        "density_rel_dev_eq_reverse_max": float(max(
            pick(eq_rev, "n_rel_dev_max"), pick(eq_rev, "p_rel_dev_max"))),
        "density_rel_dev_forward_bulk_max": float(max(
            pick(fwd, "n_rel_dev_bulk_max"), pick(fwd, "p_rel_dev_bulk_max"))),
        "density_rel_dev_forward_all_max": float(max(
            pick(fwd, "n_rel_dev_max"), pick(fwd, "p_rel_dev_max"))),
        "field_rel_dev_bulk_max": pick(per_bias, "E_rel_dev_bulk_max"),
    }


def run() -> dict:
    _restore_logging()

    dev = reference_device(n_nodes=N_NODES)
    sol = DDSolver(dev)
    t0 = time.perf_counter()
    ses = _sesame_profiles(sesame_system(dev), VOLTAGES, tol=1e-9)
    t_sesame = time.perf_counter() - t0

    t0 = time.perf_counter()
    per_bias, series_all, rows = [], [], []
    for bias in PROFILE_BIAS:
        m, sr = _compare(dev, sol, ses, bias)
        per_bias.append(m)
        series_all.append(sr)
        st, s = sr["st"], sr["ses"]
        for i, xi in enumerate(dev.x):
            rows.append((xi * 1e4, bias, st["n"][i], s["n"][i], st["p"][i], s["p"][i],
                         st["psi"][i], s["v"][i],
                         st["E"][i] if i < dev.N - 1 else "",
                         sr["e_ses"][i] if i < dev.N - 1 else ""))
    t_self = time.perf_counter() - t0

    # mesh check: is the forward-bias density difference a resolution artefact?
    dev2 = reference_device(n_nodes=1601)
    sol2 = DDSolver(dev2)
    ses2 = _sesame_profiles(sesame_system(dev2), VOLTAGES, tol=1e-9)
    mesh_check = {}
    for bias in MESH_CHECK_BIAS:
        m2, _ = _compare(dev2, sol2, ses2, bias)
        m1 = next(c for c in per_bias if c["bias_V"] == bias)
        mesh_check[f"V_{bias:+.2f}"] = {
            "n_rel_dev_max_801": m1["n_rel_dev_max"], "n_rel_dev_max_1601": m2["n_rel_dev_max"],
            "p_rel_dev_max_801": m1["p_rel_dev_max"], "p_rel_dev_max_1601": m2["p_rel_dev_max"],
            "psi_max_abs_diff_V_1601": m2["psi_max_abs_diff_V"],
            "E_rel_dev_bulk_max_1601": m2["E_rel_dev_bulk_max"],
        }

    head = _headline(per_bias)
    gauge_mean = float(np.mean([c["gauge_offset_V"] for c in per_bias]))
    head["psi_gauge_expected_abs_error_V"] = abs(gauge_mean - _expected_gauge(dev))
    gates_passed = {k: bool(head[k] < GATES[k]) for k in GATES}
    metrics = {
        "case": 7, "name": "sesame_profile_crosscheck", "n_nodes": dev.N,
        "sweep_points": int(len(VOLTAGES)), "sesame_tol": 1e-9,
        "profile_biases": [float(b) for b in PROFILE_BIAS],
        "density_floor_cm-3": DENSITY_FLOOR, "field_floor_V_cm": FIELD_FLOOR,
        "bulk_edge_um": BULK_EDGE * 1e4,
        "psi_gauge_offset_V": gauge_mean,
        "psi_gauge_expected_V": _expected_gauge(dev),
        "gates": GATES, "gates_passed": gates_passed,
        "all_gates_passed": all(gates_passed.values()),
        "density_rel_dev_forward_all_max_note": "reported for transparency; not gated",
        "mesh_check_1601_nodes": mesh_check,
        "per_bias": per_bias,
        "seconds_sesame": t_sesame, "seconds_self_built_profiles": t_self,
    }
    metrics.update(head)

    fig, axes = viz.figure(2, 2, figsize=(10.6, 7.0))
    fwd = next(i for i, c in enumerate(per_bias) if c["bias_V"] == 0.60)
    sr, st, s = series_all[fwd], series_all[fwd]["st"], series_all[fwd]["ses"]
    a = axes[0]
    a.semilogy(st["um"], st["n"], label="n, this solver")
    a.semilogy(st["um"], s["n"], "--", label="n, Sesame")
    a.semilogy(st["um"], st["p"], label="p, this solver")
    a.semilogy(st["um"], s["p"], "--", label="p, Sesame")
    a.axvline(dev.x_j * 1e4, color="#888", lw=0.8, ls=":", label="metallurgical junction")
    a.set(xlabel="x [µm]", ylabel="density [cm$^{-3}$]", title="Internal profiles at V = +0.60 V")
    a.legend(fontsize=7)

    c = axes[1]
    for m, sx in zip(per_bias, series_all):
        rel = np.abs(sx["ses"]["n"] - sx["st"]["n"]) / sx["st"]["n"]
        c.semilogy(sx["x"] * 1e4, np.maximum(rel, 1e-12), lw=1.0, label=f"V = {m['bias_V']:+.2f}")
    c.set(xlabel="x [µm]", ylabel="$|\\Delta n|/n$", title="Relative electron-density deviation")
    c.set_ylim(1e-10, 1e-1)
    c.legend(fontsize=7)

    b = axes[2]
    for m, sx in zip(per_bias, series_all):
        d = (sx["ses"]["v"] - sx["st"]["psi"])
        b.plot(sx["x"] * 1e4, d - d[0], lw=1.0, label=f"V = {m['bias_V']:+.2f}")
    b.axvline(dev.x_j * 1e4, color="#888", lw=0.8, ls=":")
    b.set(xlabel="x [µm]", ylabel="$\\Delta\\psi$, left-referenced [V]",
          title="Gauge-fixed potential difference")
    b.legend(fontsize=7)

    d = axes[3]
    for m, sx in zip(per_bias, series_all):
        d.semilogy(sx["xm"] * 1e4, np.maximum(np.abs(sx["e_ses"] - sx["st"]["E"]), 1e-8),
                   lw=1.0, label=f"V = {m['bias_V']:+.2f}")
    d.set(xlabel="x [µm]", ylabel="$|\\Delta E|$ [V/cm]",
          title="Field difference, identical edge stencil")
    d.legend(fontsize=7)
    metrics["fig"] = viz.save(fig, "case7_sesame_profiles.png").name

    viz.write_csv(rows, "case7_sesame_profiles.csv",
                  ["x_um", "V_V", "n_self_cm-3", "n_sesame_cm-3",
                   "p_self_cm-3", "p_sesame_cm-3", "psi_self_V", "psi_sesame_V",
                   "E_self_V_cm", "E_sesame_V_cm"])

    (ROOT / "data" / "metrics_case7.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    m = run()
    for k, v in m.items():
        if k not in ("per_bias", "mesh_check_1601_nodes", "gates", "gates_passed"):
            print(f"{k:46s} {v}")
    print("\nper bias:")
    for c in m["per_bias"]:
        print("  " + ", ".join(f"{k}={v}" for k, v in c.items()))
    print("\ngates:", m["gates_passed"])
    print("mesh check:", json.dumps(m["mesh_check_1601_nodes"], indent=2, default=float))
