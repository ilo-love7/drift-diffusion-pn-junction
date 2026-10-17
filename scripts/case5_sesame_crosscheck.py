"""Case 5 - independent cross-check against NIST Sesame.

Sesame is a publicly maintained 1D drift-diffusion code written independently
of this solver.  Both are handed *exactly* the same input here: the same node
coordinates, the same abrupt doping, the same constant mobilities, lifetimes,
density of states, band gap, temperature and Ohmic contacts.  Sesame's
``contact_S`` with a huge surface velocity reproduces the pinned equilibrium
carrier densities that the reference solver imposes as Dirichlet rows.

Any difference between the two I-V curves is therefore a difference of
numerics only, which makes this the strongest validation in the study.
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


def _restore_logging() -> None:
    """Undo the root DEBUG handler Sesame installs at import time.

    Left alone, matplotlib's font manager writes several thousand ``findfont``
    records into the run log and buries the results.
    """
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def run() -> dict:
    import sesame
    _restore_logging()

    dev = reference_device(n_nodes=N_NODES)
    sol = DDSolver(dev)
    t0 = time.perf_counter()
    _, J_ref, reps = sol.iv_sweep(VOLTAGES)
    t_self = time.perf_counter() - t0

    sys_ = sesame_system(dev)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        J_ses = sesame.IVcurve(sys_, list(VOLTAGES), Path(tmp, "ses").as_posix(),
                               verbose=False, tol=1e-9)
    t_sesame = time.perf_counter() - t0
    J_ses = np.asarray(J_ses, dtype=float) * sys_.scaling.current

    ok = np.isfinite(J_ses) & (np.abs(J_ref) > 1e-12)
    rel = np.abs(J_ses - J_ref) / np.abs(J_ref)
    rel_ok = rel[ok]
    decades = np.abs(np.log10(np.abs(J_ses[ok])) - np.log10(np.abs(J_ref[ok])))

    metrics = {
        "case": 5, "name": "sesame_crosscheck", "n_nodes": dev.N,
        "sweep_points": int(len(VOLTAGES)),
        "points_compared": int(ok.sum()),
        "sesame_tol": 1e-9,
        "max_rel_dev": float(np.max(rel_ok)),
        "median_rel_dev": float(np.median(rel_ok)),
        "max_log10_dev": float(np.max(decades)),
        "agree_to_3_digits": bool(np.all(rel_ok < 5e-4)),
        "agree_to_2_digits": bool(np.all(rel_ok < 5e-3)),
        "J_solver_at_0p60": float(J_ref[np.argmin(abs(VOLTAGES - 0.60))]),
        "J_sesame_at_0p60": float(J_ses[np.argmin(abs(VOLTAGES - 0.60))]),
        "J_solver_at_-0p60": float(J_ref[0]),
        "J_sesame_at_-0p60": float(J_ses[0]),
        "max_newton_iterations": max(r.iterations for r in reps),
        "seconds_self_built": t_self,
        "seconds_sesame": t_sesame,
        "speed_factor": t_sesame / t_self,
    }

    fig, axes = viz.figure(1, 2, figsize=(10.6, 4.3))
    a = axes[0]
    a.semilogy(VOLTAGES, np.abs(J_ref), "o-", ms=4, label="this solver")
    a.semilogy(VOLTAGES, np.abs(J_ses), "--", lw=2, label="NIST Sesame")
    a.set_xlabel("V [V]"); a.set_ylabel("$|J|$ [A/cm$^2$]")
    a.set_title("Identical input, two independent codes")
    a.legend(fontsize=8); a.set_xlim(-0.65, 0.75)
    c = axes[1]
    c.semilogy(VOLTAGES, np.maximum(rel, 1e-16), "s-", ms=4)
    c.axhline(5e-4, color="#888", ls="--", lw=0.8, label="3-digit agreement")
    c.set_xlabel("V [V]"); c.set_ylabel("$|J_{ses}-J_{ref}|/|J_{ref}|$")
    c.set_title("Relative deviation of the terminal current")
    c.set_ylim(1e-10, 1e-1); c.legend(fontsize=8)
    metrics["fig"] = viz.save(fig, "case5_sesame_crosscheck.png").name

    viz.write_csv(zip(VOLTAGES, J_ref, J_ses, rel),
                  "case5_sesame_crosscheck.csv",
                  ["V_V", "J_self_A_cm-2", "J_sesame_A_cm-2", "rel_dev"])

    out = ROOT / "data" / "metrics_case5.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:42s} {v}")
