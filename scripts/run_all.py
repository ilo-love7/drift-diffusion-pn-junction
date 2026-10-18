"""Run every validation case in order and aggregate the acceptance metrics.

Each ``scripts/caseN_*.py`` module exposes ``run() -> dict`` and writes
``data/metrics_caseN.json``.  This driver calls them in sequence, collects the
key numbers into ``data/summary.json`` and prints one table, so a reviewer can
reproduce the whole study with a single command.

    python scripts/run_all.py            # all eight cases
    python scripts/run_all.py --fast     # skip the slow ones (mesh, Sesame, transients)
    python scripts/run_all.py --only 2 5 # selected cases
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

#: case number -> (module name, headline metrics shown in the summary table)
CASES = {
    1: ("case1_equilibrium",
        ["residual", "current_abs_A_cm2", "max_np_over_ni2_deviation",
         "Vbi_rel_error", "W_rel_error"]),
    2: ("case2_forward_iv",
        ["max_newton_iterations", "max_weighted_residual",
         "current_continuity_max_rel_dev", "psi_drop_vs_Vbi_minus_V_max_dev",
         "lnJ_vs_Vj_effective_Vt_rel_error"]),
    3: ("case3_reverse_cv",
        ["max_weighted_residual", "W_rel_dev_max",
         "mott_schottky_doping_rel_error", "srh_peak_rate_over_limit_at_-0p70"]),
    4: ("case4_mesh_convergence",
        ["W_1601_vs_3201_rel_dev", "J0p60_1601_vs_3201_rel_dev",
         "order_J_0p60_last", "order_W_last"]),
    5: ("case5_sesame_crosscheck",
        ["max_rel_dev", "median_rel_dev", "agree_to_3_digits",
         "speed_factor"]),
    6: ("case6_realistic_device",
        ["equilibrium_local_ni_invariant_real", "Vbi_shift_V", "Js_ratio"]),
    7: ("case7_sesame_profiles",
        ["psi_max_abs_diff_V", "density_rel_dev_eq_reverse_max",
         "density_rel_dev_forward_bulk_max", "field_rel_dev_bulk_max",
         "all_gates_passed"]),
    8: ("case8_slab_time_constants",
        ["dielectric_rate_rel_error_max", "dielectric_order_dev_theta1.0",
         "dielectric_order_dev_theta0.5", "plasma_rate_rel_error_max",
         "uniformity_with_displacement_max", "terminal_current_over_terms_max",
         "uniformity_ratio_displacement_gain",
         "chains_with_zero_newton_iterations", "all_gates_passed"]),
}
SLOW = {4, 5, 7, 8}


def _flatten(m: dict) -> dict:
    """Case 4 stores its headline numbers one level down."""
    if "metrics" in m and "rows" in m:
        out = dict(m["metrics"])
        out["rows"] = m["rows"]
        return out
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true",
                    help=f"skip the slow cases {sorted(SLOW)}")
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="case numbers to run")
    args = ap.parse_args(argv)

    selected = args.only if args.only is not None else sorted(CASES)
    if args.fast:
        selected = [c for c in selected if c not in SLOW]

    summary, failures = {}, []
    for case in selected:
        module_name, keys = CASES[case]
        module = importlib.import_module(module_name)
        t0 = time.perf_counter()
        try:
            metrics = _flatten(module.run())
        except Exception as exc:                      # noqa: BLE001
            failures.append((case, repr(exc)))
            print(f"case {case} ({module_name}): FAILED {exc}")
            continue
        elapsed = time.perf_counter() - t0
        metrics["wall_seconds"] = elapsed
        summary[case] = {"module": module_name, "metrics": metrics}
        print(f"\n=== case {case} - {module_name}  ({elapsed:.1f} s)")
        for k in keys:
            print(f"    {k:42s} {metrics.get(k, 'n/a')}")

    (ROOT / "data" / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n=== summary ===")
    print(f"{'case':<6}{'module':<28}{'seconds':>9}   headline")
    for case, entry in summary.items():
        m = entry["metrics"]
        headline = ", ".join(f"{k}={m[k]}" for k in CASES[case][1][:2] if k in m)
        print(f"{case:<6}{entry['module']:<28}{m['wall_seconds']:>9.1f}   {headline}")
    print(f"\nwrote {ROOT / 'data' / 'summary.json'}")
    if failures:
        print(f"{len(failures)} case(s) failed: {[c for c, _ in failures]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
