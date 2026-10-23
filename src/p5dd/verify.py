"""Reusable three-layer verification reports (L1 analytic, L2 mesh, L3 third-party).

This module adds no physics and no new metric.  It re-runs the case scripts that
already live in this repository (``scripts/caseN_*.py``, each exposing
``run() -> dict`` and committing ``data/metrics_caseN.json``) or reads the
products those cases committed, and turns their numbers into one structured
verdict per evidence layer, with that layer's own stated limit attached.

The layers are the ones defined in README
"Verification design: three evidence layers, and what each cannot show":

* **L1** analytic invariants - mandatory floor; it certifies the discretisation,
  *not* the physics model you chose.
* **L2** mesh independence - mandatory floor; it does not show that the
  converged limit is the **right** answer.
* **L3** cross-check vs NIST Sesame 2.0 - bonus; independent code is not an
  independent validation chain, and because both codes share the same model set
  this layer has zero power over model completeness.

Gate, verbatim from the same README section: an L1 or L2 failure fails the
deliverable and L3 cannot compensate; an L3 that could not be executed means
"external consistency unproven", not "rejected".

Nothing here asserts that passing layers make a solver correct, and a passing
report does not shrink the scope boundary declared in README
"Scope boundary: not implemented, not verified".

Two ways to use it:

* :func:`verify_all` runs the existing cases (or reads their committed products)
  and returns a :class:`VerificationReport` - this is the "check this
  repository" path.
* :func:`evaluate_layer` applies the same declared gates to *your* metrics
  mapping, so a third-party solver can be measured against the criteria of this
  study without importing its solver - this is the "check your own solver" path.
  Reproduce the metric names and units documented in :data:`L1_GATES`,
  :data:`L2_GATES`, :data:`L3_CASE5_GATES` and :func:`case7_gates`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]

#: The three evidence layers, in the order the README table declares them.
LAYERS = ("L1", "L2", "L3")

LAYER_TITLES = {
    "L1": "L1 - analytic invariants (numeric vs closed form)",
    "L2": "L2 - mesh independence (five grids, observed order)",
    "L3": "L3 - third-party cross-check vs NIST Sesame 2.0",
}

LAYER_ENTRY_POINTS = {
    "L1": "scripts/case1_equilibrium.py::run",
    "L2": "scripts/case4_mesh_convergence.py::run",
    "L3": ("scripts/case5_sesame_crosscheck.py::run + "
           "scripts/case7_sesame_profiles.py::run"),
}

#: What each layer does **not** prove, copied from the README evidence table so
#: a report can never be read as a stronger claim than the study makes.
CANNOT_SHOW = {
    "L1": ("model completeness - it certifies the discretisation, not the physics "
           "model you chose"),
    "L2": "that the converged limit is the *right* answer",
    "L3": ("any independent ground truth - independent code is not an independent "
           "validation chain, and because both codes share the same model set this "
           "layer has zero power over model completeness"),
}


@dataclass(frozen=True)
class Gate:
    """One acceptance criterion, declared before any number is read off it."""

    metric: str
    criterion: str
    kind: str                      # "lt" | "between"
    bound: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    gated: bool = True             # False -> reported, does not change layer status
    note: str = ""

    def evaluate(self, measured: Any) -> bool:
        if isinstance(measured, bool) or not isinstance(measured, (int, float)):
            return False
        value = float(measured)
        if self.kind == "lt":
            return value < self.bound
        if self.kind == "between":
            return self.lower <= value <= self.upper
        raise ValueError(f"unknown gate kind {self.kind!r}")


# --------------------------------------------------------------------------- #
# Declared gates (mirrors of the thresholds the cases and the README already use)
# --------------------------------------------------------------------------- #

#: L1 gates on ``data/metrics_case1.json`` (``scripts/case1_equilibrium.py::run``).
#: The depletion-width and peak-field entries are deliberately loose: case 1
#: compares against the *depletion approximation*, whose own error is ~2.8 %, so a
#: tight gate there would measure the analytic model rather than the solver.
#: Case 3 measures the mesh-robust eps/C route at 1.67 %.
L1_GATES: tuple[Gate, ...] = (
    Gate("current_abs_A_cm2", "< 1e-12 A/cm^2", "lt", bound=1e-12,
         note="a non-zero equilibrium current would break the SG zero-flux property"),
    Gate("quasi_fermi_flatness_V", "< 1e-12 V", "lt", bound=1e-12,
         note="quasi-Fermi levels must be flat at V = 0"),
    Gate("max_np_over_ni2_deviation", "< 1e-12", "lt", bound=1e-12,
         note="n*p = n_i^2 must hold everywhere, inside the space-charge region too"),
    Gate("Vbi_rel_error", "< 1e-9", "lt", bound=1e-9,
         note="numeric built-in potential vs the closed form"),
    Gate("W_rel_error", "< 5e-2", "lt", bound=5e-2,
         note="2.8 % measured: the depletion approximation's own error, not the solver's"),
    Gate("Emax_rel_error", "< 5e-2", "lt", bound=5e-2,
         note="same depletion-approximation offset as W_rel_error"),
    Gate("residual_absolute", "< 1e-7", "lt", bound=1e-7,
         note="weighted-residual floor; the sparse LU sets ~3e-9 at 1601 nodes"),
)

#: L2 gates on ``data/metrics_case4.json`` (``scripts/case4_mesh_convergence.py::run``).
L2_GATES: tuple[Gate, ...] = (
    Gate("order_J_0p60_last", "1.8 <= p <= 2.2", "between", lower=1.8, upper=2.2,
         note="observed order of the terminal current at 0.60 V; formal order is 2"),
    Gate("order_W_last", "1.8 <= p <= 2.2", "between", lower=1.8, upper=2.2,
         note="observed order of the depletion width"),
    Gate("J0p60_1601_vs_3201_rel_dev", "< 1e-3", "lt", bound=1e-3,
         note="1601 vs 3201 nodes at 0.60 V"),
    Gate("W_1601_vs_3201_rel_dev", "< 1e-3", "lt", bound=1e-3,
         note="1601 vs 3201 nodes, depletion width"),
    Gate("order_J_0p30_last", "not gated", "between", lower=0.0, upper=0.0, gated=False,
         note=("measured 4.52: the J@0.30 difference has reached the 7.2e-8 solver noise "
               "floor, so its order is distorted and the README does NOT use it as a "
               "criterion. Reported for transparency only.")),
)

#: L3 gate on the case-5 product.  Case 5 declares no gate of its own; 5e-3 is the
#: 2-digit-agreement threshold the case itself computes as ``agree_to_2_digits``.
L3_CASE5_GATES: tuple[Gate, ...] = (
    Gate("max_rel_dev", "< 5e-3", "lt", bound=5e-3,
         note=("case 5 declares no gate; 5e-3 is the 2-digit agreement threshold it "
               "computes itself (agree_to_2_digits). The README marks the full-sweep "
               "maximum as display-level, not one of the |J| > 1e-3 gated numbers")),
)


def _case7_gates() -> tuple[Gate, ...]:
    """L3 gates for the profile cross-check, taken from case 7's own ``GATES`` dict.

    The thresholds are read from the case module, not from the metrics mapping
    handed to :func:`evaluate_layer`, so a caller cannot widen a gate by editing
    the report it submits.
    """
    module = _load_case("case7_sesame_profiles")
    notes = {
        "psi_gauge_offset_spread_V":
            "the two codes reference potential to different zeros; the offset must be "
            "the same constant at every bias",
        "psi_gauge_expected_abs_error_V":
            "the offset must equal -(affinity + Eg/2 + (Vt/2)*ln(Nc/Nv))",
        "psi_max_abs_diff_V": "potential difference after removing the gauge constant",
        "density_rel_dev_eq_reverse_max":
            "density agreement at equilibrium and reverse bias",
        "density_rel_dev_forward_bulk_max":
            "forward-bias density agreement away from the contacts (the contact premise "
            "differ; the README reports the all-node maximum separately and does not "
            "gate it)",
        "field_rel_dev_bulk_max":
            "field agreement in the bulk; Sesame does not output field, so both codes "
            "difference their own potential",
    }
    return tuple(
        Gate(name, f"< {float(bound):g}", "lt", bound=float(bound),
             note=notes.get(name, ""))
        for name, bound in module.GATES.items()
    )


# --------------------------------------------------------------------------- #
# Report schema
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Check:
    """One measured value against one declared criterion."""

    name: str
    measured: float | None
    criterion: str
    passed: bool
    gated: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "measured": self.measured,
                "criterion": self.criterion, "passed": bool(self.passed),
                "gated": bool(self.gated), "note": self.note}


@dataclass(frozen=True)
class LayerReport:
    """The verdict for one evidence layer, with the limit of that layer attached."""

    layer: str
    title: str
    status: str                     # "pass" | "fail" | "not_executed"
    checks: tuple[Check, ...]
    entry_point: str
    cannot_show: str
    source: str
    reason: str = ""

    @property
    def passed(self) -> bool:
        """True only for an executed, fully passing layer - never for a skip."""
        return self.status == "pass"

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.gated and not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "title": self.title, "status": self.status,
                "entry_point": self.entry_point, "source": self.source,
                "cannot_show": self.cannot_show, "reason": self.reason,
                "checks": [c.to_dict() for c in self.checks]}


def _layer_status(checks: Sequence[Check]) -> str:
    gated = [c for c in checks if c.gated]
    if gated and all(c.passed for c in gated):
        return "pass"
    return "fail"


#: The deliverable verdicts.  "external consistency unproven" is the README's
#: wording for L3 being unavailable; it is *not* a rejection.
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_UNPROVEN = "external consistency unproven"
VERDICT_NOT_EXECUTED = "not executed"


@dataclass(frozen=True)
class VerificationReport:
    """The three layer verdicts plus the deliverable-level gate."""

    layers: tuple[LayerReport, ...]

    def layer(self, name: str) -> LayerReport:
        for report in self.layers:
            if report.layer == name:
                return report
        raise KeyError(name)

    @property
    def status(self) -> str:
        mandatory = [r for r in self.layers if r.layer in ("L1", "L2")]
        if any(r.status == "fail" for r in mandatory):
            return VERDICT_FAIL
        if any(r.status == "not_executed" for r in mandatory):
            return VERDICT_NOT_EXECUTED
        l3 = [r for r in self.layers if r.layer == "L3"]
        if any(r.status == "fail" for r in l3):
            return VERDICT_FAIL
        if any(r.status == "not_executed" for r in l3):
            return VERDICT_UNPROVEN
        return VERDICT_PASS

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "gate": (
            "an L1 or L2 failure fails the deliverable and L3 cannot compensate; "
            "L3 not executed means 'external consistency unproven', not 'rejected'"),
            "layers": [r.to_dict() for r in self.layers]}

    def format_table(self) -> str:
        width = max([len(r.source) for r in self.layers] + [len("source")]) + 2
        lines = [f"{'layer':<6}{'status':<16}{'checks':<10}"
                 f"{'source':<{width}}entry point",
                 "-" * (6 + 16 + 10 + width + 40)]
        for r in self.layers:
            gated = [c for c in r.checks if c.gated]
            ok = sum(1 for c in gated if c.passed)
            counts = f"{ok}/{len(gated)}" if gated else "n/a"
            lines.append(f"{r.layer:<6}{r.status:<16}{counts:<10}"
                         f"{r.source:<{width}}{r.entry_point}")
        lines.append("")
        lines.append(f"deliverable: {self.status}")
        for r in self.layers:
            lines.append(f"  {r.layer} cannot show: {r.cannot_show}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Loading the existing implementation
# --------------------------------------------------------------------------- #

def _load_case(name: str):
    """Import a ``scripts/caseN_*.py`` module exactly as ``scripts/run_all.py`` does."""
    for path in (ROOT / "src", ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module(name)


def _sesame_available() -> bool:
    path = ROOT / "reference" / "sesame"
    if not path.is_dir():
        return False
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.util.find_spec("sesame") is not None


def _flatten(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a case product the way ``scripts/run_all.py::_flatten`` does.

    Case 4 stores its headline numbers one level down.  The observed-order series
    are also hoisted to ``order_<quantity>_last`` so a single flat mapping can be
    either run or read back from a committed JSON file.
    """
    flat = dict(mapping["metrics"]) if ("metrics" in mapping and "rows" in mapping) \
        else dict(mapping)
    orders = flat.get("observed_orders")
    if isinstance(orders, Mapping):
        for key, series in orders.items():
            if series:
                flat[f"order_{str(key).replace('.', 'p')}_last"] = float(series[-1])
    return flat


PRODUCT_FILES = {"L1": ("metrics_case1.json",), "L2": ("metrics_case4.json",),
                 "L3": ("metrics_case5.json", "metrics_case7.json")}


def load_products(layer: str) -> dict[str, Any]:
    """Read the committed ``data/metrics_caseN.json`` products for one layer.

    Raises :class:`FileNotFoundError` when a product is missing, so the caller can
    report the layer as *not executed* instead of quietly treating it as a pass.
    """
    if layer not in LAYERS:
        raise KeyError(layer)
    out: dict[str, Any] = {}
    for filename in PRODUCT_FILES[layer]:
        path = ROOT / "data" / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing")
        out[filename] = json.loads(path.read_text(encoding="utf-8"))
    if layer == "L3":
        return {"case5": _flatten(out["metrics_case5.json"]),
                "case7": _flatten(out["metrics_case7.json"])}
    return _flatten(next(iter(out.values())))


def _not_executed(layer: str, source: str, reason: str) -> LayerReport:
    return LayerReport(layer=layer, title=LAYER_TITLES[layer], status="not_executed",
                       checks=(), entry_point=LAYER_ENTRY_POINTS[layer],
                       cannot_show=CANNOT_SHOW[layer], source=source, reason=reason)


def run_layer(layer: str, *, source: str = "cases") -> LayerReport:
    """Evaluate one layer.

    ``source="cases"`` re-runs the existing case scripts (product paths unchanged);
    ``source="products"`` reads the products they committed (fast, offline).
    A layer that cannot be executed is returned as ``not_executed`` - never as a
    pass, and never as a silent failure.
    """
    if layer not in LAYERS:
        raise KeyError(layer)

    if source == "products":
        try:
            metrics = load_products(layer)
        except FileNotFoundError as exc:
            return _not_executed(layer, f"products:{exc}", str(exc))
        tag = "products:data/" + "+".join(PRODUCT_FILES[layer])
        return evaluate_layer(layer, metrics, source=tag)

    if source != "cases":
        raise ValueError(f"unknown source {source!r}: use 'cases' or 'products'")

    if layer == "L3":
        if not _sesame_available():
            return _not_executed(
                layer, "cases:sesame unavailable",
                "the vendored NIST Sesame 2.0 under reference/ is absent, so case 5 and "
                "case 7 cannot run: external consistency unproven, not rejected")
        metrics = {"case5": _flatten(_load_case("case5_sesame_crosscheck").run()),
                   "case7": _flatten(_load_case("case7_sesame_profiles").run())}
        return evaluate_layer(layer, metrics, source="cases:case5+case7")

    module_name = "case1_equilibrium" if layer == "L1" else "case4_mesh_convergence"
    metrics = _flatten(_load_case(module_name).run())
    return evaluate_layer(layer, metrics, source=f"cases:{module_name}")


def evaluate_layer(layer: str, metrics: Mapping[str, Any], *,
                   source: str = "injected") -> LayerReport:
    """Apply one layer's declared gates to a metrics mapping - no solver needed.

    ``metrics`` uses the case-product schema:

    * L1 - the keys of ``data/metrics_case1.json``;
    * L2 - the keys of ``data/metrics_case4.json`` (observationally nested products
      are flattened, including the observed-order series);
    * L3 - ``{"case5": <metrics_case5 keys>, "case7": <metrics_case7 keys>}``.

    Thresholds always come from this module / case 7's ``GATES``, never from the
    submitted mapping, so editing ``gates_passed`` in a report cannot turn a
    breach into a pass.  A metric that is absent or non-numeric fails its check:
    an unverifiable criterion is not a pass.
    """
    if layer not in LAYERS:
        raise KeyError(layer)

    if layer == "L3":
        missing = [k for k in ("case5", "case7") if not isinstance(metrics.get(k), Mapping)]
        if missing:
            return _not_executed(layer, source,
                                 f"L3 needs both case products; missing {missing}")
        flat5 = _flatten(metrics["case5"])
        flat7 = _flatten(metrics["case7"])
        pairs = [(g, flat5.get(g.metric)) for g in L3_CASE5_GATES]
        pairs += [(g, flat7.get(g.metric)) for g in _case7_gates()]
    else:
        flat = _flatten(metrics)
        gates = L1_GATES if layer == "L1" else L2_GATES
        pairs = [(g, flat.get(g.metric)) for g in gates]

    checks = []
    for gate, measured in pairs:
        value = float(measured) if isinstance(measured, (int, float)) \
            and not isinstance(measured, bool) else None
        passed = gate.evaluate(value)
        note = gate.note
        if value is None:
            note = "metric missing from the supplied report: cannot be confirmed"
        checks.append(Check(name=gate.metric, measured=value, criterion=gate.criterion,
                            passed=passed, gated=gate.gated, note=note))

    return LayerReport(layer=layer, title=LAYER_TITLES[layer], status=_layer_status(checks),
                       checks=tuple(checks), entry_point=LAYER_ENTRY_POINTS[layer],
                       cannot_show=CANNOT_SHOW[layer], source=source)


def verify_all(*, layers: Sequence[str] = LAYERS,
               source: str = "cases") -> VerificationReport:
    """Evaluate the requested layers and return the deliverable-level report."""
    unknown = [name for name in layers if name not in LAYERS]
    if unknown:
        raise KeyError(f"unknown layer(s) {unknown}; known: {list(LAYERS)}")
    return VerificationReport(tuple(run_layer(name, source=source) for name in layers))


def main(argv: Sequence[str] | None = None) -> int:
    """Print the report and exit non-zero when the deliverable-level gate fails.

    ``--source products`` reads ``data/metrics_caseN.json`` (fast, offline);
    ``--source cases`` re-runs the case scripts (the full ≈90 s suite).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("cases", "products"), default="cases",
                        help="re-run the case scripts, or read their committed products")
    parser.add_argument("--layers", nargs="*", default=list(LAYERS),
                        choices=list(LAYERS), help="layers to evaluate")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    report = verify_all(layers=tuple(args.layers), source=args.source)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_table())
    return 0 if report.status in (VERDICT_PASS, VERDICT_UNPROVEN) else 1


if __name__ == "__main__":
    raise SystemExit(main())