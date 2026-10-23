"""Schema and honest-verdict tests for the reusable verify API (``p5dd.verify``).

Three things are checked, and nothing else:

* the report **schema** - three ordered layers, and every layer carrying its
  entry point, its checks and the limit of what it cannot show;
* each layer's **pass/fail** - the committed products pass, and a breached
  criterion fails;
* **honest failure** - a failed or skipped layer is never reported as a pass, and
  a tampered report cannot widen a gate.

These tests run off the committed ``data/metrics_caseN.json`` products, so they
are offline and take milliseconds.  The layer gates themselves are exercised
through :func:`p5dd.verify.evaluate_layer`; the wiring to the real case scripts is
``scripts/case1_equilibrium.py::run`` etc. and is covered by CI running
``scripts/run_all.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p5dd.verify import (CANNOT_SHOW, LAYERS, VERDICT_FAIL, VERDICT_UNPROVEN,
                         VerificationReport, evaluate_layer, verify_all)

ROOT = Path(__file__).resolve().parents[1]


def _product(name: str) -> dict:
    """Read one committed case product (case 4 nests its headline numbers)."""
    data = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
    return data["metrics"] if ("metrics" in data and "rows" in data) else data


@pytest.fixture(scope="module")
def case1() -> dict:
    return _product("metrics_case1.json")


@pytest.fixture(scope="module")
def case4() -> dict:
    return _product("metrics_case4.json")


@pytest.fixture(scope="module")
def case5() -> dict:
    return _product("metrics_case5.json")


@pytest.fixture(scope="module")
def case7() -> dict:
    return _product("metrics_case7.json")


@pytest.fixture(scope="module")
def report() -> VerificationReport:
    return verify_all(source="products")


def test_report_schema_and_layer_identity(report):
    """① Schema: three ordered layers, and every promised field present."""
    assert [layer.layer for layer in report.layers] == list(LAYERS)

    payload = report.to_dict()
    assert set(payload) == {"status", "gate", "layers"}

    for layer in report.layers:
        assert layer.status in ("pass", "fail", "not_executed")
        assert layer.title and layer.entry_point.startswith("scripts/")
        assert layer.cannot_show == CANNOT_SHOW[layer.layer]
        assert layer.checks, f"{layer.layer} reported no checks"
        for check in layer.checks:
            assert check.name and check.criterion and check.note
            assert isinstance(check.passed, bool)
            assert isinstance(check.gated, bool)
            assert check.measured is None or isinstance(check.measured, float)

    json.dumps(payload)  # the report must be serialisable as-is

    text = report.format_table()
    assert "layer" in text and "deliverable:" in text
    for layer in report.layers:
        assert layer.cannot_show[:24] in text


def test_all_three_layers_pass_on_the_committed_products(report):
    """② Pass: this repository meets the gates its own README declares."""
    for layer in report.layers:
        assert layer.status == "pass", f"{layer.layer}: {layer.failed}"
        assert layer.passed is True
    assert report.status == "pass"
    assert report.layer("L2").failed == []          # the un-gated J@0.30 order is not a criterion


def test_each_layer_can_fail_on_a_breached_criterion(case1, case4, case5, case7):
    """② Fail: every layer's own gate set can actually fail (no always-pass API)."""
    l1 = evaluate_layer("L1", {**case1, "current_abs_A_cm2": 1e-3})
    assert l1.status == "fail"
    assert any(c.name == "current_abs_A_cm2" for c in l1.failed)

    l2_metrics = {k: v for k, v in case4.items() if k != "observed_orders"}
    l2_metrics["order_J_0p60_last"] = 1.0
    l2 = evaluate_layer("L2", l2_metrics)
    assert l2.status == "fail"
    assert any(c.name == "order_J_0p60_last" for c in l2.failed)

    l3 = evaluate_layer("L3", {"case5": case5, "case7": {**case7, "psi_max_abs_diff_V": 1e-3}})
    assert l3.status == "fail"
    assert any(c.name == "psi_max_abs_diff_V" for c in l3.failed)

    # a missing metric is not a pass either
    assert evaluate_layer("L1", {"current_abs_A_cm2": 0.0}).status == "fail"


def test_declared_thresholds_beat_a_tampered_report(case7):
    """③ Honest failure: submitting gates_passed=True cannot hide a breach."""
    tampered = {**case7,
                "psi_max_abs_diff_V": 1.0,
                "gates_passed": {k: True for k in case7["gates_passed"]},
                "all_gates_passed": True}
    layer = evaluate_layer("L3", {"case5": _product("metrics_case5.json"), "case7": tampered})
    assert layer.status == "fail"
    assert layer.failed and layer.failed[0].name == "psi_max_abs_diff_V"


def test_not_executed_is_never_a_pass_and_l3_cannot_compensate(case1, case4, case5, case7):
    """③ Honest failure: a skip is "unproven", and a mandatory failure is not rescued."""
    unproven = VerificationReport((
        evaluate_layer("L1", case1),
        evaluate_layer("L2", case4),
        evaluate_layer("L3", {"case5": case5}),        # case 7 product absent -> not executed
    ))
    assert unproven.layer("L3").status == "not_executed"
    assert unproven.layer("L3").passed is False        # a skip is never counted as a pass
    assert unproven.status == VERDICT_UNPROVEN

    broken = VerificationReport((
        evaluate_layer("L1", {**case1, "Vbi_rel_error": 5e-1}),
        evaluate_layer("L2", case4),
        evaluate_layer("L3", {"case5": case5, "case7": case7}),
    ))
    assert broken.layer("L3").status == "pass"
    assert broken.status == VERDICT_FAIL               # L3 cannot compensate for L1