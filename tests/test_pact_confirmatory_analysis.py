from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_pact_confirmatory",
        ROOT / "scripts" / "analyze_pact_confirmatory.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analysis = _load()


def test_wilson_interval_known_half_success():
    low, high = analysis.wilson_interval(40, 80)
    assert low == pytest.approx(0.391, abs=0.002)
    assert high == pytest.approx(0.609, abs=0.002)


def _comparison(difference, low, high):
    return {"difference": difference, "ci_95": [low, high]}


def _fisher(p):
    return {"p_value_two_sided": p}


def test_decision_requires_both_ablation_and_act_comparisons():
    token = analysis.choose_decision(
        reconciled=True,
        pact_vs_act=_comparison(0.3, 0.1, 0.5),
        pact_vs_zero=_comparison(0.2, 0.05, 0.4),
        fisher_pact_act=_fisher(0.001),
        fisher_pact_zero=_fisher(0.01),
    )
    assert token == "PACT_BENEFIT_ESTABLISHED"
    token = analysis.choose_decision(
        reconciled=True,
        pact_vs_act=_comparison(0.3, 0.1, 0.5),
        pact_vs_zero=_comparison(0.02, -0.1, 0.1),
        fisher_pact_act=_fisher(0.001),
        fisher_pact_zero=_fisher(0.8),
    )
    assert token == "PACT_NO_CONFIRMED_BENEFIT"


def test_significantly_worse_is_not_buried():
    token = analysis.choose_decision(
        reconciled=True,
        pact_vs_act=_comparison(-0.25, -0.4, -0.05),
        pact_vs_zero=_comparison(0.0, -0.1, 0.1),
        fisher_pact_act=_fisher(0.01),
        fisher_pact_zero=_fisher(1.0),
    )
    assert token == "PACT_WORSE_THAN_ACT"


def test_report_last_nonblank_line_is_exact_token():
    document = analysis.render_report(
        {
            "results_available": False,
            "reconciliation": {
                "expected": 240,
                "valid": 239,
                "missing": ["x"],
                "driver_noncomplete": [],
                "invalid": [],
            },
        },
        {"decision": "PACT_EXPERIMENT_INCOMPLETE"},
    )
    assert document.rstrip().splitlines()[-1] == "PACT_EXPERIMENT_INCOMPLETE"
