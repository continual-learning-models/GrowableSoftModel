"""A1 Domhan-lineage extrapolator tests (10)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.growthpolicy.extrapolate_domhan import (  # noqa: E402
    DomhanExtrapolator)

E = DomhanExtrapolator()
T = np.arange(200, dtype=float)


def _noisy(clean, seed, s=0.01):
    return clean + np.random.default_rng(seed).normal(0, s, len(clean))


def test_pow3_recovery_within_ci():
    y = _noisy(0.30 + 0.9 * (T + 1) ** -0.7, 1)
    r = E.fit(y, seed=0)
    assert r["ci_low"] - 0.05 <= 0.30 <= r["ci_high"] + 0.05


def test_exp3_recovery_within_ci():
    y = _noisy(0.20 + 0.8 * np.exp(-0.03 * T), 2)
    r = E.fit(y, seed=0)
    assert r["ci_low"] - 0.05 <= 0.20 <= r["ci_high"] + 0.05


def test_dlog3_delayed_payoff_recovery():
    y = _noisy(0.10 + 0.6 / (1 + np.exp(0.08 * (T - 120))), 3)
    r = E.fit(y, seed=0)
    assert abs(r["asymptote"] - 0.10) < 0.12
    assert r["family"] == "dlog3"


def test_bic_ensemble_recovers_exponential_asymptote():
    # dlog3 (one extra parameter) can match an exponential decay
    # almost exactly, so the family LABEL is not the contract — the
    # extrapolated asymptote is
    y = _noisy(0.25 + 0.7 * np.exp(-0.05 * T), 4, s=0.005)
    r = E.fit(y, seed=0)
    assert r["family"] in ("exp3", "dlog3")
    assert abs(r["asymptote"] - 0.25) < 0.05


def test_constant_series_tight():
    r = E.fit(np.full(100, 0.5), seed=0)
    assert abs(r["asymptote"] - 0.5) < 1e-6
    assert r["rel_ci_width"] < 1e-3 or (r["ci_high"] - r["ci_low"]) < 1e-9


def test_short_series_refusal():
    assert "refusal" in E.fit(np.ones(10))


def test_nan_refusal():
    y = np.ones(100); y[3] = np.nan
    assert "refusal" in E.fit(y)


def test_bootstrap_determinism_two_calls():
    y = _noisy(0.3 + 0.9 * (T + 1) ** -0.7, 5)
    assert E.fit(y, seed=7) == E.fit(y, seed=7)


def test_p_useful_semantics():
    falling = _noisy(0.3 + 0.9 * (T + 1) ** -0.4, 6, s=0.002)  # far from asymptote
    flat = np.full(100, 0.5) + np.random.default_rng(7).normal(0, 1e-4, 100)
    assert E.fit(falling, seed=0)["p_useful"] > 0.8
    assert E.fit(flat, seed=0)["p_useful"] < 0.2


def test_predict_at_matches_curve():
    y = 0.2 + 0.8 * np.exp(-0.03 * T)
    r = E.fit(y, seed=0)
    assert abs(E.predict_at(r, 199) - y[-1]) < 0.02
