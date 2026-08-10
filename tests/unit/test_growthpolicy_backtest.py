"""A4 rolling-origin backtest tests (4)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.growthpolicy.backtest_rolling import (  # noqa: E402
    RollingOriginBacktest)
from reference_net.growthpolicy.extrapolate_domhan import (  # noqa: E402
    DomhanExtrapolator)

B, E = RollingOriginBacktest(), DomhanExtrapolator()
T = np.arange(200, dtype=float)


def test_clean_pow3_passes():
    y = 0.3 + 0.9 * (T + 1) ** -0.7
    r = B.skill(E, y)
    assert r["passed"] and r["error"] < 0.1


def test_white_noise_fails():
    y = np.random.default_rng(0).normal(0.5, 0.3, 200)
    assert not B.skill(E, y)["passed"]


def test_stub_extrapolator_composability():
    class Stub:
        def fit(self, s):
            return {"family": "stub", "params": {}, "last": s[-1]}

        def predict_at(self, fr, t):
            return fr["last"]
    y = np.linspace(1.0, 0.2, 200)          # stub predicts origin's last
    r = B.skill(Stub(), y)
    assert "error" in r                      # composes fine


def test_short_series_refusal():
    assert "refusal" in B.skill(E, np.ones(10))
