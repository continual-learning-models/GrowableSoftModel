"""A3 BOCPD tests (5)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.growthpolicy.changepoint_bocpd import BOCPD  # noqa: E402

C = BOCPD()


def test_level_shift_flagged_with_location():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0.0, 0.05, 170),
                        rng.normal(1.0, 0.05, 30)])
    r = C.detect(y)
    assert r["p_recent_change"] > 0.5
    assert abs(r["location"] - 170) <= 8
    assert not r["passed"]


def test_stationary_low_p():
    y = np.random.default_rng(1).normal(0.0, 0.05, 200)
    r = C.detect(y)
    assert r["p_recent_change"] < 0.2 and r["passed"]


def test_smooth_decay_low_p():
    t = np.arange(200.)
    y = 0.3 + 0.7 * np.exp(-0.02 * t) \
        + np.random.default_rng(2).normal(0, 0.01, 200)
    assert C.detect(y)["p_recent_change"] < 0.35


def test_short_series_refusal():
    assert "refusal" in C.detect(np.ones(10))


def test_determinism():
    y = np.random.default_rng(3).normal(size=128)
    assert C.detect(y) == C.detect(y)
