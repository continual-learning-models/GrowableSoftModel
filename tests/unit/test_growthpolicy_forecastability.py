"""A2 spectral-entropy forecastability tests (5)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
from reference_net.growthpolicy.forecastability_spectral import (  # noqa: E402
    SpectralEntropyForecastability)

F = SpectralEntropyForecastability()


def test_damped_sine_trend_scores_high():
    t = np.arange(256.)
    y = 1.0 - 0.003 * t + 0.2 * np.sin(0.2 * t) * np.exp(-t / 200)
    assert F.score(y)["score"] > 0.3            # structured


def test_shuffled_noise_scores_low():
    t = np.arange(256.)
    y = 1.0 - 0.003 * t + 0.2 * np.sin(0.2 * t)
    rng = np.random.default_rng(0)
    z = y.copy(); rng.shuffle(z)
    assert F.score(z)["score"] < F.score(y)["score"]


def test_white_noise_near_zero():
    z = np.random.default_rng(1).normal(size=512)
    assert F.score(z)["score"] < 0.2


def test_zero_variance_scores_one():
    assert F.score(np.full(100, 3.14)) == {"score": 1.0, "passed": True}


def test_short_series_refusal():
    assert "refusal" in F.score(np.ones(10))
