"""Spectral-entropy forecastability (ForeCA, Goerg ICML 2013):
low spectral entropy = structured, predictable series. Score =
1 - H_spectral / H_max in [0, 1]."""
import numpy as np

from .interfaces import Forecastability, register

_MIN_LEN = 64


class SpectralEntropyForecastability(Forecastability):
    NAME = "spectral_entropy"

    def score(self, series, threshold=0.4, min_len=_MIN_LEN):
        y = np.asarray(series, dtype=float).ravel()
        if len(y) < min_len:
            return {"refusal": f"series too short ({len(y)} < {min_len})"}
        # NOTE (spec correction, logged in DEV_PLAN): the original
        # plan said "first-difference, then entropy". For learning
        # curves with steep early decay the DIFFERENCED series is
        # impulse-like, and an impulse has a FLAT spectrum — smooth,
        # perfectly forecastable curves scored ~0.08. The measure is
        # therefore taken on the mean-removed series itself: a smooth
        # trend concentrates at low frequencies (low entropy, high
        # score); shuffled or white series stay flat-spectrum (low
        # score).
        d = y - y.mean()
        if np.allclose(d, 0.0):
            return {"score": 1.0, "passed": True}
        P = np.abs(np.fft.rfft(d)[1:]) ** 2
        if P.sum() <= 0.0:
            return {"score": 1.0, "passed": True}
        p = P / P.sum()
        p = p[p > 0]
        H = float(-(p * np.log(p)).sum() / np.log(len(P)))
        s = 1.0 - H
        return {"score": s, "passed": bool(s >= threshold)}


register("forecastability", SpectralEntropyForecastability.NAME,
         SpectralEntropyForecastability)
