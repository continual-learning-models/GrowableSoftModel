"""Rolling-origin backtest (Tashman 2000): the installed
extrapolator must demonstrate skill on the scope's own recent
held-out segment before its extrapolations are trusted — a
held-out gate for the forecaster itself."""
import numpy as np

from .interfaces import BacktestEvaluator, register

_MIN_LEN = 64


class RollingOriginBacktest(BacktestEvaluator):
    NAME = "rolling_origin"

    def skill(self, extrapolator, series, threshold=0.25,
              min_len=_MIN_LEN):
        y = np.asarray(series, dtype=float).ravel()
        if len(y) < min_len:
            return {"refusal": f"series too short ({len(y)} < {min_len})"}
        T = len(y)
        origins = [int(f * T) for f in (0.60, 0.70, 0.80, 0.90)]
        span = max(abs(y[0] - y[-1]), 1e-12)
        errs = []
        for o in origins:
            fit = extrapolator.fit(y[:o])
            if "refusal" in fit:
                errs.append(1.0)                # failed origin
                continue
            pred = extrapolator.predict_at(fit, T - 1)
            errs.append(abs(pred - y[-1]) / span)
        e = float(np.mean(errs))
        return {"error": e, "passed": bool(e <= threshold),
                "per_origin": errs}


register("backtest", RollingOriginBacktest.NAME, RollingOriginBacktest)
