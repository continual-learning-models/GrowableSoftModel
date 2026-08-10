"""Curve-family extrapolator (Domhan, Springenberg & Hutter, IJCAI
2015 lineage): three parametric families fitted to a decreasing
energy series, BIC-weighted; asymptote CI by seeded residual
bootstrap. Families are linear in (a, b) given the nonlinear
parameter(s), so fitting = grid over the nonlinear parameter +
exact linear least squares — deterministic, no iterative optimizer.

  pow3 : y(t) = a + b * (t+1)^(-c)
  exp3 : y(t) = a + b * exp(-c t)
  dlog3: y(t) = a + b / (1 + exp(c (t - t0)))   [delayed payoff]
"""
import numpy as np

from .interfaces import Extrapolator, register

_MIN_LEN = 64


def _basis(family, t, c, t0=None):
    if family == "pow3":
        return (t + 1.0) ** (-c)
    if family == "exp3":
        return np.exp(-c * t)
    return 1.0 / (1.0 + np.exp(np.clip(c * (t - t0), -60, 60)))


def _fit_family(family, y, t):
    grids = np.geomspace(1e-3, 3.0, 20)
    t0s = (np.linspace(0.0, 2.0 * len(y), 20) if family == "dlog3"
           else [None])
    best = None
    for c in grids:
        for t0 in t0s:
            g = _basis(family, t, c, t0)
            A = np.column_stack([np.ones_like(g), g])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            rss = float(np.sum((A @ coef - y) ** 2))
            if best is None or rss < best["rss"]:
                best = {"family": family, "a": float(coef[0]),
                        "b": float(coef[1]), "c": float(c),
                        "t0": (None if t0 is None else float(t0)),
                        "rss": rss}
    return best


class DomhanExtrapolator(Extrapolator):
    NAME = "domhan2015"

    def fit(self, series, seed=0, margin_frac=0.02,
            min_len=_MIN_LEN):
        y = np.asarray(series, dtype=float).ravel()
        if len(y) < min_len:
            return {"refusal": f"series too short ({len(y)} < {min_len})"}
        if not np.all(np.isfinite(y)):
            return {"refusal": "series contains non-finite values"}
        t = np.arange(len(y), dtype=float)
        fits = [_fit_family(f, y, t) for f in ("pow3", "exp3", "dlog3")]
        # identifiability guard: if the best curve's trend component is
        # indistinguishable from the residual noise, (a, b) are
        # collinear-unidentifiable and the extrapolated intercept is
        # meaningless — the honest answer for a flat series is "the
        # asymptote is the current level".
        best0 = min(fits, key=lambda f: f["rss"])
        noise = float(np.sqrt(max(best0["rss"], 1e-300) / len(y)))
        g0 = _basis(best0["family"], t, best0["c"], best0["t0"])
        signal = abs(best0["b"]) * float(np.ptp(g0))
        # SNR floor of 10: dlog3 can overfit a single noise wiggle with
        # a steep step (signal ~ 3x noise on flat series); a trend must
        # exceed the residual noise by an order of magnitude before its
        # extrapolation is meaningful.
        if signal < 10.0 * noise:
            tail = float(np.mean(y[-max(len(y) // 4, 1):]))
            return {"asymptote": tail, "ci_low": tail - 2 * noise,
                    "ci_high": tail + 2 * noise, "p_useful": 0.0,
                    "rel_ci_width": float(4 * noise
                                          / max(abs(y[0] - tail),
                                                4 * noise, 1e-12)),
                    "family": "flat", "params": {"a": tail, "b": 0.0,
                                                 "c": 0.0, "t0": None}}
        n = len(y)
        bics = np.array([n * np.log(max(f["rss"], 1e-300) / n)
                         + (3 if f["family"] != "dlog3" else 4)
                         * np.log(n) for f in fits])
        w = np.exp(-(bics - bics.min()) / 2.0)
        w = w / w.sum()
        asym = float(sum(wi * f["a"] for wi, f in zip(w, fits)))
        best = fits[int(np.argmax(w))]
        resid = y - self._curve(best, t)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(200):
            yb = self._curve(best, t) + rng.choice(resid, size=n,
                                                   replace=True)
            boots.append(_fit_family(best["family"], yb, t)["a"])
        boots = np.array(boots)
        ci_low, ci_high = np.percentile(boots, [10, 90])
        spread = max(abs(y[0] - ci_low), 1e-12)
        margin = margin_frac * abs(y[0]) + 1e-12
        return {"asymptote": asym, "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_useful": float(np.mean(boots < y[-1] - margin)),
                "rel_ci_width": float((ci_high - ci_low) / spread),
                "family": best["family"],
                "params": {k: best[k] for k in ("a", "b", "c", "t0")}}

    @staticmethod
    def _curve(fit, t):
        return fit["a"] + fit["b"] * _basis(fit["family"], t,
                                            fit["c"], fit["t0"])

    def predict_at(self, fit_result, t):
        if fit_result.get("family") == "flat":
            return float(fit_result["asymptote"])
        pr = fit_result["params"]
        fake = {"family": fit_result["family"], **pr}
        return float(self._curve(fake, np.asarray([float(t)]))[0])


register("extrapolator", DomhanExtrapolator.NAME, DomhanExtrapolator)
