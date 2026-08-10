"""Bayesian online changepoint detection (Adams & MacKay 2007):
run-length posterior under a Normal-Inverse-Gamma conjugate model,
constant hazard. p_recent_change = posterior mass on run lengths
< RECENT at the final step."""
import numpy as np

from .interfaces import ChangepointDetector, register

_MIN_LEN = 64
_RECENT = 32


class BOCPD(ChangepointDetector):
    NAME = "bocpd"

    def detect(self, series, threshold=0.2, hazard=1.0 / 50.0,
               min_len=_MIN_LEN, recent=_RECENT):
        y = np.asarray(series, dtype=float).ravel()
        if len(y) < min_len:
            return {"refusal": f"series too short ({len(y)} < {min_len})"}
        mu0 = float(y[:16].mean())
        kappa0, alpha0 = 1.0, 1.0
        beta0 = float(max(y[:16].var(), 1e-12))
        # per-run-length sufficient stats
        mu = np.array([mu0]); kappa = np.array([kappa0])
        alpha = np.array([alpha0]); beta = np.array([beta0])
        R = np.array([1.0])
        for i, x in enumerate(y):
            # Student-t predictive per run length
            df = 2.0 * alpha
            scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
            z = (x - mu) / scale
            pred = (np.exp(-0.5 * (df + 1.0) * np.log1p(z * z / df))
                    * np.exp(np.vectorize(_lgam)(0.5 * (df + 1.0))
                             - np.vectorize(_lgam)(0.5 * df))
                    / (scale * np.sqrt(np.pi * df)))
            growth = R * pred * (1.0 - hazard)
            cp = float((R * pred * hazard).sum())
            R = np.concatenate([[cp], growth])
            R = R / max(R.sum(), 1e-300)
            # update stats
            mu_new = (kappa * mu + x) / (kappa + 1.0)
            beta_new = beta + 0.5 * kappa * (x - mu) ** 2 / (kappa + 1.0)
            mu = np.concatenate([[mu0], mu_new])
            kappa = np.concatenate([[kappa0], kappa + 1.0])
            alpha = np.concatenate([[alpha0], alpha + 0.5])
            beta = np.concatenate([[beta0], beta_new])
            if len(R) > 4096:                    # cost bound
                R, mu, kappa, alpha, beta = (a[:4096] for a in
                                             (R, mu, kappa, alpha, beta))
        p_recent = float(R[:recent].sum())
        # location: the MAP run length at the final step says how long
        # ago the current regime began
        location = len(y) - int(np.argmax(R))
        return {"p_recent_change": p_recent,
                "location": location,
                "passed": bool(p_recent <= threshold)}


def _lgam(v):
    from math import lgamma
    return lgamma(v)


register("changepoint", BOCPD.NAME, BOCPD)
