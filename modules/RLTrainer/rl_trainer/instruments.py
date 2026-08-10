"""Width-fair health instruments (plan 84 D-P3; doc 88 B-2
boundary record as DESIGN INPUT): the prestudy's registered
erank/width ratio was width-confounded — it trivially favors
narrow nets. The width-fair form reports the ABSOLUTE
effective dimension (entropy-based effective rank of the
hidden-activation matrix, never divided by width) plus the
dead-unit fraction; width is echoed separately so any caller
can still form ratios EXPLICITLY if it wants them.

Read-only instrumentation: organ_health reads the organ's
hidden activations through the same private-instrument
pattern the house already uses (gates.width_demand reads
_ema_dw); nothing is written."""
import numpy as np


def effective_dim(H, eps=1e-12):
    """Absolute entropy-based effective rank of activation
    matrix H (rows = samples): p_i = s_i^2 / sum s^2 over
    singular values, eff_dim = exp(-sum p ln p)."""
    s = np.linalg.svd(np.asarray(H, dtype=float),
                      compute_uv=False)
    e2 = s * s
    tot = float(e2.sum())
    if tot <= eps:
        return 0.0
    p = e2 / tot
    p = p[p > eps]
    return float(np.exp(-np.sum(p * np.log(p))))


def dead_unit_fraction(H, tol=0.01):
    """Fraction of hidden units (columns) whose activation
    std over the probe batch is below tol (population std —
    the doc 88 §1 instrument definition)."""
    H = np.asarray(H, dtype=float)
    return float(np.mean(H.std(axis=0) < tol))


def organ_health(organ, probe_X, tol=0.01):
    """Width-fair health snapshot of a live organ on a probe
    batch: {'effective_dim': absolute, 'dead_frac', 'width'}.
    Uses the organ's own standardization + hidden forward
    (read-only)."""
    X = organ._bk.ingest(np.asarray(probe_X, dtype=float))
    Xs = organ._std_x(X) if organ._x_mu is not None else X
    _, Hact = organ._hidden(Xs)
    H = organ._bk.to_numpy(Hact)
    return {"effective_dim": effective_dim(H),
            "dead_frac": dead_unit_fraction(H, tol=tol),
            "width": int(H.shape[1])}
