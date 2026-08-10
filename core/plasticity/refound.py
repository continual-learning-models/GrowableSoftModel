"""Phi — re-founding / metamorphosis (Stage 5).

When the FOUNDATION itself is wrong or outgrown, patching (rho) only
bloats fine scales. Phi births a NEW skeleton sized from ACCUMULATED
experience (never from batch one), trains it FRESH on stored REAL
rows (never on the old model's own answers — error inheritance), and
takes over ONLY through the normal reality gate. The old self stays
serving until beaten, and remains in lineage for rollback: identity
lives in the record, not in the weight matrix.

Triggers use Stage-1 metrics with the E0-frozen thresholds.
"""
from __future__ import annotations

import numpy as np

from core._modules import reference_net, generator  # noqa: F401
from reference_net.net import Network
from generator.trainer import auto_hidden

INV_ALARM = 0.5          # frozen in E0 REPORT
INV_CONSECUTIVE = 3
DISPOSED_ALARM = 2
LP_FLAT = 1e-3


def should_refound(inv_series, disposed=0, lp_tail=None,
                   max_saturation=0.0, inv_alarm=None,
                   inv_consecutive=INV_CONSECUTIVE,
                   disposed_alarm=DISPOSED_ALARM,
                   lp_flat=LP_FLAT) -> dict:
    """Pre-registered trigger. Returns {"refound": bool, "reason"}.
    inv_alarm: overridable for calibration studies; defaults to the
    frozen constant."""
    alarm = INV_ALARM if inv_alarm is None else inv_alarm
    inv = list(inv_series)[-inv_consecutive:]
    if len(inv) == inv_consecutive and all(v >= alarm
                                           for v in inv):
        return {"refound": True,
                "reason": f"amplitude-hierarchy inversion sustained "
                          f"({[round(v, 3) for v in inv]})"}
    if disposed >= disposed_alarm and max_saturation >= 0.5:
        return {"refound": True,
                "reason": f"{disposed} disposed growths under uniform "
                          f"saturation {max_saturation:.2f}"}
    if lp_tail is not None and len(lp_tail) >= 3 \
            and max(abs(v) for v in lp_tail) < lp_flat \
            and max_saturation >= 0.5:
        return {"refound": True,
                "reason": "LP flat under high demand"}
    return {"refound": False, "reason": "healthy"}


def size_from_store(n_features: int, n_rows: int) -> int:
    """Birth rule re-run on ACCUMULATED experience (current schema,
    full store) — never on batch one."""
    return auto_hidden(n_features, 1, n_rows)[0]


def refound(old: Network, X_store: np.ndarray, y_store: np.ndarray,
            steps: int = 4000, seed: int = 7,
            mode: str = "fresh") -> Network:
    """Build and train the candidate skeleton. mode='fresh' (default)
    or 'shrink_perturb' (old weights embedded, shrunk, noised — the
    G2 warm-start remedy). NEVER trains on old's answers."""
    d_in = X_store.shape[1]
    H = size_from_store(d_in, len(X_store))
    cand = Network(d_in=d_in, hidden=H, seed=seed + 1)
    if mode == "shrink_perturb":
        rng = np.random.default_rng(seed + 2)
        h0 = min(old.H, H)
        d0 = min(old.d_in, d_in)
        cand.W1[:h0, :d0] = (0.4 * old.W1[:h0, :d0]
                             + rng.normal(0, 0.01, (h0, d0)))
        cand.W2[:, :h0] = (0.4 * old.W2[:, :h0]
                           + rng.normal(0, 0.01, (1, h0)))
    for _ in range(steps):
        cand.train_step(X_store, y_store)
    return cand


def gated_takeover(old: Network, cand: Network,
                   X_hold: np.ndarray, y_hold: np.ndarray) -> dict:
    """The normal reality gate: promote ONLY if strictly better on the
    untouched holdout. The incumbent is never modified."""
    mse_old = float(np.mean((old.predict(X_hold) - y_hold) ** 2))
    mse_new = float(np.mean((cand.predict(X_hold) - y_hold) ** 2))
    promoted = mse_new < mse_old
    return {"promoted": promoted, "incumbent_mse": mse_old,
            "candidate_mse": mse_new,
            "event": {"event": "metamorphosis", "promoted": promoted,
                      "old": {"H": old.H, "d_in": old.d_in,
                              "params": old.n_params()},
                      "new": {"H": cand.H, "d_in": cand.d_in,
                              "params": cand.n_params()}}}
