"""Zero-attach probe pricer (DEV_PLAN A5).

Both probes price a candidate part in the SAME units — the scope's
window energy (standardized MSE) — so their curves' extrapolated
asymptotes are directly comparable. Probes never touch the living
scope (sha256 fingerprint asserted).

WIDEN probe: one omega-style unit (out-weight zero at attach) with
the remainder of the scope frozen. Because the attach is additive
and the remainder frozen, the scope-copy's window error equals
e - v * gelu(w x + b) with e the frozen scope's standardized window
error — computed directly, which IS the frozen-copy computation.
Part-only additive training against the scope's error is residual
fitting (Jones 1992 semantics).

DEEPEN probe: a deep copy of the scope with one zero-initialized
composition block; ONLY the block parameters train (plain seeded
SGD); the copy's standardized window MSE is the curve.
"""
import copy
import hashlib

import numpy as np

from .interfaces import ProbePricer, register
from ..net import gelu, gelu_d


def _np_bytes(scope, arr):
    bk = getattr(scope, "_bk", None)
    a = bk.to_numpy(arr) if bk is not None else arr
    return np.ascontiguousarray(a).tobytes()


def _port_bytes(scope, h):
    """Fold the fullwidth port coupling into the hash (Growth
    Interface Reform): every body AND its assembly A_g."""
    port = getattr(scope, "_port_site", None)
    if port is None:
        return
    for s in port.bodies:
        h.update(_np_bytes(scope, s["A"]))
        h.update(_fingerprint(s["body"]).encode())


def _fingerprint(scope):
    h = hashlib.sha256()
    if not hasattr(scope, "W1"):
        # non-Network inner body anywhere in a priced subtree
        # (DESIGN_GROW_BODY_TYPE 4b, T0 finding): hash its type
        # name + parameter dict — same bytes-level determinism,
        # no Network attribute assumptions.
        h.update(type(scope).__name__.encode())
        for k in sorted(scope.P):
            h.update(_np_bytes(scope, scope.P[k]))
        for j in sorted(scope.inner):
            h.update(_fingerprint(scope.inner[j]).encode())
        _port_bytes(scope, h)
        return h.hexdigest()
    for arr in (scope.W1, scope.b1, scope.W2, scope.c):
        h.update(_np_bytes(scope, arr))
    for blk in scope.blocks:
        for key in ("Bin", "bb", "Bout"):
            h.update(_np_bytes(scope, blk[key]))
    for j in sorted(scope.inner):
        h.update(_fingerprint(scope.inner[j]).encode())
    _port_bytes(scope, h)
    return h.hexdigest()


class ZeroAttachPricer(ProbePricer):
    NAME = "zero_attach_v1"

    def price(self, scope, policy):
        rows = list(scope.window_ring)
        min_rows = policy.get("min_window_rows", 64)
        if len(rows) < min_rows:
            return {"refusal":
                    f"window ring too small ({len(rows)} < {min_rows})"}
        if scope._y_mu is None:
            return {"refusal": "scope not yet trained"}
        X = np.stack([r[0] for r in rows])
        Y = np.stack([r[1] for r in rows]).reshape(-1, 1)
        steps = int(policy.get("probe_steps", 300))
        lr = float(policy.get("probe_lr", 0.05))
        fp = _fingerprint(scope)
        widen = self._widen_probe(scope, X, Y, steps, lr)
        deepen = self._deepen_probe(scope, X, Y, steps, lr)
        assert _fingerprint(scope) == fp, "probe touched the living scope"
        return {"widen_curve": widen, "deepen_curve": deepen,
                "steps": steps}

    @staticmethod
    def _widen_probe(scope, X, Y, steps, lr):
        Xs = scope._std_x(X)
        ys = (Y - scope._y_mu) / scope._y_sd
        base = (scope.predict(X) - scope._y_mu) / scope._y_sd
        e = (ys - base).ravel()
        n, d = Xs.shape
        rng = np.random.default_rng(scope._seed_counter + 101)
        w = rng.normal(0, np.sqrt(2.0 / d), d)
        b, v = 0.0, 0.0
        curve = []
        for _ in range(steps):
            z = Xs @ w + b
            g = gelu(z)
            r = e - v * g
            mse = float(np.mean(r ** 2))
            if not np.isfinite(mse):        # divergence guard
                break
            curve.append(mse)
            dv = -np.mean(r * g)
            dz = -(r * v) * gelu_d(z)
            dw = Xs.T @ dz / n
            db = float(np.mean(dz))
            v -= lr * dv
            w -= lr * dw
            b -= lr * db
        return curve

    @staticmethod
    def _deepen_probe(scope, X, Y, steps, lr):
        dc = copy.deepcopy(scope)
        dc.deepen()
        Xs = dc._std_x(X)
        ys = (Y - dc._y_mu) / dc._y_sd
        curve = []
        for _ in range(steps):
            with np.errstate(all="ignore"):   # divergence handled below
                grads, aux = dc._grads(Xs, ys)
            if not np.isfinite(aux["mse"]):   # divergence guard
                break
            curve.append(aux["mse"])
            i = 4
            for blk in dc.blocks:
                blk["Bin"] -= lr * grads[i]
                blk["bb"] -= lr * grads[i + 1]
                blk["Bout"] -= lr * grads[i + 2]
                i += 3
        return curve


register("pricer", ZeroAttachPricer.NAME, ZeroAttachPricer)
