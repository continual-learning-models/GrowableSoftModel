"""transformer_plus — the transformer substrate with the omega
operator (Stage 3b).

v0 design decision (recorded in TOTAL_PLASTICITY_DESIGN §2.2): FFN
widening is UNIFORM across layers, keeping the frozen host's shared-m
invariant — the smallest change that removes the outward cap.
Per-layer widths are v1 (deferred). Attention/d_model are not widened
(foundation-inadequate cases route to Phi). Inner reference-net Networks
widen at every scale via net_ops.widen_at.

Pure-method subclass (no new state): pickle round-trips as
TransformerPlus; inherits the full transformer contract unchanged."""
from __future__ import annotations

import numpy as np

from core.substrates.transformer import TransformerSubstrate
from core.plasticity.net_ops import widen_at


class TransformerPlus(TransformerSubstrate):
    NAME = "transformer_plus"

    def widen_ffn(self, k: int = 1) -> dict:
        """omega at scale 0: append k FFN units to EVERY layer (uniform
        v0). Exact function preservation: W2 rows enter at ZERO."""
        if k < 1:
            raise ValueError("k must be >= 1")
        d, m0 = self.d, self.m
        self._seed_counter += 1
        rng = np.random.default_rng(self._seed_counter)
        for l in range(self.L):
            w1, b1 = f"W1_{l}", f"b1_{l}"
            w2 = f"W2_{l}"
            self.P[w1] = np.hstack(
                [self.P[w1], rng.normal(0, np.sqrt(2.0 / d), (d, k))])
            self.P[b1] = np.concatenate([self.P[b1], np.zeros(k)])
            self.P[w2] = np.vstack([self.P[w2], np.zeros((k, d))])
            for key, grow in ((w1, lambda a: np.hstack(
                    [a, np.zeros((d, k))])),
                    (b1, lambda a: np.concatenate([a, np.zeros(k)])),
                    (w2, lambda a: np.vstack([a, np.zeros((k, d))]))):
                mv = self._adam[key]
                self._adam[key] = [grow(mv[0]), grow(mv[1])]
            # EMA warm-start at column-mean (units are columns here)
            for ema in (self._ema_dw, self._ema_adw):
                col = ema[l].mean(axis=1, keepdims=True)
                ema[l] = np.hstack([ema[l], np.repeat(col, k, 1)])
        self.m = m0 + k
        return {"widened": k, "m": self.m,
                "new_units": list(range(m0, self.m)),
                "params": self.n_params()}

    def widen_inner(self, site_path: str, k: int = 1) -> dict:
        """omega below scale 0: widen an inner reference-net Network —
        'layer{l}/ffn[{j}]' widens that inner net's root;
        'layer{l}/ffn[{j}]::0/2' widens deeper hops."""
        outer, _, inner_path = site_path.partition("::")
        l, j = self._parse(outer)
        net = self.inner[(l, j)]
        out = widen_at(net, inner_path or "root", k)
        out["site"] = site_path
        return out
