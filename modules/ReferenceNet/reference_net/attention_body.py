"""AttentionBody — a COMPLETE small transformer (with attention)
usable as a grown inner body (DESIGN_GROW_BODY_TYPE v1.2, T1).

Adapted at body scale from the released, FD-verified transformer
substrate mathematics (core/substrates/transformer.py): the same
feature-as-token encoding, LN -> multi-head attention -> residual
-> LN -> FFN -> residual layers, here followed by a mean-pool and
a ZERO-INITIALIZED readout. Exact entry replicates BOTH halves of
the reference convention: zero readout AND the zero_out y-scaler
pin (y_mu = 0 for life), so a freshly grown body predicts EXACTLY
0 and the host's output is bitwise unchanged at grow time.

Contract (DESIGN §3, frozen by the T0 audit): predict(X)->(n,1);
train_step(X, r, sgd_lr=None) fitting its own scalers on first
call; .inner == {} (no growth inside, v1); .blocks == [] (no
delta, v1); instability() == [] (empty sequence — the signal
walks enumerate it); depth() == 1; structure(path) rows with the
reference keys; n_params(); pickling; removability.
"""
import numpy as np

from .net import gelu, gelu_d


# _ln_fwd/_ln_bwd moved VERBATIM with the K4 kernels (B1);
# re-imported for any direct user.
from engine.primitives import ln_bwd as _ln_bwd, ln_fwd as _ln_fwd  # noqa: E402,F401

class AttentionBody:
    BODY_TYPE = "attention"

    def __init__(self, d_in, lr=1e-2, seed=0, zero_out=True,
                 d_model=8, n_layers=1, n_heads=2, ffn=16,
                 backend=None, out_width: int = 1):
        # out_width (Growth Interface Reform, doc 35 D2/W2(e)):
        # readout width; DEFAULT 1 keeps every pre-reform
        # construction bitwise identical. The att kernels are
        # shape-generic in Wout's column count (audited).
        if not isinstance(out_width, (int, np.integer)) \
                or isinstance(out_width, bool) or out_width < 1:
            raise ValueError(
                f"out_width must be an int >= 1; got {out_width!r}")
        from engine.backends import current_backend
        self._bk = backend or current_backend()
        if d_model % n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        rng = np.random.default_rng(seed)
        self.d_in, self.d, self.L = int(d_in), int(d_model), int(n_layers)
        self.h, self.m, self.lr = int(n_heads), int(ffn), lr
        self.out_width = int(out_width)
        d = self.d
        P = {"Wv": rng.normal(0, 0.5, (d_in, d)),
             "Bf": rng.normal(0, 0.5, (d_in, d)),
             "Wout": np.zeros((d, self.out_width)),  # zero half
             "bout": np.zeros(self.out_width)}
        for l in range(self.L):
            P[f"g1_{l}"], P[f"b1n_{l}"] = np.ones(d), np.zeros(d)
            P[f"g2_{l}"], P[f"b2n_{l}"] = np.ones(d), np.zeros(d)
            for w in ("Wq", "Wk", "Wk2", "Wo"):
                P[f"{w}_{l}"] = rng.normal(0, np.sqrt(1.0 / d), (d, d))
            P[f"W1_{l}"] = rng.normal(0, np.sqrt(2.0 / d), (d, self.m))
            P[f"b1_{l}"] = np.zeros(self.m)
            P[f"W2_{l}"] = rng.normal(0, np.sqrt(1.0 / self.m),
                                      (self.m, d))
            P[f"b2_{l}"] = np.zeros(d)
        if not zero_out:
            # PORT body (doc 35 R4 zero-side doctrine): born LIVE —
            # the coupling's zero side is the assembly A_g, not the
            # readout. Drawn AFTER all other params so zero_out=True
            # callers stay bitwise identical.
            P["Wout"] = rng.normal(0, np.sqrt(1.0 / d),
                                   (d, self.out_width))
        self.P = {k: self._bk.ingest(v) for k, v in P.items()}
        self._adam = {k: [self._bk.zeros_like(v),
                          self._bk.zeros_like(v)]
                      for k, v in self.P.items()}
        self._t = 0
        self._x_mu = self._x_sd = self._y_mu = self._y_sd = None
        self._zero_out = zero_out               # y_mu pin half
        self._step_count = 0
        self._seed_counter = seed
        self.inner = {}                          # v1: no growth inside
        self.blocks = []                         # v1: no delta
        self.H = d                               # reporting analog

    # ---------- contract surface ----------
    def n_params(self):
        return int(sum(self._bk.numel(v) for v in self.P.values()))

    def depth(self):
        return 1

    def instability(self):
        return []                                # EMPTY SEQUENCE

    def structure(self, path=""):
        return [{"path": path or "root", "H": self.H,
                 "composite": [], "blocks": 0,
                 "body_type": self.BODY_TYPE}]

    def __getstate__(self):
        bk = self._bk
        st = self.__dict__.copy()
        st.pop("_bk", None)
        st["P"] = {k: bk.to_numpy(v) for k, v in self.P.items()}
        st["_adam"] = {k: [bk.to_numpy(m), bk.to_numpy(v)]
                       for k, (m, v) in self._adam.items()}
        for k in ("_x_mu", "_x_sd"):
            if st.get(k) is not None and not isinstance(
                    st[k], (int, float)):
                st[k] = bk.to_numpy(st[k])
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_bk" not in self.__dict__:
            from engine.backends import get_default_backend
            self._bk = get_default_backend()

    def _std_x(self, X):
        return self._bk.standardize(X, self._x_mu, self._x_sd)

    # ---------- forward ----------
    def _forward(self, Xs, cache=False, token_mask=None):
        """K4 kernel (body moved verbatim to the backend, B1)."""
        return self._bk.att_forward(self.P, Xs, self.d, self.L,
                                    self.h, self.m, cache=cache,
                                    token_mask=token_mask)

    def train_from_grad(self, X, dU, sgd_lr=None):
        """Growth-port training entry (doc 35 D4, W2(e)): one
        step against the RAW output gradient dU (n, out_width).
        The att backward is ALREADY gradient-seeded
        (_grads_from_draw takes d(objective)/d(raw) directly),
        so no target trick is needed here. Port-owned bodies
        only (identity scalers)."""
        if not getattr(self, "_port_owned", False):
            raise ValueError(
                "train_from_grad serves port-owned bodies "
                "(identity scalers); construct via "
                "growth_port.make_port_body")
        bk = self._bk
        X = bk.ingest(X)
        dU = bk.ingest(dU)
        Xs = self._std_x(X)                # identity (port-owned)
        raw, Pool, caches = self._forward(Xs, cache=True)
        G = self._grads_from_draw(dU, Pool, caches)
        if sgd_lr is not None:
            self._bk.sgd_step_dict(self.P, G, sgd_lr)
        else:
            self._t = self._bk.adam_step_dict(self.P, G,
                                              self._adam,
                                              self._t, self.lr)
        self._step_count += 1
        return 0.0

    def predict(self, X):
        X = self._bk.ingest(X)
        if self._y_mu is None and self._zero_out:
            return self._bk.zeros((len(X),
                                   getattr(self, "out_width", 1)))
        Xs = self._std_x(X)
        raw = self._forward(Xs)
        if self._y_mu is None:
            return raw
        return raw * self._y_sd + self._y_mu

    # ---------- backward ----------
    def _loss_and_grads(self, Xs, ys):
        """MSE on standardized data and hand gradients for every
        parameter (adapted from the substrate's FD-verified
        backward; no causal branch, no hand-offs, mean-pool
        head)."""
        n = len(Xs)
        d, hh = self.d, self.h
        dh = d // hh
        raw, Pool, caches = self._forward(Xs, cache=True)
        err = raw - ys
        loss = float((err ** 2).mean())      # backend-neutral glue
        dlog = 2 * err / n
        return loss, self._grads_from_draw(dlog, Pool, caches)

    def _grads_from_draw(self, dlog, Pool, caches, token_mask=None):
        """K4 backward (moved verbatim to the backend, B1)."""
        return self._bk.att_backward(self.P, dlog, Pool, caches,
                                     self.d, self.L, self.h,
                                     self.m, token_mask=token_mask)

    # ---------- one training step ----------
    def train_step(self, X, y, sgd_lr=None):
        X = self._bk.ingest(X)
        y = self._bk.ingest(y).reshape(-1, 1)
        if self._x_mu is None:                   # fit scalers once
            self._x_mu, self._x_sd = self._bk.fit_x_stats(X)
            ym, ys_ = self._bk.y_stats(y)
            self._y_mu = 0.0 if self._zero_out else ym
            self._y_sd = ys_
        else:
            # self-shaping y refresh (reference convention: rescale
            # the readout, pin y_mu under zero_out, reset Adam)
            mb, sb = self._bk.y_stats(y)
            if sb > 1.5 * self._y_sd or \
                    abs(mb - self._y_mu) > 2 * self._y_sd:
                mu_n = 0.0 if self._zero_out else mb
                sd_n = max(sb, self._y_sd)
                self.P["Wout"], self.P["bout"] = \
                    self._bk.refresh_readout(
                        self.P["Wout"], self.P["bout"],
                        self._y_mu, self._y_sd, mu_n, sd_n)
                self._y_mu, self._y_sd = mu_n, sd_n
                self._adam = {k: [self._bk.zeros_like(v),
                                  self._bk.zeros_like(v)]
                              for k, v in self.P.items()}
                self._t = 0
        Xs = self._std_x(X)
        ys = (y - self._y_mu) / self._y_sd
        loss, G = self._loss_and_grads(Xs, ys)
        if sgd_lr is not None:                   # plain SGD mode
            self._bk.sgd_step_dict(self.P, G, sgd_lr)
        else:
            self._t = self._bk.adam_step_dict(self.P, G,
                                              self._adam,
                                              self._t, self.lr)
        self._step_count += 1
        return loss
