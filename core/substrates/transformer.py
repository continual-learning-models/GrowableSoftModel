"""The `transformer` substrate (plan SWP4; priority 1).

A small from-scratch encoder — attention + FFN — trained entirely on the
caller's data (published architecture math is borrowed; weights never
are). The owner's multi-scale growth mechanism runs on this host exactly
as on the mlp substrate:

- growth SITES are the FFN hidden units of each layer;
- an unstable unit grows a ZERO-INIT inner network — the FROZEN Phase-2
  recursive Network class as the inner body (same operator, new host);
- cross-scale learning is Reading-B: the unit's output target hands
  inward; the inner network trains itself (and recurses identically);
- attention is untouched by growth in v0 (single-variable discipline).

Forward (F feature tokens, d model dims, L layers, h heads, m FFN units):
  tokens  T0[n,F,d] = x[n,F,None] * Wv[F,d] + Bf[F,d]
  layer:  T -> LN1 -> MHA -> +T -> LN2 -> FFN(growable) -> +T
  pool    P = mean over tokens -> shared heads (numeric | categorical)

numpy only; hand-written backprop (finite-difference-checked in tests);
Adam with SGD consolidation mode; deterministic under seed.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from core._modules import reference_net  # noqa: F401
import math
from engine.backends import get_default_backend
from reference_net.net import Network, gelu, gelu_d, ETA_TARGET


def _eta_t(organ):
    """Instance-first eta_target read (S9.5 D-N1 pattern;
    param-interface batch docs/system/22 item 5)."""
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
    gp = getattr(organ, "_growth_policy", DEFAULT_GROWTH_POLICY)
    return float(gp.get("eta_target", ETA_TARGET))
from reference_net.trainer import collect_instability

from core.substrates.base import Substrate, CONTRACT_V
from core.substrates.heads import B_NEG
# (heads.softmax / ce_loss_and_grad no longer imported — the
# categorical path runs on the backend kernels since GSM-I2)


def _ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    sd = np.sqrt(var + 1e-5)
    xhat = (x - mu) / sd
    return xhat * g + b, (xhat, sd)


def _ln_bwd(dy, cache, g):
    xhat, sd = cache
    d = xhat.shape[-1]
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / sd
    return dx, dg, db


class TransformerSubstrate(Substrate):
    NAME = "transformer"
    DATA_FORM = "vector"
    SUPPORTED_HEADS = ("point", "dist")   # 60A

    CAUSAL = False          # sequence variant flips this (one shared core)
    WINDOW = 16             # max timesteps (v0: fixed-window inputs)
    INNER_LR_FACTOR = 0.3   # two-timescale rule, host-calibrated by A/B

    def __init__(self, d_in, hidden, mode="numeric", vocab=None,
                 lr=1e-2, seed=7, d_model=32, n_layers=2, n_heads=2,
                 backend=None, window=None, inner_lr_factor=None,
                 new_class_bias=None):
        # window: causal context length (positional-table size);
        # None -> the class default (16). Param-interface batch,
        # docs/system/22 item 2 (sequence host inherits).
        if window is not None:
            if not isinstance(window, (int, np.integer)) \
                    or isinstance(window, bool) or window < 1:
                raise ValueError(
                    f"window must be an integer >= 1; got {window!r}")
            self.WINDOW = int(window)
        if inner_lr_factor is not None:
            # docs/system/22 item 3: two-timescale factor; None ->
            # class default (host-calibrated 0.3)
            if not isinstance(inner_lr_factor, (int, float)) \
                    or isinstance(inner_lr_factor, bool) \
                    or inner_lr_factor <= 0:
                raise ValueError(
                    "inner_lr_factor must be a number > 0; "
                    f"got {inner_lr_factor!r}")
            self.INNER_LR_FACTOR = float(inner_lr_factor)
        if new_class_bias is not None:
            if isinstance(new_class_bias, bool) or not isinstance(
                    new_class_bias, (int, float)) or new_class_bias > 0:
                raise ValueError(
                    "new_class_bias must be a number <= 0; "
                    f"got {new_class_bias!r}")
            self.new_class_bias = float(new_class_bias)
        from engine.backends import current_backend
        self._bk = backend or current_backend()
        # (the B4 v1 boundary refusal stood here; GSM-I2 ports
        # the categorical and causal paths onto the kernel
        # contract and removes it — kit BK14 holds the parity)
        if mode not in ("numeric", "numeric_dist",
                        "categorical"):
            # 60A L3: whitelist assertion — unknown modes refuse
            # loudly; the old ELSE bucket silently mis-built them
            # as categorical (the crash class)
            raise ValueError(
                f"transformer serves mode='numeric', "
                f"'numeric_dist' and 'categorical'; got {mode!r}")
        rng = np.random.default_rng(seed)
        self.d_in, self.m, self.mode = d_in, hidden, mode
        self.d, self.L, self.h = d_model, n_layers, n_heads
        self.lr, self.seed = lr, seed
        self.vocab = list(vocab) if vocab else []
        n_out = (1 if mode == "numeric" else
                 2 if mode == "numeric_dist" else
                 max(1, len(self.vocab)))
        d = d_model
        if self.CAUSAL:
            # sequence tokens: step vector (d_in dims) -> d, + positions
            P = {"Wv": rng.normal(0, np.sqrt(1.0 / max(1, d_in)),
                                  (d_in, d)),
                 "Bf": rng.normal(0, 0.1, (self.WINDOW, d)),
                 "Wh": rng.normal(0, np.sqrt(1.0 / d), (d, n_out)),
                 "bh": np.zeros(n_out)}
        else:
            P = {"Wv": rng.normal(0, 0.5, (d_in, d)),
                 "Bf": rng.normal(0, 0.5, (d_in, d)),
                 "Wh": rng.normal(0, np.sqrt(1.0 / d), (d, n_out)),
                 "bh": np.zeros(n_out)}
        for l in range(n_layers):
            P[f"g1_{l}"], P[f"b1n_{l}"] = np.ones(d), np.zeros(d)
            P[f"g2_{l}"], P[f"b2n_{l}"] = np.ones(d), np.zeros(d)
            for w in ("Wq", "Wk", "Wk2", "Wo"):
                P[f"{w}_{l}"] = rng.normal(0, np.sqrt(1.0 / d), (d, d))
            P[f"W1_{l}"] = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
            P[f"b1_{l}"] = np.zeros(hidden)
            P[f"W2_{l}"] = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, d))
            P[f"b2_{l}"] = np.zeros(d)
        if mode == "numeric_dist":
            # 60A: heteroscedastic head born ZERO (birth honesty,
            # the mlp precedent core/substrate.py:74-83 — the
            # newborn predicts the data mean with standardized
            # sigma 1; first-step loss is the closed-form 0.5)
            P["Wh"] = np.zeros((d, 2))
            P["bh"] = np.zeros(2)
        self.P = {k: self._bk.ingest(v) for k, v in P.items()}
        self._adam = {k: [self._bk.zeros_like(v),
                          self._bk.zeros_like(v)]
                      for k, v in self.P.items()}
        self._t = 0
        self.inner: dict[tuple, Network] = {}      # (layer, unit) -> body
        self._ema_dw = {l: np.zeros((d, hidden)) for l in range(n_layers)}
        self._ema_adw = {l: np.zeros((d, hidden)) + 1e-12
                         for l in range(n_layers)}
        self._x_mu = self._x_sd = None
        self._y_mu = self._y_sd = None
        self._seed_counter = seed

    # ---------------- forward ----------------
    def _forward(self, X, cache=False):
        n = len(X)
        if cache and getattr(self, "_spu_policy", None) is not None:
            self._spu_next_cache = {}
        d, hh = self.d, self.h
        dh = d // hh
        if self.CAUSAL:
            Tlen = X.shape[1]
            T = X @ self.P["Wv"] + self.P["Bf"][None, :Tlen, :]
        else:
            T = X[:, :, None] * self.P["Wv"][None] + self.P["Bf"][None]
        caches = [("tokens", X)]
        for l in range(self.L):
            Tn, c1 = self._bk.ln_fwd(T, self.P[f"g1_{l}"],
                                     self.P[f"b1n_{l}"])
            Q = Tn @ self.P[f"Wq_{l}"]
            K = Tn @ self.P[f"Wk_{l}"]
            V = Tn @ self.P[f"Wk2_{l}"]

            def heads_(M):
                return self._bk.perm4(M.reshape(n, -1, hh, dh),
                                      (0, 2, 1, 3))
            Qh, Kh, Vh = heads_(Q), heads_(K), heads_(V)
            S = Qh @ self._bk.perm4(Kh, (0, 1, 3, 2)) \
                / math.sqrt(dh)
            if self.CAUSAL:
                Tlen = S.shape[-1]
                S = S + self._bk.ingest(
                    np.triu(np.full((Tlen, Tlen), -1e9), k=1))
            A = self._bk.exp(S - self._bk.rowmax_keep(S))
            A = A / self._bk.rowsum_keep(A)
            Oh = A @ Vh
            O = self._bk.perm4(Oh, (0, 2, 1, 3)).reshape(n, -1, d) \
                @ self.P[f"Wo_{l}"]
            T1 = T + O
            Tn2, c2 = self._bk.ln_fwd(T1, self.P[f"g2_{l}"],
                                      self.P[f"b2n_{l}"])
            pre = Tn2 @ self.P[f"W1_{l}"] + self.P[f"b1_{l}"]
            H = self._bk.gelu(pre)
            inner_in = Tn2.reshape(-1, d)
            if cache and getattr(self, "_spu_policy", None) is not None:
                # SPU seam (DEV_PLAN v2.1 V3): transient per-layer
                # input cache for the next step's host walk; written
                # only by training forwards, never by predict.
                self._spu_next_cache[l] = inner_in
            from reference_net.growth_port import \
                legacy_attach_layer
            H = legacy_attach_layer(H, inner_in, self.inner,
                                    l, self._bk, n)
            ports = getattr(self, "_port_sites", None)
            if ports and ports.get(l) is not None \
                    and ports[l].bodies:
                H = ports[l].forward(
                    H.reshape(-1, self.m),
                    inner_in).reshape(n, -1, self.m)
            F_out = H @ self.P[f"W2_{l}"] + self.P[f"b2_{l}"]
            T2 = T1 + F_out
            if cache:
                caches.append((T, Tn, c1, Q, K, V, A, Vh, T1, Tn2, c2,
                               pre, H, inner_in))
            T = T2
        Pool = T[:, -1, :] if self.CAUSAL else T.mean(axis=1)
        logits = Pool @ self.P["Wh"] + self.P["bh"]
        return (logits, Pool, caches) if cache else logits

    # ---------------- serving ----------------
    def _stdx(self, X):
        X = self._bk.ingest(X)
        if self._x_mu is None:
            return X
        return (X - self._x_mu) / self._x_sd     # broadcasts over steps

    def _fit_x_scalers(self, X):
        if self.CAUSAL:
            # round-trip: numpy ddof-0 std exactly as before
            # (torch .std defaults to ddof=1 — parity trap),
            # identity on the judge (GSM-I2)
            Xn = self._bk.to_numpy(X)
            self._x_mu = self._bk.ingest(Xn.mean((0, 1)))
            self._x_sd = self._bk.ingest(Xn.std((0, 1)) + 1e-8)
        else:
            self._x_mu, self._x_sd = self._bk.fit_x_stats(X)

    def predict(self, X):
        if self.mode == "numeric_dist":                     # 60A
            value, _ = self.predict_dist(X)
            return np.asarray(value).reshape(-1, 1)
        assert self.mode == "numeric"
        if self._y_mu is None:
            return self._bk.zeros((len(X), 1))
        z = self._forward(self._stdx(X))
        return z * self._y_sd + self._y_mu

    def predict_dist(self, X):
        """60A: heteroscedastic serve — the mlp conventions
        transcribed (standardized (mu, v) -> value =
        mu*y_sd+y_mu, std = exp(v/2)*y_sd; the v clamp mirrors
        the kernel's NLL_CLAMP)."""
        assert self.mode == "numeric_dist"
        if self._y_mu is None:
            raise ValueError(
                "untrained: predict_dist needs at least one "
                "training step (the target scalers are unfitted)")
        z = np.asarray(self._bk.to_numpy(
            self._forward(self._stdx(X))))
        _cl = getattr(self, "nll_clamp", None)
        if _cl is None:
            _cl = self._bk.NLL_CLAMP
        mu = z[:, 0]
        v = np.clip(z[:, 1], -_cl, _cl)
        value = mu * self._y_sd + self._y_mu
        std = np.exp(v / 2.0) * self._y_sd
        return value, std

    def predict_proba(self, X):
        assert self.mode == "categorical"
        X = np.asarray(X, float)
        if self._x_mu is None:
            c = max(1, len(self.vocab))
            return np.full((len(X), c), 1.0 / c)
        logits = self._forward(self._stdx(X))
        return self._bk.to_numpy(self._bk.softmax(logits))

    def predict_label(self, X):
        p = self.predict_proba(X)
        idx = p.argmax(axis=1)
        return [self.vocab[i] for i in idx], p[np.arange(len(idx)), idx]

    def add_class(self, label):
        assert self.mode == "categorical" and label not in self.vocab
        self.vocab.append(label)
        # 2a surgery: numpy round-trip + ingest (GSM-I2)
        Wh = self._bk.to_numpy(self.P["Wh"])
        bh = self._bk.to_numpy(self.P["bh"])
        self.P["Wh"] = self._bk.ingest(
            np.hstack([Wh, np.zeros((self.d, 1))]))
        self.P["bh"] = self._bk.ingest(
            np.concatenate(
                [bh, [getattr(self, 'new_class_bias', B_NEG)]]))
        self._adam["Wh"] = [self._bk.zeros_like(self.P["Wh"]),
                            self._bk.zeros_like(self.P["Wh"])]
        self._adam["bh"] = [self._bk.zeros_like(self.P["bh"]),
                            self._bk.zeros_like(self.P["bh"])]

    # ---------------- training ----------------
    def train_step(self, X, y, sgd_lr=None):
        X = self._bk.ingest(X)
        # SPU integration seam (DEV_PLAN v2.1 V3): process grown
        # inner bodies BEFORE this step's forward, against the
        # previous training forward's cached layer inputs.
        _pol = getattr(self, "_spu_policy", None)
        if _pol is not None and _pol["spu_enabled"] \
                and self._x_mu is not None:
            from reference_net.spu.spu_host_walk import spu_host_pre_forward
            spu_host_pre_forward(self, X, y, _pol)
        n = len(X)
        if self._x_mu is None:
            self._fit_x_scalers(X)
            if self.mode in ("numeric", "numeric_dist"):
                ya = self._bk.ingest(y).reshape(-1, 1)
                self._y_mu, self._y_sd = self._bk.y_stats(ya)
        Xs = self._stdx(X)
        if self.mode == "numeric":
            ya = self._bk.ingest(y).reshape(-1, 1)
            mb, sb = self._bk.y_stats(ya)
            if sb > 1.5 * self._y_sd or abs(mb - self._y_mu) > 2 * self._y_sd:
                sd_n = max(sb, self._y_sd)
                self.P["Wh"] = self.P["Wh"] * (self._y_sd / sd_n)
                self.P["bh"] = (self.P["bh"] * self._y_sd
                                + self._y_mu - mb) / sd_n
                self._y_mu, self._y_sd = mb, sd_n
                self._adam = {k: [self._bk.zeros_like(v),
                                  self._bk.zeros_like(v)]
                              for k, v in self.P.items()}
                self._t = 0
            ys = (ya - self._y_mu) / self._y_sd
        if self.mode == "numeric_dist":                     # 60A
            ya = self._bk.ingest(y).reshape(-1, 1)
            ys = (ya - self._y_mu) / self._y_sd

        logits, Pool, caches = self._forward(Xs, cache=True)
        if getattr(self, "_spu_policy", None) is not None:
            self._spu_input_cache = getattr(self, "_spu_next_cache",
                                            None)
        if self.mode == "numeric":
            err = logits - ys
            loss = float((err ** 2).mean())
            dlog = 2 * err / n
        elif self.mode == "numeric_dist":
            # 60A: engine NLL kernels reused VERBATIM (X-1); the
            # mlp-shaped trunk arguments are inert zero
            # placeholders (the kernel's gW2/gc/dH outputs do not
            # depend on them — verified in both backends)
            t = ys.reshape(-1)
            _cl = getattr(self, "nll_clamp", None)
            mu, v = self._bk.nll_forward(self.P["Wh"].T,
                                         self.P["bh"], Pool,
                                         nll_clamp=_cl)
            loss = self._bk.nll_loss(t, mu, v)
            _z1 = self._bk.zeros((n, 1))
            _gk, _dHk = self._bk.nll_backward(
                _z1, _z1, self.P["Wh"].T, self.P["bh"], _z1,
                self._bk.zeros_like(Pool), Pool, mu, v, t,
                nll_clamp=_cl)
            dlog = None            # head grads mapped below
        else:
            labels = [self.vocab.index(v) for v in np.asarray(y).ravel()]
            probs = self._bk.softmax(logits)
            onehot_np = np.zeros((n, len(self.vocab)))
            onehot_np[np.arange(n), labels] = 1.0
            onehot = self._bk.ingest(onehot_np)
            loss = self._bk.cat_ce(probs, onehot)
            dlog = (probs - onehot) / n

        G = {k: self._bk.zeros_like(v) for k, v in self.P.items()}
        if self.mode == "numeric_dist":
            # 60A seam identity: kernel gW2 = err.T@Pool/n is
            # G["Wh"].T; gc = err.mean(0) is G["bh"]; kernel
            # dH = err@Wh.T so dPool = dH/n (the /n convention
            # of this backward's dlog)
            G["Wh"] = _gk[2].T
            G["bh"] = _gk[3]
            dPool = _dHk / n
        else:
            G["Wh"] = Pool.T @ dlog
            G["bh"] = dlog.sum(0)
            dPool = dlog @ self.P["Wh"].T
        Fn = Xs.shape[1]
        if self.CAUSAL:
            dT = self._bk.zeros((len(Xs), Fn, self.d))
            dT[:, -1, :] = dPool                 # last-token head
        else:
            dT = self._bk.repeat_tokens(dPool, Fn)

        d, hh = self.d, self.h
        dh = d // hh
        handoffs = []
        port_calls = []
        for l in range(self.L - 1, -1, -1):
            (T, Tn, c1, Q, K, V, A, Vh, T1, Tn2, c2,
             pre, H, inner_in) = caches[1 + l]
            # FFN branch
            dF = dT                                   # into T1 + F_out
            G[f"W2_{l}"] += H.reshape(-1, self.m).T @ dF.reshape(-1, d)
            G[f"b2_{l}"] += dF.sum((0, 1))
            dH = dF @ self.P[f"W2_{l}"].T             # (n,F,m)
            from reference_net.growth_port import \
                legacy_collect_layer
            for net_, r_ in legacy_collect_layer(
                    self.inner, l, H, dH, pre, _eta_t(self),
                    self._bk.gelu):
                handoffs.append((net_, inner_in, r_))
            ports = getattr(self, "_port_sites", None)
            if ports and ports.get(l) is not None \
                    and ports[l].bodies:
                port_calls.append(
                    (ports[l], dH.reshape(-1, self.m),
                     inner_in))
            dpre = dH * self._bk.gelu_d(pre)
            G[f"W1_{l}"] += Tn2.reshape(-1, d).T @ dpre.reshape(-1, self.m)
            G[f"b1_{l}"] += dpre.sum((0, 1))
            dTn2 = dpre @ self.P[f"W1_{l}"].T
            dT1, dg2, db2 = self._bk.ln_bwd(dTn2, c2,
                                            self.P[f"g2_{l}"])
            G[f"g2_{l}"] += dg2
            G[f"b2n_{l}"] += db2
            dT1 = dT1 + dT                            # residual skip
            # attention branch
            dO = dT1
            Oh = (A @ Vh)
            Ocat = self._bk.perm4(Oh, (0, 2, 1, 3)).reshape(
                len(X), -1, d)
            G[f"Wo_{l}"] += Ocat.reshape(-1, d).T @ dO.reshape(-1, d)
            dOcat = dO @ self.P[f"Wo_{l}"].T
            dOh = self._bk.perm4(
                dOcat.reshape(len(X), -1, hh, dh), (0, 2, 1, 3))
            dA = dOh @ self._bk.perm4(Vh, (0, 1, 3, 2))
            dVh = self._bk.perm4(A, (0, 1, 3, 2)) @ dOh
            dS = A * (dA - self._bk.rowsum_keep(dA * A))
            dS = dS / math.sqrt(dh)
            Qh = self._bk.perm4(
                Q.reshape(len(X), -1, hh, dh), (0, 2, 1, 3))
            Kh = self._bk.perm4(
                K.reshape(len(X), -1, hh, dh), (0, 2, 1, 3))
            dQh = dS @ Kh
            dKh = self._bk.perm4(dS, (0, 1, 3, 2)) @ Qh

            def unheads(M):
                return self._bk.perm4(M, (0, 2, 1, 3)).reshape(
                    len(X), -1, d)
            dQ, dK, dV = unheads(dQh), unheads(dKh), unheads(dVh)
            G[f"Wq_{l}"] += Tn.reshape(-1, d).T @ dQ.reshape(-1, d)
            G[f"Wk_{l}"] += Tn.reshape(-1, d).T @ dK.reshape(-1, d)
            G[f"Wk2_{l}"] += Tn.reshape(-1, d).T @ dV.reshape(-1, d)
            dTn = (dQ @ self.P[f"Wq_{l}"].T + dK @ self.P[f"Wk_{l}"].T
                   + dV @ self.P[f"Wk2_{l}"].T)
            dT0, dg1, db1 = self._bk.ln_bwd(dTn, c1,
                                            self.P[f"g1_{l}"])
            G[f"g1_{l}"] += dg1
            G[f"b1n_{l}"] += db1
            dT = dT0 + dT1                            # residual skip
        # token embeddings
        Xarr = caches[0][1]
        if self.CAUSAL:
            G["Wv"] += self._bk.einsum("ntf,ntd->fd", Xarr, dT)
            G["Bf"][:Xarr.shape[1]] += dT.sum(0)
        else:
            G["Wv"] += self._bk.einsum("nf,nfd->fd", Xarr, dT)
            G["Bf"] += dT.sum(0)

        # update (instability stats on W1 columns per layer)
        old_w1 = {l: self._bk.copy(self.P[f"W1_{l}"])
                  for l in range(self.L)}
        self._step(G, sgd_lr)
        for l in range(self.L):
            dw = self._bk.to_numpy(self.P[f"W1_{l}"] - old_w1[l])
            self._ema_dw[l] = 0.95 * self._ema_dw[l] + 0.05 * dw
            self._ema_adw[l] = 0.95 * self._ema_adw[l] + 0.05 * np.abs(dw)
        # cross-scale target handoff (Reading B) — same primitive
        for net, x_in, r in handoffs:
            net.train_step(x_in, r, sgd_lr=sgd_lr)
        for port, dH_f, x_in in port_calls:
            port.backward_step(dH_f, x_in, self.lr,
                               sgd_lr=sgd_lr)
        from reference_net.instrument import monitor_tick
        monitor_tick(self)         # FR-16 (D-6.3): guard-only
        #                            when unarmed (bit contract)
        return loss


    # ---- 59 R-2: per-region lr scales (host side; shared
    # grammar/blend from reference_net.net — ONE point) ----
    @staticmethod
    def _lr_region_of(key):
        base = {"Bf": "embed", "Wv": "embed",
                "Wh": "out", "bh": "out"}
        if key in base:
            return base[key]
        stem, _, tail = str(key).rpartition("_")
        if stem and tail.isdigit():
            return f"layer:{tail}"
        raise RuntimeError(
            f"train_lr_scales: unclassified P key {key!r} — "
            "new keys must be added to the region map "
            "(59 R-2.3, never silently bucketed)")

    def _lr_scales_active(self):
        from reference_net.net import validate_lr_scales
        sc = getattr(self, "_growth_policy",
                     {}).get("train_lr_scales")
        if sc:
            validate_lr_scales(sc)
        return sc

    def _blend_P(self, old, scales):
        from reference_net.net import blend_scaled
        for k in self.P:
            r = self._lr_region_of(k)
            if r in scales:
                self.P[k] = blend_scaled(old[k], self.P[k],
                                         float(scales[r]))

    def _step(self, G, sgd_lr):
        scales = self._lr_scales_active()      # 59 R-2
        old = dict(self.P) if scales else None
        if sgd_lr is not None:
            self._bk.sgd_step_dict(self.P, G, sgd_lr)
            if scales:
                self._blend_P(old, scales)
            return
        self._t = self._bk.adam_step_dict_mul(self.P, G,
                                               self._adam,
                                               self._t, self.lr)
        if scales:
            self._blend_P(old, scales)
        return
        # (moved verbatim into the K7 dict kernels, B4)
        self._t += 1
        for k in self.P:
            m, v = self._adam[k]
            m[:] = 0.9 * m + 0.1 * G[k]
            v[:] = 0.999 * v + 0.001 * G[k] * G[k]
            mh = m / (1 - 0.9 ** self._t)
            vh = v / (1 - 0.999 ** self._t)
            self.P[k] = self.P[k] - self.lr * mh / (np.sqrt(vh) + 1e-8)

    # ---------------- growth (contract) ----------------
    def _unit_instability(self, l):
        num = np.linalg.norm(self._ema_dw[l], axis=0)
        den = np.linalg.norm(self._ema_adw[l], axis=0) + 1e-12
        return 1.0 - num / den

    def growth_sites(self):
        sites = []
        grown = getattr(self, "_port_js", set())  # W3: port grows
        for l in range(self.L):
            inst = self._unit_instability(l)
            for j in range(self.m):
                if (l, j) not in self.inner \
                        and (l, j) not in grown:
                    sites.append((f"layer{l}/ffn[{j}]", float(inst[j])))
        deep_bodies = list(self.inner.items())
        for l, site in getattr(self, "_port_sites", {}).items():
            deep_bodies += [(s.get("key", (l, g)), s["body"])
                            for g, s in enumerate(site.bodies)]
        for (l, j), net in deep_bodies:
            for path, k, score, owner in collect_instability(net):
                if k not in owner.inner and k not in \
                        getattr(owner, "_port_js", set()):
                    sites.append(
                        (f"layer{l}/ffn[{j}]::{path}[{k}]", float(score)))
        return sorted(sites, key=lambda t: -t[1])

    # ---------- B2: whole-layer insertion (doc 55 s3-B2) ----------
    _LAYER_KEY_ROOTS = ("g1", "b1n", "g2", "b2n",
                        "Wq", "Wk", "Wk2", "Wo",
                        "W1", "b1", "W2", "b2")

    def insert_layer(self, position, recipe="random",
                     source_index=None, zero_side="default",
                     force=False):
        import time as _time
        _t0 = _time.perf_counter()
        from reference_net.growthpolicy import \
            DEFAULT_GROWTH_POLICY as _gpp
        # G-ASPECT (60D): pre-snapshot; width = d_model,
        # depth_after = L + 1 (the inserted layer)
        from reference_net.method.gates import gate_aspect
        gate_aspect(getattr(self, "_growth_policy", _gpp),
                    self.d, self.L + 1, force=force)
        from reference_net.growth_store import auto_snapshot
        auto_snapshot(self, getattr(self, "_growth_policy",
                                    _gpp))       # FR-13 (D-6.2)
        """Whole-layer insertion at any p in [0, L] (doc 55
        s3-B2; the formally proven function-preserving form,
        survey 49 s2.2): keys renumber l -> l+1 for l >= p
        (shared surgery); the new layer is born LN-identity
        with ZERO attention out-projection (Wo) and ZERO FFN
        second matrix (W2) under the default zero side —
        exactly nothing is contributed at birth, at any p.
        recipe "copy_layer" copies the designated source
        layer's tensors (with zero_side="none": COMPLETE copy,
        SOLAR form, non-preserving by declared choice). Events
        carry verbatim specs in self.growth_events (these
        hosts have no gain ledger — recorded fact)."""
        from reference_net.growth_port import (layer_census,
                                               shift_layer_maps)
        from reference_net.foundation.specs import (
            ALL, BirthSpec, PlacementSpec, StructureSpec, Tap,
            WiringSpec, specs_as_dict)
        p = int(position)
        if not 0 <= p <= self.L:
            raise ValueError(f"insert position {p} out of "
                             f"range 0..{self.L}")
        if recipe not in ("random", "copy_layer"):
            raise ValueError(f"insert_layer recipe {recipe!r}: "
                             "allowed: random, copy_layer")
        nbr = source_index
        if recipe == "copy_layer" and nbr is None:
            nbr = p - 1 if p > 0 else 0
        census_pre = layer_census(self, self._LAYER_KEY_ROOTS)
        self._seed_counter += 1
        rng = np.random.default_rng(self._seed_counter)
        shift_layer_maps(self, p, self._LAYER_KEY_ROOTS)
        d, m = self.d, self.m
        bk = self._bk
        n_ = lambda a: np.asarray(bk.to_numpy(a))
        if recipe == "copy_layer":
            src = nbr if nbr < p else nbr + 1   # post-shift
            vals = {r: n_(self.P[f"{r}_{src}"]).copy()
                    for r in self._LAYER_KEY_ROOTS}
        else:
            vals = {"g1": np.ones(d), "b1n": np.zeros(d),
                    "g2": np.ones(d), "b2n": np.zeros(d),
                    "W1": rng.normal(0, np.sqrt(2.0 / d),
                                     (d, m)),
                    "b1": np.zeros(m),
                    "W2": rng.normal(0, np.sqrt(1.0 / m),
                                     (m, d)),
                    "b2": np.zeros(d)}
            for w in ("Wq", "Wk", "Wk2", "Wo"):
                vals[w] = rng.normal(0, np.sqrt(1.0 / d),
                                     (d, d))
        if zero_side == "default":
            vals["W2"] = np.zeros_like(vals["W2"])
            vals["Wo"] = np.zeros_like(vals["Wo"])
        for r in self._LAYER_KEY_ROOTS:
            self.P[f"{r}_{p}"] = bk.ingest(vals[r])
            self._adam[f"{r}_{p}"] = [
                bk.zeros_like(self.P[f"{r}_{p}"]),
                bk.zeros_like(self.P[f"{r}_{p}"])]
        def _zl(a):        # D-W6-2 (same family law)
            return (np.zeros_like(a)
                    if isinstance(a, np.ndarray)
                    else bk.zeros_like(a))
        if isinstance(getattr(self, "_ema_dw", None), dict):
            self._ema_dw[p] = _zl(
                self._ema_dw[min(self._ema_dw)])
        if isinstance(getattr(self, "_ema_adw", None), dict):
            self._ema_adw[p] = _zl(
                self._ema_adw[min(self._ema_adw)])
        self.L += 1
        census_post = layer_census(self, self._LAYER_KEY_ROOTS)
        assert len(census_post["P"]) ==             len(census_pre["P"]) + len(self._LAYER_KEY_ROOTS)
        sspec = StructureSpec(kind="attn_layer",
                              params={"d": d, "m": m,
                                      "heads": self.h},
                              seed=self._seed_counter,
                              lr=self.lr)
        wspec = WiringSpec(reads=[Tap("stream", span=ALL)],
                           write={"target": "stream",
                                  "span": ALL})
        pspec = PlacementSpec(chain="layers", position=p)
        bspec = BirthSpec(zero_side=("attn_out_proj+ffn_W2"
                                     if zero_side == "default"
                                     else "none"),
                          recipe=recipe,
                          recipe_params={"source_index": nbr}
                          if nbr is not None else {})
        if not hasattr(self, "growth_events"):
            self.growth_events = []
        self.growth_events.append(
            {"event": "deepen_layer[attn]", "position": p,
             "specs": specs_as_dict(sspec, wspec, pspec,
                                    bspec),
             "wall_ms": (_time.perf_counter() - _t0) * 1e3,
             **({"forced": True} if force else {})})
        return p

    def grow_site(self, site_path, hidden=16, body_type=None,
                  force=False):
        # body_type applies to Network-body growth (the deep path);
        # layer FFN sites create this host's two-timescale reference
        # inner BY DESIGN (S9.5 docstring convention).
        if "::" in site_path:                       # deep growth
            outer, inner_site = site_path.split("::", 1)
            l, j = self._parse(outer)
            net = self.grown_body(l, j)   # port or legacy carrier
            if net is None:
                raise ValueError(
                    f"deep growth under an ungrown site: {outer}")
            path, k = inner_site.rsplit("[", 1)
            k = int(k.rstrip("]"))
            owner = net
            if path != "root":
                for hop in path.split("/"):
                    owner = owner.grown_body(int(hop))
                    if owner is None:
                        raise KeyError(int(hop))
            owner.grow(k, hidden=hidden, body_type=body_type,
                       force=force)
        else:
            l, j = self._parse(site_path)
            # Two-timescale rule for THIS host (addition protocol,
            # evidence-driven): with inner lr == outer lr, post-growth
            # coupled training CORRUPTED learning on the plateau fixture
            # (0.070 -> 0.180); at inner lr = 0.3x outer it improves to
            # 0.043 and beats the no-growth control (0.103). Applied to
            # the transformer host only; the mlp host is validated at
            # equal rates and stays unchanged (minimal core).
            # Growth Interface Reform (doc 36 W3(a)): FFN growth
            # routes through the ONE shared fullwidth port (R2).
            import time
            t0 = time.perf_counter()
            from reference_net.growthpolicy import \
                DEFAULT_GROWTH_POLICY as _gpp
            from reference_net.growth_store import auto_snapshot
            auto_snapshot(self, getattr(self, "_growth_policy",
                                        _gpp))   # FR-13 (D-6.2)
            from reference_net.growth_port import grow_ffn_body
            grow_ffn_body(self, l, j, hidden, site_path,
                          force=force)
            rec = self.growth_events[-1]         # D-6.1 stamp
            rec["wall_ms"] = (time.perf_counter() - t0) * 1e3
            if force:
                rec["forced"] = True             # C-5
        return {"grown": site_path, "params": self.n_params(),
                "depth": self.depth()}

    def grown_body(self, l, j):
        """The body grown at FFN site (l, j) — fullwidth (port
        slot) or legacy (loaded artifact). None if never grown."""
        if (l, j) in self.inner:
            return self.inner[(l, j)]
        site = getattr(self, "_port_sites", {}).get(l)
        return site.body_by_key((l, j)) if site is not None \
            else None

    @staticmethod
    def _parse(site):
        l, rest = site.split("/", 1)
        return int(l.replace("layer", "")), int(rest[4:-1])

    # ---------------- introspection ----------------
    def depth(self):
        kids = [n.depth() for n in self.inner.values()]
        for site in getattr(self, "_port_sites", {}).values():
            kids += [s["body"].depth() for s in site.bodies]
        return 1 + max(kids, default=0)

    def n_params(self):
        own = sum(self._bk.numel(v) for v in self.P.values())
        own += sum(site.n_params() for site in
                   getattr(self, "_port_sites", {}).values())
        return own + sum(n.n_params() for n in self.inner.values())

    def shape_record(self):
        return {"mode": self.mode, "vocab": list(self.vocab),
                "depth": self.depth(), "params": self.n_params(),
                "d_in": self.d_in, "hidden": self.m,
                "substrate": self.NAME}

    def perturb(self, rng, sigma):
        """Contract helper for practice attempts: jittered copy."""
        import copy
        p = copy.deepcopy(self)
        p.P["Wh"] = p.P["Wh"] + rng.normal(0, sigma, p.P["Wh"].shape)
        for l in range(p.L):
            p.P[f"W1_{l}"] = p.P[f"W1_{l}"] + rng.normal(
                0, sigma, p.P[f"W1_{l}"].shape)
        return p

    # ---------------- artifact ----------------
    # Security note: pickle for locally-produced artifacts inside the
    # system's own storage tree (same trust domain), as project-wide.
    def __getstate__(self):
        bk = getattr(self, "_bk", None) or get_default_backend()
        st = self.__dict__.copy()
        st.pop("_bk", None)
        st.pop("_spu_next_cache", None)      # transient
        st.pop("_spu_input_cache", None)     # transient
        st.pop("_snapshots", None)   # observer state (I-7):
        st.pop("_monitor", None)     # never artifact content
        st["P"] = {k: bk.to_numpy(v) for k, v in self.P.items()}
        st["_adam"] = {k: [bk.to_numpy(m_), bk.to_numpy(v_)]
                       for k, (m_, v_) in self._adam.items()}
        for k in ("_x_mu", "_x_sd"):
            if st.get(k) is not None and not isinstance(
                    st[k], (int, float)):
                st[k] = bk.to_numpy(st[k])
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_bk" not in self.__dict__:
            self._bk = get_default_backend()
        # D6 (doc 35): pre-reform grown artifacts (no port field)
        # load AS legacy_scalar — audit-visible marker only.
        if self.__dict__.get("inner") \
                and "_port_sites" not in self.__dict__ \
                and "_legacy_port" not in self.__dict__:
            from reference_net.growth_port import LegacyScalarPort
            self._legacy_port = LegacyScalarPort()

    def _ingest_state(self):
        """G7 (doc 61 I-A): move every learned tensor onto
        self._bk — the GA hook's mirror MINUS heads: P,
        _adam, x-scalers, port sites recursively. Faithful to
        LIVE construction (probed): the EMA dicts STAY numpy
        on this host (np.zeros at __init__; numpy updates)."""
        bk = self._bk
        self.P = {k: bk.ingest(v) for k, v in self.P.items()}
        self._adam = {k: [bk.ingest(m), bk.ingest(v)]
                      for k, (m, v) in self._adam.items()}
        if getattr(self, "_x_mu", None) is not None:
            self._x_mu = bk.ingest(self._x_mu)
            self._x_sd = bk.ingest(self._x_sd)
        for site in getattr(self, "_port_sites", {}).values():
            if hasattr(site, "ingest_to"):
                site.ingest_to(bk)

    def save(self, dir_path):
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "msorgan.pkl", "wb") as f:
            pickle.dump(self, f)
        (d / "substrate.json").write_text(json.dumps(
            {"substrate": self.NAME, "contract": CONTRACT_V}))

    @staticmethod
    def load(dir_path):
        with open(Path(dir_path) / "msorgan.pkl", "rb") as f:
            return pickle.load(f)
