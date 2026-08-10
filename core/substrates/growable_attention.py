"""The `growable_attention` substrate — growable, self-processing
attention host (docs/attention-build chain: REQUIREMENTS v1.6 A1-A3,
ALGO_PSEUDOCODE v1.5).

Named for the METHOD (growth + self-processing on attention), not the
implementation; the per-head UNPACKED storage that makes ragged widths
possible is an implementation note: each head owns its four matrices
{W~q, W_k, W_v, W_o} so head widths may differ and grow independently.
Scoring is SCALE-FREE: the 1/sqrt(d_k) normalizer is absorbed into W~q
at birth (W~q := W_q_raw / sqrt(d_h_birth)) — under widening a runtime
division would shift every score O(1); absorption keeps growth exact.
The block output is the additive form sum_h (A_h V_h) W_o^h, which is
the packed concat identically (blockwise matrix product).

Judge form (numpy) throughout; the backend kernel port is out of v1
scope (plan P0). Numeric mode first (S1); the forward mirrors
transformer.py's numeric path with the packed attention block replaced
by the per-head loop.

Build stage: S1 (skeleton + forward + contract + artifact).
train_step lands in S2 (backward); growth operators in S3;
self-processing in S4 (plan D2).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ENGINE = Path(__file__).resolve().parents[2] / "modules" / "Engine"
if _ENGINE.is_dir() and str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from engine.primitives import gelu, gelu_d, ln_bwd, ln_fwd  # noqa: E402

from core.substrates.base import CONTRACT_V, Substrate  # noqa: E402
from core.substrates.heads import B_NEG  # noqa: E402  (S9.2 D-G4)
from core.substrates.transformer import _eta_t  # noqa: E402  (S2)

_REFNET = Path(__file__).resolve().parents[2] / "modules" / "ReferenceNet"
if _REFNET.is_dir() and str(_REFNET) not in sys.path:
    sys.path.insert(0, str(_REFNET))

# policy defaults (ALGO P1); calibration in DEV_PLAN D8, never
# hard-coded outside this dict
POLICY = {
    "att_birth_heads": 1,
    "att_birth_dh": 1,
    "att_widen_m": 1,
    "att_kappa": 2.0,
    "att_beta": 0.1,
    "att_lambda": 0.05,
    "att_warmup": 100,
    "att_h_lo": 0.2,
    "att_h_hi": 1.0,
    "att_selfproc_heads": None,
    "att_head_age_min": 100,
    "att_probe_steps": 20,
    "att_window": 200,
}


_HMATS = ("Wq", "Wk", "Wv", "Wo")   # canonical head-matrix order
                                    # (the old flat mu/nu layout)


def _copy_headstate(hs):
    """Full tensor copy of a HeadState (fresh Adam slots)."""
    import copy as _copy
    new = _copy.deepcopy(hs)
    return new


class HeadState:
    """One attention head: four PERMANENT matrices + growth metadata.

    Wq is stored ABSORBED (already divided by sqrt(d_h_birth)).
    mu/nu are the u_h EMA buffers over the update vector — stored
    PER MATRIX since the backend port (doc 32 DD-7; elementwise
    EMA + additive norm-squares make the values identical to the
    old flat [Wq, Wk, Wv, Wo] layout; extended zeroed on widen)."""

    def __init__(self, d, d_h, rng, birth_t=0):
        sd = np.sqrt(1.0 / d)
        self.Wq = rng.normal(0, sd, (d, d_h)) / np.sqrt(d_h)
        self.Wk = rng.normal(0, sd, (d, d_h))
        self.Wv = rng.normal(0, sd, (d, d_h))
        self.Wo = rng.normal(0, sd, (d_h, d))
        self.d_h_birth = d_h
        self.birth_t = birth_t
        self.mu = {nm: np.zeros_like(getattr(self, nm))
                   for nm in _HMATS}
        self.nu = {nm: np.zeros_like(getattr(self, nm)) + 1e-12
                   for nm in _HMATS}

    @property
    def d_h(self):
        return self.Wv.shape[1]

    def n_params(self):
        return sum(int(np.prod(m.shape))
                   for m in (self.Wq, self.Wk, self.Wv, self.Wo))

    def __getstate__(self):
        """Backend-portable pickling (doc 32 D8): tensors leave as
        numpy; the host's __setstate__ re-ingests."""
        st = dict(self.__dict__)
        bk = st.pop("_bk", None)
        if bk is not None:
            for nm in _HMATS:
                st[nm] = np.asarray(bk.to_numpy(st[nm]))
            st["mu"] = {k: np.asarray(bk.to_numpy(v))
                        for k, v in st["mu"].items()}
            st["nu"] = {k: np.asarray(bk.to_numpy(v))
                        for k, v in st["nu"].items()}
        return st


class GrowableAttentionSubstrate(Substrate):
    NAME = "growable_attention"
    DATA_FORM = "vector"
    SUPPORTED_HEADS = ("point", "dist")   # 60A

    CAUSAL = False
    WINDOW = 16
    INNER_LR_FACTOR = 0.3   # was inlined at the grow site; class
                            # attr documents the calibrated default
                            # (param-interface batch, S2)

    def __init__(self, d_in, hidden, mode="numeric", vocab=None,
                 lr=1e-2, seed=7, d_model=32, n_layers=2,
                 heads_spec=None, causal=False, selfproc=False,
                 window=None, inner_lr_factor=None,
                 new_class_bias=None, backend=None):
        """heads_spec: per-layer head-width lists, e.g. [[1,3],[2,1]]
        (explicit ragged construction for tests and for artifacts);
        None -> POLICY birth: att_birth_heads heads of width
        att_birth_dh per layer.
        window: causal context length (positional-table size);
        None -> the class default (16). Param-interface batch,
        docs/system/22 item 1.
        backend: None -> the system compute policy
        (set_compute_policy); a registered backend name; or a
        backend instance — doc 32 D9, transformer-row parity."""
        if backend is None:
            from engine.backends import current_backend
            self._bk = current_backend()
        elif isinstance(backend, str):
            from engine.backends import resolve_backend
            self._bk = resolve_backend(backend)
        elif hasattr(backend, "ingest") and hasattr(backend,
                                                    "to_numpy"):
            self._bk = backend
        else:
            raise ValueError(
                "backend must be None, a registered backend name, "
                f"or a backend instance; got {type(backend).__name__}")
        self._masks = {}       # Tlen -> ingested causal mask
        if window is not None:
            if not isinstance(window, (int, np.integer)) \
                    or isinstance(window, bool) or window < 1:
                raise ValueError(
                    f"window must be an integer >= 1; got {window!r}")
            self.WINDOW = int(window)
        if inner_lr_factor is not None:
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
        if mode not in ("numeric", "numeric_dist",
                        "categorical"):
            # 60A L3 whitelist assertion (supersedes the S9.2
            # boundary: the dist head now serves here too)
            raise ValueError(
                f"growable_attention serves mode='numeric', "
                f"'numeric_dist' and 'categorical'; got {mode!r}")
        rng = np.random.default_rng(seed)
        self.d_in, self.m, self.mode = d_in, hidden, mode
        self.d, self.L = d_model, n_layers
        self.lr, self.seed = lr, seed
        self.vocab = list(vocab) if vocab else []
        n_out = (1 if mode == "numeric" else
                 2 if mode == "numeric_dist" else
                 max(1, len(self.vocab)))
        d = d_model
        if heads_spec is None:
            heads_spec = [[self._pol("att_birth_dh")]
                          * self._pol("att_birth_heads")
                          for _ in range(n_layers)]
        assert len(heads_spec) == n_layers, "heads_spec: one list/layer"
        self._t_att = 0
        self.heads = [[HeadState(d, dh, rng, birth_t=0)
                       for dh in layer_spec]
                      for layer_spec in heads_spec]

        self.CAUSAL = bool(causal)
        if self.CAUSAL:
            # sequence tokens (transformer.py:85-91 pattern)
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
            P[f"W1_{l}"] = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
            P[f"b1_{l}"] = np.zeros(hidden)
            P[f"W2_{l}"] = rng.normal(0, np.sqrt(1.0 / hidden),
                                      (hidden, d))
            P[f"b2_{l}"] = np.zeros(d)
        if mode == "numeric_dist":
            # 60A: heteroscedastic head born ZERO (birth honesty,
            # the mlp precedent — closed-form first loss 0.5)
            P["Wh"] = np.zeros((d, 2))
            P["bh"] = np.zeros(2)
        self.P = P
        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)]
                      for k, v in P.items()}
        self._adam_h = {}          # (l, h, name) -> [m, v]
        self._sync_head_slots()
        self._t = 0
        self.inner: dict = {}      # (layer, unit) -> inner body (FFN)
        self._ema_dw = {l: np.zeros((d, hidden))
                        for l in range(n_layers)}
        self._ema_adw = {l: np.zeros((d, hidden)) + 1e-12
                         for l in range(n_layers)}
        self._seed_counter = seed
        self._x_mu = self._x_sd = None
        self._y_mu = self._y_sd = None
        # A2 discipline switch (S4); S9.3 D-G2: constructor-
        # settable (reachable via substrate_params), default off
        self._selfproc_on = bool(selfproc)
        self._ingest_state()   # -> the selected backend (doc 32 D1;
        #                        identity on the numpy judge)

    # ---------------- backend plumbing (doc 32 9.1) ----------------
    def _ingest_state(self):
        """Move every learned tensor onto self._bk. Identity on the
        numpy judge; called at construction and unpickle."""
        bk = self._bk
        self.P = {k: bk.ingest(v) for k, v in self.P.items()}
        self._adam = {k: [bk.ingest(m), bk.ingest(v)]
                      for k, (m, v) in self._adam.items()}
        self._adam_h = {k: [bk.ingest(m), bk.ingest(v)]
                        for k, (m, v) in self._adam_h.items()}
        for layer in self.heads:
            for HS in layer:
                HS._bk = bk
                for nm in _HMATS:
                    setattr(HS, nm, bk.ingest(getattr(HS, nm)))
                HS.mu = {k: bk.ingest(v) for k, v in HS.mu.items()}
                HS.nu = {k: bk.ingest(v) for k, v in HS.nu.items()}
        self._ema_dw = {l: bk.ingest(v)
                        for l, v in self._ema_dw.items()}
        self._ema_adw = {l: bk.ingest(v)
                         for l, v in self._ema_adw.items()}
        if self._x_mu is not None:
            self._x_mu = bk.ingest(self._x_mu)
            self._x_sd = bk.ingest(self._x_sd)
        # G7 (doc 61 I-A) EXTENSION: port sites — couplings'
        # tensors and grown bodies recursively — follow the
        # backend; device-stale caches cleared.
        ports = getattr(self, "_port_sites", None)
        if ports:
            it = ports.values() if isinstance(ports, dict) \
                else ports
            for site in it:
                if hasattr(site, "ingest_to"):
                    site.ingest_to(bk)
        if hasattr(self, "_masks"):
            self._masks = {}

    def _causal_mask(self, Tlen):
        m = self._masks.get(Tlen)
        if m is None:
            m = self._bk.ingest(
                np.triu(np.full((Tlen, Tlen), -1e9), k=1))
            self._masks[Tlen] = m
        return m

    # ---------------- head optimizer slots ----------------
    def _sync_head_slots(self):
        """Adam slots per head matrix; created zeroed at birth,
        EXTENDED zeroed on widen (S3 rebuilds only new tails)."""
        for l, layer in enumerate(self.heads):
            for h, HS in enumerate(layer):
                for name in _HMATS:
                    key = (l, h, name)
                    M = getattr(HS, name)
                    if key not in self._adam_h or \
                            tuple(self._adam_h[key][0].shape) \
                            != tuple(M.shape):
                        # zeros(shape): M may still be numpy at
                        # construction time (ingest runs last)
                        self._adam_h[key] = [
                            self._bk.zeros(tuple(M.shape)),
                            self._bk.zeros(tuple(M.shape))]

    # ---------------- self-processing (ALGO P5; COMPUTE C3) ----
    def _pol(self, key):
        """S9.5b D-G5: instance-first attention-policy read — the
        organ's _att_policy (installed by the lifecycle as the FULL
        merged table) wins; absent attribute -> module POLICY, the
        default table and the direct-construction override surface
        (experiment drivers unchanged)."""
        return getattr(self, "_att_policy", POLICY)[key]

    def selfproc_active(self, l, h):
        """Three gates: global warmup, per-head age, per-head
        switch (None = all heads)."""
        if not self._selfproc_on:
            return False
        if self._t_att < self._pol("att_warmup"):
            return False
        HS = self.heads[l][h]
        if self._t_att - HS.birth_t < self._pol("att_head_age_min"):
            return False
        # S9.3 D-G2: per-model allow-set (organ attribute, set by
        # the set_attention_selfproc verb) takes precedence over
        # the module POLICY default; same membership semantics.
        allow = getattr(self, "_selfproc_heads", None)
        if allow is None:
            allow = self._pol("att_selfproc_heads")
        return allow is None or (l, h) in allow or h in (allow or ())

    def _datt_dS(self, A):
        """Certified J_att gradient at the scores (cert S2: causal,
        vector, and binding-upper-hinge configs; 1/N_valid carried
        AS WRITTEN). Returns dS_att, same shape as A."""
        n, F, _ = A.shape
        if self.CAUSAL:
            F_i = np.arange(1, F + 1)
        else:
            F_i = np.full(F, F)
        valid = F_i >= 2
        N_valid = int(valid.sum()) * n
        dS_att = np.zeros_like(A)
        if N_valid == 0:
            return dS_att
        lo_f, hi_f = self._pol("att_h_lo"), self._pol("att_h_hi")
        for b in range(n):
            for i in range(F):
                if not valid[i]:
                    continue
                pp = A[b, i, :F_i[i]]
                pc = np.maximum(pp, 1e-300)
                Hi = float(-(pp * np.log(pc)).sum())
                lo = lo_f * np.log(F_i[i])
                hi = hi_f * np.log(F_i[i])
                dJdH = (2 * max(0.0, Hi - hi)
                        - 2 * max(0.0, lo - Hi)) / N_valid
                if dJdH != 0.0:
                    dS_att[b, i, :F_i[i]] = \
                        dJdH * (-pp * (np.log(pc) + Hi))
        return dS_att

    def j_att_value(self, l, h, X):
        """Audit read-out (C4): the discipline objective on head
        (l, h) for input X. Read-only."""
        _, _, caches = self._forward(self._stdx(np.asarray(X, float)),
                                     cache=True)
        A = np.asarray(self._bk.to_numpy(caches[1 + l][3][h][3]))
        n, F, _ = A.shape
        F_i = (np.arange(1, F + 1) if self.CAUSAL
               else np.full(F, F))
        tot, cnt = 0.0, 0
        for b in range(n):
            for i in range(F):
                if F_i[i] < 2:
                    continue
                pp = A[b, i, :F_i[i]]
                pc = np.maximum(pp, 1e-300)
                Hi = float(-(pp * np.log(pc)).sum())
                lo = self._pol("att_h_lo") * np.log(F_i[i])
                hi = self._pol("att_h_hi") * np.log(F_i[i])
                tot += max(0.0, Hi - hi) ** 2 \
                    + max(0.0, lo - Hi) ** 2
                cnt += 1
        return tot / max(cnt, 1)

    # ---------------- forward (ALGO P3) ----------------
    def _forward(self, X, cache=False):
        # NATIVE-OPERATOR form (doc 32 D2): matmul/add/slice are
        # written exactly as before (identical on numpy arrays and
        # backend tensors); kernels only at divergence points.
        bk = self._bk
        n = len(X)
        if self.CAUSAL:
            Tlen = X.shape[1]
            T = X @ self.P["Wv"] + self.P["Bf"][None, :Tlen, :]
        else:
            T = X[:, :, None] * self.P["Wv"][None] + self.P["Bf"][None]
        caches = [("tokens", X)]
        for l in range(self.L):
            Tn, c1 = bk.ln_fwd(T, self.P[f"g1_{l}"],
                               self.P[f"b1n_{l}"])
            O = bk.zeros_like(T)
            att_cache_l = []
            for HS in self.heads[l]:
                Qh = Tn @ HS.Wq
                Kh = Tn @ HS.Wk
                Vh = Tn @ HS.Wv
                S = Qh @ Kh.swapaxes(1, 2)             # SCALE-FREE
                if self.CAUSAL:
                    S = S + self._causal_mask(S.shape[-1])
                A = bk.exp(S - bk.rowmax_keep(S))
                A = A / bk.rowsum_keep(A)
                Oh = A @ Vh
                O = O + Oh @ HS.Wo                     # additive form
                att_cache_l.append((Qh, Kh, Vh, A, Oh))
            T1 = T + O
            Tn2, c2 = bk.ln_fwd(T1, self.P[f"g2_{l}"],
                                self.P[f"b2n_{l}"])
            pre = Tn2 @ self.P[f"W1_{l}"] + self.P[f"b1_{l}"]
            H = bk.gelu(pre)
            inner_in = Tn2.reshape(-1, self.d)
            from reference_net.growth_port import \
                legacy_attach_layer
            H = legacy_attach_layer(H, inner_in, self.inner,
                                    l, bk, n)
            ports = getattr(self, "_port_sites", None)
            if ports and ports.get(l) is not None \
                    and ports[l].bodies:
                H = ports[l].forward(
                    H.reshape(-1, self.m),
                    inner_in).reshape(n, -1, self.m)
            F_out = H @ self.P[f"W2_{l}"] + self.P[f"b2_{l}"]
            T2 = T1 + F_out
            if cache:
                caches.append((T, Tn, c1, att_cache_l, T1, Tn2, c2,
                               pre, H, inner_in))
            T = T2
        Pool = T[:, -1, :] if self.CAUSAL else T.mean(1)
        logits = Pool @ self.P["Wh"] + self.P["bh"]
        return (logits, Pool, caches) if cache else logits

    # ---------------- serving ----------------
    def _stdx(self, X):
        """Ingest boundary: accepts numpy (all callers), returns a
        standardized backend tensor."""
        X = self._bk.ingest(np.asarray(X, float))
        if self._x_mu is None:
            return X
        return (X - self._x_mu) / self._x_sd

    def _fit_x_scalers(self, X):
        X = np.asarray(X, float)
        if self.CAUSAL:
            self._x_mu = self._bk.ingest(X.mean((0, 1)))
            self._x_sd = self._bk.ingest(X.std((0, 1)) + 1e-8)
        else:
            self._x_mu = self._bk.ingest(X.mean(0))
            self._x_sd = self._bk.ingest(X.std(0) + 1e-8)

    def predict(self, X):
        if self.mode == "numeric_dist":                     # 60A
            value, _ = self.predict_dist(X)
            return np.asarray(value).reshape(-1, 1)
        assert self.mode == "numeric"
        X = np.asarray(X, float)
        if self._y_mu is None:
            return np.zeros((len(X), 1))
        z = self._bk.to_numpy(self._forward(self._stdx(X)))
        return z * self._y_sd + self._y_mu

    def predict_dist(self, X):
        """60A: heteroscedastic serve (mlp conventions; the v
        clamp mirrors the kernel's NLL_CLAMP)."""
        assert self.mode == "numeric_dist"
        if self._y_mu is None:
            raise ValueError(
                "untrained: predict_dist needs at least one "
                "training step (the target scalers are unfitted)")
        X = np.asarray(X, float)
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
        # S9.2 D-G4: transformer.py:204-212 pattern (numpy form) —
        # uniform prior before the scalers exist, softmax after.
        assert self.mode == "categorical"
        X = np.asarray(X, float)
        if self._x_mu is None:
            c = max(1, len(self.vocab))
            return np.full((len(X), c), 1.0 / c)
        logits = self._bk.to_numpy(self._forward(self._stdx(X)))
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict_label(self, X):
        p = self.predict_proba(X)
        idx = p.argmax(axis=1)
        return ([self.vocab[i] for i in idx],
                p[np.arange(len(idx)), idx])

    def add_class(self, label):
        # S9.2 D-G4: transformer.py:218-231 pattern — zero logit
        # column + B_NEG bias (function-preserving within epsilon),
        # adam slots rebuilt for the head matrices only.
        assert self.mode == "categorical" and label not in self.vocab
        self.vocab.append(label)
        bk = self._bk          # event round-trip (doc 32 D3)
        self.P["Wh"] = bk.ingest(np.hstack(
            [np.asarray(bk.to_numpy(self.P["Wh"])),
             np.zeros((self.d, 1))]))
        self.P["bh"] = bk.ingest(np.concatenate(
            [np.asarray(bk.to_numpy(self.P["bh"])),
             [getattr(self, "new_class_bias", B_NEG)]]))
        self._adam["Wh"] = [bk.zeros_like(self.P["Wh"]),
                            bk.zeros_like(self.P["Wh"])]
        self._adam["bh"] = [bk.zeros_like(self.P["bh"]),
                            bk.zeros_like(self.P["bh"])]

    # ---------------- training (ALGO P4; task path — the J_att
    # injection point is marked and lands in S4) ----------------
    def train_step(self, X, y, sgd_lr=None):
        X = np.asarray(X, float)
        n = len(X)
        if self._x_mu is None:
            self._fit_x_scalers(X)
            if self.mode in ("numeric", "numeric_dist"):
                ya = np.asarray(y, float).reshape(-1, 1)
                self._y_mu = float(ya.mean())
                self._y_sd = float(ya.std() + 1e-8)
        bk = self._bk
        Xs = self._stdx(X)
        if self.mode == "numeric":
            # y-rescale guard is NUMERIC-ONLY (S9.2 D-G4, as on the
            # transformer host) — the block below is the pre-S9.2
            # text verbatim.
            ya = np.asarray(y, float).reshape(-1, 1)
            mb, sb = float(ya.mean()), float(ya.std() + 1e-8)
            if sb > 1.5 * self._y_sd or abs(mb - self._y_mu) \
                    > 2 * self._y_sd:
                sd_n = max(sb, self._y_sd)
                self.P["Wh"] = self.P["Wh"] * (self._y_sd / sd_n)
                self.P["bh"] = (self.P["bh"] * self._y_sd
                                + self._y_mu - mb) / sd_n
                self._y_mu, self._y_sd = mb, sd_n
                self._adam = {k: [bk.zeros_like(v),
                                  bk.zeros_like(v)]
                              for k, v in self.P.items()}
                for key, (m_, v_) in self._adam_h.items():
                    self._adam_h[key] = [bk.zeros_like(m_),
                                         bk.zeros_like(v_)]
                self._t = 0
            ys = bk.ingest((ya - self._y_mu) / self._y_sd)
        if self.mode == "numeric_dist":                     # 60A
            ya = np.asarray(y, float).reshape(-1, 1)
            ys = bk.ingest((ya - self._y_mu) / self._y_sd)

        logits, Pool, caches = self._forward(Xs, cache=True)
        if self.mode == "numeric":
            err = logits - ys
            loss = float(bk.to_numpy((err ** 2).mean()))
            dlog = 2 * err / n
        elif self.mode == "numeric_dist":
            # 60A: engine NLL kernels reused VERBATIM; inert zero
            # placeholders for the mlp-shaped trunk arguments
            t = ys.reshape(-1)
            _cl = getattr(self, "nll_clamp", None)
            mu, v = bk.nll_forward(self.P["Wh"].T,
                                   self.P["bh"], Pool,
                                   nll_clamp=_cl)
            loss = bk.nll_loss(t, mu, v)
            _z1 = bk.zeros((n, 1))
            _gk, _dHk = bk.nll_backward(
                _z1, _z1, self.P["Wh"].T, self.P["bh"], _z1,
                bk.zeros_like(Pool), Pool, mu, v, t,
                nll_clamp=_cl)
            dlog = None            # head grads mapped below
        else:
            # S9.2 D-G4: softmax cross-entropy with one-hot labels
            # (transformer.py:274-281 pattern; loss formula matches
            # the backends' cat_ce: -mean(log p_true))
            labels = [self.vocab.index(v)
                      for v in np.asarray(y).ravel()]
            e = bk.exp(logits - bk.rowmax_keep(logits))
            probs = e / bk.rowsum_keep(e)
            onehot = np.zeros((n, len(self.vocab)))
            onehot[np.arange(n), labels] = 1.0
            onehot = bk.ingest(onehot)
            loss = float(-np.mean(np.log(
                np.asarray(bk.to_numpy((probs * onehot).sum(1)))
                + 1e-12)))
            dlog = (probs - onehot) / n

        G = {k: bk.zeros_like(v) for k, v in self.P.items()}
        Gh = {k: bk.zeros_like(getattr(self.heads[k[0]][k[1]],
                                       k[2]))
              for k in self._adam_h}
        if self.mode == "numeric_dist":
            G["Wh"] = _gk[2].T     # 60A seam identity (kernel /n)
            G["bh"] = _gk[3]
            dPool = _dHk / n
        else:
            G["Wh"] = Pool.T @ dlog
            G["bh"] = dlog.sum(0)
            dPool = dlog @ self.P["Wh"].T
        Fn = Xs.shape[1]
        if self.CAUSAL:
            dT = bk.zeros((n, Fn, self.d))
            dT[:, -1, :] = dPool                    # last-token head
        else:
            dT = bk.repeat_tokens(dPool, Fn)

        d = self.d
        handoffs = []
        port_calls = []
        for l in range(self.L - 1, -1, -1):
            (T, Tn, c1, att_cache_l, T1, Tn2, c2,
             pre, H, inner_in) = caches[1 + l]
            # FFN branch (transformer.py:300-319 pattern)
            dF = dT
            G[f"W2_{l}"] += H.reshape(-1, self.m).T \
                @ dF.reshape(-1, d)
            G[f"b2_{l}"] += dF.sum((0, 1))
            dH = dF @ self.P[f"W2_{l}"].T
            from reference_net.growth_port import \
                legacy_collect_layer
            for net_, r_ in legacy_collect_layer(
                    self.inner, l, H, dH, pre, _eta_t(self),
                    bk.gelu):
                handoffs.append((net_, inner_in, r_))
            ports = getattr(self, "_port_sites", None)
            if ports and ports.get(l) is not None \
                    and ports[l].bodies:
                port_calls.append(
                    (ports[l], dH.reshape(-1, self.m),
                     inner_in))
            dpre = dH * bk.gelu_d(pre)
            G[f"W1_{l}"] += Tn2.reshape(-1, d).T \
                @ dpre.reshape(-1, self.m)
            G[f"b1_{l}"] += dpre.sum((0, 1))
            dTn2 = dpre @ self.P[f"W1_{l}"].T
            dT1, dg2, db2 = bk.ln_bwd(dTn2, c2, self.P[f"g2_{l}"])
            G[f"g2_{l}"] += dg2
            G[f"b2n_{l}"] += db2
            dT1 = dT1 + dT                          # residual skip
            # attention branch — per-head loop (ALGO P4)
            dO = dT1
            dTn = bk.zeros_like(Tn)
            for h, HS in enumerate(self.heads[l]):
                (Qh, Kh, Vh, A, Oh) = att_cache_l[h]
                d_h = HS.d_h
                Gh[(l, h, "Wo")] += Oh.reshape(-1, d_h).T \
                    @ dO.reshape(-1, d)
                dOh = dO @ HS.Wo.T
                dA = dOh @ Vh.swapaxes(1, 2)
                dVh = A.swapaxes(1, 2) @ dOh
                dS_task = A * (dA - bk.rowsum_keep(dA * A))
                # ### ACCUMULATION POINT (COMPUTE C3, binding —
                # SPLIT): PARAMETER grads use dS_total; the
                # BLOCK-INPUT dTn uses dS_task ONLY (routing
                # dS_total into dTn leaks J_att into LN/embedding/
                # every lower layer — measured; T6c red-arms it).
                if self.selfproc_active(l, h):
                    # discipline hinge = declared CPU island
                    # (doc 32 9.2 I): judge-form loops verbatim
                    dS_total = dS_task + self._pol("att_lambda") \
                        * bk.ingest(self._datt_dS(
                            np.asarray(bk.to_numpy(A))))
                else:
                    dS_total = dS_task
                dQh = dS_total @ Kh
                dKh = dS_total.swapaxes(1, 2) @ Qh
                Gh[(l, h, "Wq")] += Tn.reshape(-1, d).T \
                    @ dQh.reshape(-1, d_h)
                Gh[(l, h, "Wk")] += Tn.reshape(-1, d).T \
                    @ dKh.reshape(-1, d_h)
                Gh[(l, h, "Wv")] += Tn.reshape(-1, d).T \
                    @ dVh.reshape(-1, d_h)
                dQh_t = dS_task @ Kh
                dKh_t = dS_task.swapaxes(1, 2) @ Qh
                dTn += dQh_t @ HS.Wq.T + dKh_t @ HS.Wk.T \
                    + dVh @ HS.Wv.T
            dT0, dg1, db1 = bk.ln_bwd(dTn, c1, self.P[f"g1_{l}"])
            G[f"g1_{l}"] += dg1
            G[f"b1n_{l}"] += db1
            dT = dT0 + dT1                          # residual skip
        # token embeddings
        Xarr = caches[0][1]
        if self.CAUSAL:
            G["Wv"] += bk.einsum("ntf,ntd->fd", Xarr, dT)
            G["Bf"][:Xarr.shape[1]] += dT.sum(0)
        else:
            G["Wv"] += bk.einsum("nf,nfd->fd", Xarr, dT)
            G["Bf"] += dT.sum(0)

        # TWO-PASS step (ALGO P4): host dict + per-head attributes
        old_heads = {k: bk.copy(getattr(self.heads[k[0]][k[1]],
                                        k[2]))
                     for k in Gh}
        old_w1 = {l: bk.copy(self.P[f"W1_{l}"])
                  for l in range(self.L)}
        self._step(G, Gh, sgd_lr)
        for l in range(self.L):
            dw = self.P[f"W1_{l}"] - old_w1[l]
            self._ema_dw[l] = 0.95 * self._ema_dw[l] + 0.05 * dw
            self._ema_adw[l] = 0.95 * self._ema_adw[l] \
                + 0.05 * bk.abs(dw)
        # u_h EMA update (AFTER the step, on realized updates;
        # per-head PER-MATRIX buffers — doc 32 DD-7; elementwise,
        # value-identical to the old flat [Wq, Wk, Wv, Wo] layout)
        for l, layer in enumerate(self.heads):
            for h, HS in enumerate(layer):
                for nm in _HMATS:
                    dpart = getattr(HS, nm) - old_heads[(l, h, nm)]
                    HS.mu[nm] = 0.95 * HS.mu[nm] + 0.05 * dpart
                    HS.nu[nm] = 0.95 * HS.nu[nm] \
                        + 0.05 * bk.abs(dpart)
        self._t_att += 1
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

    def _step(self, G, Gh, sgd_lr):
        scales = self._lr_scales_active()      # 59 R-2
        if sgd_lr is not None:
            old = dict(self.P) if scales else None
            for k in self.P:
                self.P[k] = self.P[k] - sgd_lr * G[k]
            for k, g in Gh.items():
                HS = self.heads[k[0]][k[1]]
                nm = getattr(HS, k[2]) - sgd_lr * g
                if scales:
                    from reference_net.net import blend_scaled
                    r = f"head:{k[0]}:{k[1]}"
                    if r in scales:
                        nm = blend_scaled(getattr(HS, k[2]),
                                          nm,
                                          float(scales[r]))
                setattr(HS, k[2], nm)
            if scales:
                self._blend_P(old, scales)
            return
        # Adam via the backend kernels (doc 32 9.2): the P dict
        # rides adam_step_dict_mul (the 0.001*G*G association this
        # host has always used — backend ULP doctrine); the head
        # matrices ride the list-form adam_step (same association).
        # BOTH passes see the SAME incremented t (one step).
        bk = self._bk
        keys = sorted(Gh)
        m_list = [self._adam_h[k][0] for k in keys]
        v_list = [self._adam_h[k][1] for k in keys]
        params = [getattr(self.heads[k[0]][k[1]], k[2])
                  for k in keys]
        grads = [Gh[k] for k in keys]
        old = dict(self.P) if scales else None
        t_dict = bk.adam_step_dict_mul(self.P, G, self._adam,
                                       self._t, self.lr)
        new_params, t_heads = bk.adam_step(m_list, v_list,
                                           self._t, self.lr,
                                           params, grads)
        assert t_dict == t_heads
        if scales:
            self._blend_P(old, scales)
            from reference_net.net import blend_scaled
        for i, k in enumerate(keys):
            self._adam_h[k] = [m_list[i], v_list[i]]
            npm = new_params[i]
            if scales:
                r = f"head:{k[0]}:{k[1]}"
                if r in scales:
                    npm = blend_scaled(params[i], npm,
                                       float(scales[r]))
            setattr(self.heads[k[0]][k[1]], k[2], npm)
        self._t = t_dict

    # ---------------- growth contract (FFN sites, ported from
    # transformer.py:394-449; the ATTENTION operators below use
    # their own driver — they do NOT ride this site-path grammar,
    # plan P0) ----------------
    def _unit_instability(self, l):
        # read-only instrument at adjudication rate (doc 32 9.2 I)
        num = np.linalg.norm(
            np.asarray(self._bk.to_numpy(self._ema_dw[l])), axis=0)
        den = np.linalg.norm(
            np.asarray(self._bk.to_numpy(self._ema_adw[l])),
            axis=0) + 1e-12
        return 1.0 - num / den

    def growth_sites(self):
        from reference_net.trainer import collect_instability
        sites = []
        grown = getattr(self, "_port_js", set())  # W3: port grows
        for l in range(self.L):
            inst = self._unit_instability(l)
            for j in range(self.m):
                if (l, j) not in self.inner \
                        and (l, j) not in grown:
                    sites.append((f"layer{l}/ffn[{j}]",
                                  float(inst[j])))
        deep_bodies = list(self.inner.items())
        for l, site in getattr(self, "_port_sites", {}).items():
            deep_bodies += [(s.get("key", (l, g)), s["body"])
                            for g, s in enumerate(site.bodies)]
        for (l, j), net in deep_bodies:
            for path, k, score, owner in collect_instability(net):
                if k not in owner.inner and k not in \
                        getattr(owner, "_port_js", set()):
                    sites.append((f"layer{l}/ffn[{j}]::{path}[{k}]",
                                  float(score)))
        return sorted(sites, key=lambda t: -t[1])

    # ---------- B2: whole-layer insertion (doc 55 s3-B2) ----------
    _LAYER_KEY_ROOTS = ("g1", "b1n", "g2", "b2n",
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
        """Whole-layer insertion at ANY index p in [0, L]
        (doc 55 s3-B2; survey 49 s2.2 function-preserving
        form): per-layer keys renumber l -> l+1 for l >= p
        (shared surgery, growth_port.shift_layer_maps); the
        new layer is born with LN identity and — under the
        default zero side — ZERO attention out-projections
        (every head Wo) and ZERO FFN second matrix, so with
        the residual stream the inserted layer contributes
        EXACTLY nothing at birth, at any p. recipe:
        "random" (fresh seeded draws) or "copy_layer" (tensors
        copied from the designated source layer; with
        zero_side="none" the copy is COMPLETE — SOLAR
        copy-splice, non-preserving by declared choice).
        Attention-HEAD methodology is untouched (C-4): this
        inserts a LAYER; head operators keep their own paths.
        Event recorded with verbatim specs (FR-7) in
        self.growth_events (these hosts carry no gain
        ledger — recorded design fact)."""
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
            src = nbr if nbr < p else nbr + 1   # post-shift index
            vals = {r: n_(self.P[f"{r}_{src}"]).copy()
                    for r in self._LAYER_KEY_ROOTS}
            # heads list is not shifted yet — source heads
            # by ORIGINAL index:
            new_heads = [_copy_headstate(hs)
                         for hs in self.heads[nbr]]
        else:
            vals = {"g1": np.ones(d), "b1n": np.zeros(d),
                    "g2": np.ones(d), "b2n": np.zeros(d),
                    "W1": rng.normal(0, np.sqrt(2.0 / d),
                                     (d, m)),
                    "b1": np.zeros(m),
                    "W2": rng.normal(0, np.sqrt(1.0 / m),
                                     (m, d)),
                    "b2": np.zeros(d)}
            geometry = [hs.d_h for hs in
                        self.heads[min(p, self.L - 1)]]                 if self.heads else []
            new_heads = [HeadState(d, dh, rng, birth_t=0)
                         for dh in geometry]
        def _zl(a):        # D-W6-2: family-dispatching
            return (np.zeros_like(a)
                    if isinstance(a, np.ndarray)
                    else bk.zeros_like(a))
        if zero_side == "default":
            vals["W2"] = np.zeros_like(vals["W2"])
            for hs in new_heads:
                hs.Wo = _zl(hs.Wo)
        for r in self._LAYER_KEY_ROOTS:
            self.P[f"{r}_{p}"] = bk.ingest(vals[r])
            self._adam[f"{r}_{p}"] = [
                bk.zeros_like(self.P[f"{r}_{p}"]),
                bk.zeros_like(self.P[f"{r}_{p}"])]
        for HS in new_heads:   # D-W6-2b: onto the device
            HS._bk = bk        # (the _ingest_state program;
            for nm in _HMATS:  # identity on the numpy judge)
                setattr(HS, nm, bk.ingest(getattr(HS, nm)))
            HS.mu = {k: bk.ingest(v) for k, v in HS.mu.items()}
            HS.nu = {k: bk.ingest(v) for k, v in HS.nu.items()}
        self.heads.insert(p, new_heads)
        for h_i, hs in enumerate(new_heads):
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                self._adam_h[(p, h_i, nm)] = [
                    _zl(getattr(hs, nm)),
                    _zl(getattr(hs, nm))]
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
                                      "heads": [hs.d_h for hs
                                                in new_heads]},
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
        # layer FFN sites create the reference inner BY DESIGN.
        from reference_net.net import Network
        if "::" in site_path:
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
            import time
            t0 = time.perf_counter()
            from reference_net.growthpolicy import \
                DEFAULT_GROWTH_POLICY as _gpp
            from reference_net.growth_store import auto_snapshot
            auto_snapshot(self, getattr(self, "_growth_policy",
                                        _gpp))   # FR-13 (D-6.2)
            l, j = self._parse(site_path)
            self._grow_ffn_port(l, j, hidden, site_path,
                                force=force)
            rec = self.growth_events[-1]         # D-6.1 stamp
            rec["wall_ms"] = (time.perf_counter() - t0) * 1e3
            if force:
                rec["forced"] = True             # C-5
        return {"grown": site_path, "params": self.n_params(),
                "depth": self.depth()}

    def _grow_ffn_port(self, l, j, hidden, site_path,
                       force=False):
        # ONE shared routing implementation (doc 35 R2); reference
        # body BY DESIGN at layer FFN sites (S9.5 convention)
        from reference_net.growth_port import grow_ffn_body
        grow_ffn_body(self, l, j, hidden, site_path,
                      force=force)

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

    # ---------------- attention growth operators (ALGO P6;
    # COMPUTE C2) — run ONLY at adjudication points; both
    # self-check exactness at apply time ----------------
    def _probe_batch(self):
        rng = np.random.default_rng(1234)
        if self.CAUSAL:
            return rng.normal(size=(3, min(5, self.WINDOW),
                                    self.d_in))
        return rng.normal(size=(4, self.d_in))

    def head_add(self, l, seed=None):
        """New head: generators SEEDED at host scale, W_o = 0 —
        output contribution identically zero (bitwise exact);
        trainable from step 1 on the zero side (two-step law)."""
        if seed is None:
            self._seed_counter += 1
            seed = self._seed_counter
        rng = np.random.default_rng(seed)
        bk = self._bk
        d_h = self._pol("att_birth_dh")
        probe = self._probe_batch()
        before = np.asarray(bk.to_numpy(
            self._forward(self._stdx(probe))))
        HS = HeadState(self.d, d_h, rng, birth_t=self._t_att)
        HS.Wo = np.zeros((d_h, self.d))            # zero side
        HS._bk = bk                                # onto the device
        for nm in _HMATS:
            setattr(HS, nm, bk.ingest(getattr(HS, nm)))
        HS.mu = {k: bk.ingest(v) for k, v in HS.mu.items()}
        HS.nu = {k: bk.ingest(v) for k, v in HS.nu.items()}
        self.heads[l].append(HS)
        self._sync_head_slots()
        after = np.asarray(bk.to_numpy(
            self._forward(self._stdx(probe))))
        assert np.array_equal(before, after), \
            "head_add broke function preservation (must be bitwise)"
        return {"event": "head_add", "layer": l,
                "head": len(self.heads[l]) - 1, "d_h": d_h}

    def head_widen(self, l, h, m=None, seed=None):
        """Widen head h by m columns: SEEDED W~q/W_v (W~q at the
        SAME birth normalizer — uniform absorbed scale), ZERO
        W_k columns / W_o rows (pairwise one-zero-factor: exact);
        all-zero new dims are a certified TOTAL SADDLE and are
        FORBIDDEN (asserted)."""
        if m is None:
            m = self._pol("att_widen_m")
        if seed is None:
            self._seed_counter += 1
            seed = self._seed_counter
        rng = np.random.default_rng(seed)
        bk = self._bk
        n_ = bk.to_numpy
        HS = self.heads[l][h]
        d = self.d
        sd = np.sqrt(1.0 / d)
        probe = self._probe_batch()
        before = np.asarray(n_(self._forward(self._stdx(probe))))
        q_new = rng.normal(0, sd, (d, m)) / np.sqrt(HS.d_h_birth)
        v_new = rng.normal(0, sd, (d, m))
        assert np.abs(q_new).max() > 0 and np.abs(v_new).max() > 0, \
            "all-zero widen forbidden: certified total saddle"
        # event round-trip surgery (doc 32 D3): numpy hstack/pad
        # exactly as before, then re-ingest onto the backend
        HS.Wq = bk.ingest(np.hstack([np.asarray(n_(HS.Wq)),
                                     q_new]))
        HS.Wv = bk.ingest(np.hstack([np.asarray(n_(HS.Wv)),
                                     v_new]))
        HS.Wk = bk.ingest(np.hstack([np.asarray(n_(HS.Wk)),
                                     np.zeros((d, m))]))  # zero side
        HS.Wo = bk.ingest(np.vstack([np.asarray(n_(HS.Wo)),
                                     np.zeros((m, d))]))  # zero side
        # optimizer slots + EMA buffers EXTEND zeroed
        for nm in _HMATS:
            key = (l, h, nm)
            M = getattr(HS, nm)
            m_, v_ = self._adam_h[key]
            m_np, v_np = np.asarray(n_(m_)), np.asarray(n_(v_))
            pad = [(0, M.shape[0] - m_np.shape[0]),
                   (0, M.shape[1] - m_np.shape[1])]
            self._adam_h[key] = [bk.ingest(np.pad(m_np, pad)),
                                 bk.ingest(np.pad(v_np, pad))]
            # mu/nu parts: POSITION-CORRECT extension (owner
            # ruling 2026-07-21, doc 34 5 defect FIXED): every
            # bookkeeping value stays attached to its parameter
            # coordinate — new columns/rows enter zeroed (nu at
            # its 1e-12 floor). Replaces the historical
            # ravel-tail rebuild, whose misalignment is proven
            # by tests/unit/test_ga_mu_alignment_defect.py.
            mu_np = np.asarray(n_(HS.mu[nm]))
            nu_np = np.asarray(n_(HS.nu[nm]))
            HS.mu[nm] = bk.ingest(np.pad(mu_np, pad))
            HS.nu[nm] = bk.ingest(
                np.pad(nu_np, pad, constant_values=1e-12))
        after = np.asarray(n_(self._forward(self._stdx(probe))))
        dev = np.abs(before - after).max()
        assert dev <= 1e-12, \
            f"head_widen broke preservation: {dev}"
        return {"event": "head_widen", "layer": l, "head": h,
                "m": m, "d_h": HS.d_h}

    # ---------------- evidence + decision (ALGO P7; COMPUTE
    # C2.3) — read-only statistics; decide() runs ONLY when the
    # caller's demand trigger fires (plateau pattern), never
    # free-running ----------------
    def u_stats(self, l):
        # DD-7: concatenate the per-matrix parts back into the old
        # flat [Wq, Wk, Wv, Wo] layout and take the SAME norms —
        # bitwise-identical numbers to the pre-port code
        n_ = self._bk.to_numpy
        out = []
        for HS in self.heads[l]:
            mu = np.concatenate([np.asarray(n_(HS.mu[nm])).ravel()
                                 for nm in _HMATS])
            nu = np.concatenate([np.asarray(n_(HS.nu[nm])).ravel()
                                 for nm in _HMATS])
            out.append(1.0 - np.linalg.norm(mu)
                       / (np.linalg.norm(nu) + 1e-12))
        return out

    def head_loading(self, l, X):
        """C4: L_h = ||(A_h V_h) W_o^h||_F per head, mean over the
        batch. Read-only."""
        Xs = self._stdx(np.asarray(X, float))
        _, _, caches = self._forward(Xs, cache=True)
        att = caches[1 + l][3]
        out = []
        for h, HS in enumerate(self.heads[l]):
            Oh = att[h][4]
            contrib = np.asarray(
                self._bk.to_numpy(Oh @ HS.Wo))    # (n, F, d)
            out.append(float(np.mean(
                [np.linalg.norm(contrib[b]) for b in
                 range(len(contrib))])))
        return out

    def age(self, l, h):
        return self._t_att - self.heads[l][h].birth_t

    def last_widen_failed(self, l, events):
        """Escalation memory (ALGO P7): True iff the most recent
        accepted head_widen row for layer l was followed by the
        SAME demand trigger firing again within att_window steps
        of that row's t (the caller passes the event ledger)."""
        rows = [e for e in events
                if e.get("event") == "head_widen"
                and e.get("layer") == l
                and e.get("verdict") == "accepted"]
        if not rows:
            return False
        last = rows[-1]
        return (self._t_att - last.get("t", -10**9)
                <= self._pol("att_window"))

    def decide(self, l, X, events=()):
        """ADD-vs-WIDEN suggestion (COMPUTE C2.3). Returns
        ("widen", h, m) | ("add",) | None. Newborns are excluded
        from BOTH sides of the localization test."""
        u = self.u_stats(l)
        H = len(self.heads[l])
        cand = [h for h in range(H)
                if self.age(l, h) >= self._pol("att_head_age_min")]
        if cand:
            u_c = [u[h] for h in cand]
            if max(u_c) >= self._pol("att_kappa") \
                    * (sum(u_c) / len(u_c)):
                best = cand[int(np.argmax(u_c))]
                return ("widen", best, self._pol("att_widen_m"))
        q = self.head_loading(l, X)
        PR = sum(q) ** 2 / (sum(v * v for v in q) + 1e-12)
        if H >= 2 and PR >= (1 - self._pol("att_beta")) * H:
            return ("add",)
        if H == 1:
            if not self.last_widen_failed(l, list(events)):
                return ("widen", 0, self._pol("att_widen_m"))
            return ("add",)
        return None

    # ---------------- instruments (ALGO P8; COMPUTE C4) — pure
    # read-only functions; zero state mutation (T10 hashes) -----
    def disposition_head(self, l, h):
        """D_h = W_o^h — the exact one-hop linear factor from the
        head's pre-projection output into the residual stream
        (w.r.t. the ENCLOSING block's output)."""
        return np.array(self._bk.to_numpy(self.heads[l][h].Wo),
                        copy=True)

    def disposition_unit(self, l, j):
        """FFN unit j: row j of THIS host's W2_l (m, d) — same
        quantity as C4's leaf D_j, different storage orientation."""
        return np.array(self._bk.to_numpy(self.P[f"W2_{l}"])[j, :],
                        copy=True)

    def row_entropies(self, l, h, X):
        Xs = self._stdx(np.asarray(X, float))
        _, _, caches = self._forward(Xs, cache=True)
        A = np.asarray(self._bk.to_numpy(caches[1 + l][3][h][3]))
        n, F, _ = A.shape
        F_i = (np.arange(1, F + 1) if self.CAUSAL
               else np.full(F, F))
        out = np.zeros((n, F))
        for b in range(n):
            for i in range(F):
                pp = A[b, i, :F_i[i]]
                pc = np.maximum(pp, 1e-300)
                out[b, i] = float(-(pp * np.log(pc)).sum())
        return out

    @staticmethod
    def capacity(q):
        """Concentration statistics over any unit family (C4):
        PR + entropy (xlogy convention). Nonzero loads -> PR in
        [1, #units]; the all-zero load yields 0 (eps-guarded)."""
        q = np.asarray(q, float)
        s = q.sum()
        pr = s * s / ((q * q).sum() + 1e-12)
        pp = q / (s + 1e-12)
        pc = np.maximum(pp, 1e-300)
        ent = float(-(pp * np.log(pc)).sum())
        return {"PR": float(pr), "entropy": ent}

    # ---------------- introspection ----------------
    def depth(self):
        kids = [n.depth() for n in self.inner.values()]
        for site in getattr(self, "_port_sites", {}).values():
            kids += [s["body"].depth() for s in site.bodies]
        return 1 + max(kids, default=0)

    def n_params(self):
        own = sum(int(np.prod(v.shape)) for v in self.P.values())
        own += sum(HS.n_params() for layer in self.heads
                   for HS in layer)
        own += sum(site.n_params() for site in
                   getattr(self, "_port_sites", {}).values())
        return own + sum(n.n_params() for n in self.inner.values())

    def shape_record(self):
        return {"mode": self.mode, "vocab": list(self.vocab),
                "depth": self.depth(), "params": self.n_params(),
                "d_in": self.d_in, "hidden": self.m,
                "substrate": self.NAME,
                "heads": [[HS.d_h for HS in layer]
                          for layer in self.heads]}   # per-head census

    def perturb(self, rng, sigma):
        import copy
        p = copy.deepcopy(self)
        bk = p._bk
        p.P["Wh"] = p.P["Wh"] + bk.ingest(
            rng.normal(0, sigma, tuple(p.P["Wh"].shape)))
        for l in range(p.L):
            p.P[f"W1_{l}"] = p.P[f"W1_{l}"] + bk.ingest(
                rng.normal(0, sigma, tuple(p.P[f"W1_{l}"].shape)))
        return p

    # ---------------- artifact ----------------
    # Backend-portable pickling (doc 32 D8, reference_net pattern):
    # tensors leave as numpy; loading ingests under the CURRENT
    # compute policy. Old (pre-port) pickles carry flat mu/nu and
    # no _bk — migrated on setstate.
    def __getstate__(self):
        st = dict(self.__dict__)
        bk = st.pop("_bk", None)
        st.pop("_masks", None)
        st.pop("_snapshots", None)   # observer state (I-7):
        st.pop("_monitor", None)     # never artifact content
        if bk is not None:
            def n_(x):
                return np.asarray(bk.to_numpy(x))
            st["P"] = {k: n_(v) for k, v in st["P"].items()}
            st["_adam"] = {k: [n_(m), n_(v)]
                           for k, (m, v) in st["_adam"].items()}
            st["_adam_h"] = {k: [n_(m), n_(v)]
                             for k, (m, v) in st["_adam_h"].items()}
            st["_ema_dw"] = {l: n_(v)
                             for l, v in st["_ema_dw"].items()}
            st["_ema_adw"] = {l: n_(v)
                              for l, v in st["_ema_adw"].items()}
            if st.get("_x_mu") is not None:
                st["_x_mu"] = n_(st["_x_mu"])
                st["_x_sd"] = n_(st["_x_sd"])
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        from engine.backends import current_backend
        self._bk = current_backend()
        self._masks = {}
        for layer in self.heads:
            for HS in layer:
                if not isinstance(HS.mu, dict):
                    # migrate the old flat [Wq, Wk, Wv, Wo] layout
                    flat_mu, flat_nu = HS.mu, HS.nu
                    mu, nu, ofs = {}, {}, 0
                    for nm in _HMATS:
                        M = np.asarray(getattr(HS, nm))
                        size = int(np.prod(M.shape))
                        mu[nm] = np.asarray(
                            flat_mu[ofs:ofs + size]).reshape(M.shape)
                        nu[nm] = np.asarray(
                            flat_nu[ofs:ofs + size]).reshape(M.shape)
                        ofs += size
                    HS.mu, HS.nu = mu, nu
        # D6 (doc 35): pre-reform grown artifacts (no port field)
        # load AS legacy_scalar — audit-visible marker only.
        if self.__dict__.get("inner") \
                and "_port_sites" not in self.__dict__ \
                and "_legacy_port" not in self.__dict__:
            from reference_net.growth_port import LegacyScalarPort
            self._legacy_port = LegacyScalarPort()
        self._ingest_state()

    # Security note: pickle for locally-produced artifacts inside the
    # system's own storage tree (same trust domain), as project-wide.
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


# -------------------- governance driver (ALGO P10; COMPUTE C6) --
# Mirrors the trainer.py grow_event pattern: checkpoint -> apply ->
# adjudicate -> rollback. Feeds the EXISTING adjudication idea (a
# held-out gate decides); builds no new adjudication machinery.
# The plateau-trigger CALLER is outside the lib (experiments /
# facade verb) — this driver never free-runs.

def attention_grow_event(sub, l, X_eval, heldout_X, heldout_y,
                         probe_X=None, probe_y=None,
                         events=None, tol=1.05, metric_fn=None):
    """One governed attention-growth event on layer l.

    sub        the GrowableAttentionSubstrate (mutated only on
               acceptance)
    X_eval     evidence batch for decide()
    heldout    quarantined data for the gate: SCORED ONLY, never
               trained on. The event is ACCEPTED only if the
               held-out gate metric after a probe-training epoch
               is within tol * before (a growth step is exact at
               application; the gate guards the training that
               follows).
    probe      TRAINING-LANE data for the probe epoch (doc 18,
               audit B1): the trial copy trains att_probe_steps
               steps on (probe_X, probe_y). Two-lane contract —
               the probe lane learns, the heldout lane judges;
               they must be disjoint batches. Growth proposed
               with probe_X=None is REFUSED loudly (a probe-less
               pass would rubber-stamp: exact application means
               before == after).
    events     the event ledger (list of dicts); the new row is
               appended and returned
    metric_fn  S9.2 D-G4: the gate's probe metric must match the
               model's mode. None (default) = held-out MSE on
               float targets — the pre-S9.2 path verbatim.
               Otherwise metric_fn(model, hx, hy) -> LOSS-LIKE
               scalar (lower is better; e.g. error rate for
               categorical); heldout_y and probe_y are then
               passed through unconverted (labels stay labels).
               The gate PRINCIPLE is unchanged either way:
               accept iff after <= tol * before.
    """
    import copy

    if events is None:
        events = []
    sug = sub.decide(l, X_eval, events=events)
    row = {"event": None, "t": sub._t_att, "layer": l,
           "evidence": {"u": [float(v) for v in sub.u_stats(l)],
                        "loadings": [float(v) for v in
                                     sub.head_loading(l, X_eval)]},
           "policy": {k: (list(v) if isinstance(v, (set,))
                          else v) for k, v in
                      getattr(sub, "_att_policy", POLICY).items()}}
    if sug is None:
        row.update({"event": "none", "verdict": "no_trigger"})
        events.append(row)
        return row

    if probe_X is None:
        row.update({"event": "none", "verdict": "refused",
                    "suggestion": sug[0],
                    "reason": "no probe data — the gate probe "
                              "trains on the training lane only, "
                              "never the scoring holdout"})
        events.append(row)
        return row

    hx = np.asarray(heldout_X, float)
    px = np.asarray(probe_X, float)
    if metric_fn is None:
        hy = np.asarray(heldout_y, float).reshape(-1, 1)
        py = np.asarray(probe_y, float).reshape(-1, 1).ravel()

        def gate_metric(model):
            pred = model.predict(hx)
            return float(((pred - hy) ** 2).mean())
    else:
        hy = np.asarray(heldout_y)
        py = np.asarray(probe_y)

        def gate_metric(model):
            return float(metric_fn(model, hx, hy))

    before = gate_metric(sub)
    trial = copy.deepcopy(sub)
    if sug[0] == "widen":
        ev = trial.head_widen(l, sug[1], m=sug[2])
    else:
        ev = trial.head_add(l)
    row["event"] = ev["event"]
    row["head"] = ev.get("head")
    # two-lane contract (doc 18): the probe epoch trains the trial
    # copy on TRAINING-LANE data only; the heldout batch above is
    # scored, never trained on
    pol = getattr(sub, "_att_policy", POLICY)
    for _ in range(int(pol.get("att_probe_steps",
                               POLICY["att_probe_steps"]))):
        trial.train_step(px, py)
    after = gate_metric(trial)
    row["heldout_before"] = before
    row["heldout_after"] = after
    if after <= tol * before:
        # ACCEPT: the trial copy becomes the serving model state
        sub.__dict__.update(trial.__dict__)
        row["verdict"] = "accepted"
    else:
        # REFUSE: sub was never touched — byte-identical by
        # construction, and T12 pins the behavior anyway
        row["verdict"] = "refused"
    events.append(row)
    return row
