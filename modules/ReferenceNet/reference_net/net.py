"""Recursive multi-scale network (the STUDENT's substrate).

One class, closed under refinement (PLAN Part I S2.1): a Network is a
1-hidden-layer MLP whose hidden NODES may each contain an inner Network
(composite nodes). Depth is data, never code structure.

Learning (Reading B, recursive; PLAN S2.2):
- within a scale: ordinary backprop on this scale's parameters, treating
  composite nodes' inner outputs as constants (pass-through gradient via
  the atomic branch only — v0 choice R5);
- across scales: the enclosing scale hands each composite node a TARGET
  for its output; the inner network takes its own local step toward it —
  and recurses identically for ITS composite nodes. One primitive, every
  depth, one step per enclosing iteration (synchronous single-step, R5).

Growth (S2.3): grow(j) attaches an inner network whose output layer is
ZERO-INITIALIZED — exact function preservation at the moment of growth.

numpy only. Deterministic under seed.
"""
from __future__ import annotations

from collections import deque

import numpy as np

ETA_TARGET = 0.5  # target step: t = h - ETA_TARGET * dL/dh  (PLAN 7.1)

# ---- 59 R-2: per-region lr scales, shared pieces (ONE
# implementation point for all three model classes) ----
import re as _re_scales

_SCALE_GRAMMAR = _re_scales.compile(
    r"^(encoder|readout|loop|embed|out|block:\d+|"
    r"layer:\d+|head:\d+:\d+)$")


def validate_lr_scales(scales):
    """UNION-grammar check (59 R-2.3b) — called BEFORE any
    state mutation; grammar-valid names not live on an
    instance are inert (S2 propagation safety)."""
    bad = [k for k in scales
           if not _SCALE_GRAMMAR.match(str(k))]
    if bad:
        raise ValueError(
            f"train_lr_scales: unknown region name(s) "
            f"{bad}; legal grammar: encoder|readout|loop|"
            f"embed|out|block:<k>|layer:<l>|head:<l>:<h>")


def blend_scaled(old, new, s):
    """59 D-2.2: s == 0 keeps the ORIGINAL object (bitwise
    stillness, P3-F2); else the exact lr x s identity."""
    return old if s == 0.0 else old + s * (new - old)



# gelu/gelu_d moved VERBATIM to backends/numpy_backend.py (B1 code
# motion); re-exported here so every existing importer is untouched.
from engine.backends import get_default_backend                # noqa: E402
from engine.primitives import gelu, gelu_d         # noqa: E402


class _Adam:
    """Optimizer STATE holder; the update math lives in the
    backend's K7 kernel (moved verbatim, B1)."""

    def __init__(self, shapes, lr, bk=None):
        self._bk = bk or get_default_backend()
        self.lr, self.t = lr, 0
        self.m = [self._bk.zeros(s) for s in shapes]
        self.v = [self._bk.zeros(s) for s in shapes]

    def step(self, params, grads):
        bk = getattr(self, "_bk", None) or get_default_backend()
        out, self.t = bk.adam_step(self.m, self.v, self.t,
                                   self.lr, params, grads)
        return out

    def __getstate__(self):
        bk = getattr(self, "_bk", None) or get_default_backend()
        st = self.__dict__.copy()
        st.pop("_bk", None)
        st.pop("_snapshots", None)    # runtime ring (FR-13):
        st.pop("_monitor", None)      # never artifact content
        st["m"] = [bk.to_numpy(a) for a in self.m]
        st["v"] = [bk.to_numpy(a) for a in self.v]
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_bk" not in self.__dict__:
            self._bk = get_default_backend()


class Network:
    """Recursively refinable 1-hidden-layer regressor: d_in -> H -> 1."""

    def __init__(self, d_in: int, hidden: int, lr: float = 1e-2,
                 seed: int = 7, zero_out: bool = False,
                 backend=None, out_width: int = 1):
        # init randomness is numpy-sourced on EVERY backend
        # (DESIGN_BACKEND 4); arrays are ingested to the device
        # out_width (Growth Interface Reform, doc 35 D2): the
        # output dimension; DEFAULT 1 keeps every pre-reform
        # construction bitwise identical. Vector-output bodies
        # (out_width > 1) serve the fullwidth growth port.
        if not isinstance(out_width, (int, np.integer)) \
                or isinstance(out_width, bool) or out_width < 1:
            raise ValueError(
                f"out_width must be an int >= 1; got {out_width!r}")
        from engine.backends import current_backend
        self._bk = backend or current_backend()
        rng = np.random.default_rng(seed)
        self.d_in, self.H, self.lr = d_in, hidden, lr
        self.out_width = int(out_width)
        k = self.out_width
        self.W1 = self._bk.ingest(
            rng.normal(0, np.sqrt(2.0 / d_in), (hidden, d_in)))
        self.b1 = self._bk.ingest(np.zeros(hidden))
        self.W2 = self._bk.ingest(
            np.zeros((k, hidden)) if zero_out
            else rng.normal(0, np.sqrt(1.0 / hidden), (k, hidden)))
        self.c = self._bk.ingest(np.zeros(k))
        self.opt = _Adam([(hidden, d_in), (hidden,), (k, hidden),
                          (k,)], lr, self._bk)
        self.inner: dict[int, Network] = {}          # composite nodes
        # standardization (fit at first train_step; zero_out nets keep
        # y_mu=0 so a freshly grown inner net predicts EXACTLY 0)
        self._zero_out = zero_out
        self._x_mu = self._x_sd = self._y_mu = self._y_sd = None
        # per-node instability stats (owner's oscillation signal u_j)
        self._ema_dw = np.zeros((hidden, d_in))      # EMA of signed updates
        self._ema_adw = np.zeros((hidden, d_in)) + 1e-12   # EMA of |updates|
        self._seed_counter = seed
        # ---- instrumentation (DEV_PLAN S1): observation only ----
        self._step_count = 0
        self._E = None                       # EMA of standardized MSE
        self.energy_beta = 0.95              # policy
        self.gain_horizon = 200              # policy (W)
        self.energy_ring = deque(maxlen=2048)
        self.window_ring = deque(maxlen=512)  # (x_row, y_row) as received
        self.gain_ledger = []                # growth-event records
        self._pending_gain = []              # ledger indices awaiting E_after
        self._cos_series = deque(maxlen=32)  # (update_norm, cos_prev)
        self._prev_dw = None
        self.blocks = []                     # composition blocks (delta)
        self.loop_block = None               # lambda block (loop)
        self._loop_k_last = None             # training-path stats only
        self._loop_k_ema = None
        self._loop_projections = 0

    # ---------- forward ----------
    def _hidden(self, X):
        A, Hact = self._bk.dense_forward(self.W1, self.b1, X)
        # attach through the ONE shared port module (reform,
        # doc 35 R2): legacy structures via the moved v1 code,
        # fullwidth bodies via the port (no-op when absent)
        from .growth_port import legacy_attach
        Hact = legacy_attach(Hact, X, self.inner)
        port = getattr(self, "_port_site", None)
        if port is not None:
            Hact = port.forward(Hact, X)
        return A, Hact

    # ---------- instrumentation (S1): observation only ----------
    def residual_energy(self):
        """EMA-smoothed standardized mean-squared error of THIS scope
        (root: vs its targets; inner: vs the residual stream handed to
        it). None before the first training step."""
        return self._E

    def _observe_step(self, X_in, y_in, mse):
        self._step_count += 1
        if self._E is None:
            self._E = mse
        else:
            b = self.energy_beta
            self._E = b * self._E + (1.0 - b) * mse
        self.energy_ring.append(self._E)
        for xi, yi in zip(X_in, y_in):
            self.window_ring.append((np.array(xi, copy=True),
                                     np.array(yi, copy=True)))
        still = []
        for idx in self._pending_gain:
            evt = self.gain_ledger[idx]
            if self._step_count >= evt["due"]:
                evt["E_after"] = self._E
                eb = evt["E_before"]
                evt["gain"] = (None if (eb is None or eb <= 0.0)
                               else (eb - self._E) / eb)
            else:
                still.append(idx)
        self._pending_gain = still

    def _record_update_direction(self, dw):
        flat = np.asarray(dw, dtype=float).ravel()
        norm = float(np.linalg.norm(flat))
        cos = 0.0
        if (self._prev_dw is not None
                and self._prev_dw.shape == flat.shape):
            # a shape change (omega widening / sigma features) is a new
            # geometry: no cosine across the edit, series continues fresh
            pn = float(np.linalg.norm(self._prev_dw))
            if norm > 0.0 and pn > 0.0:
                cos = float(flat @ self._prev_dw / (norm * pn))
        self._cos_series.append((norm, cos))
        self._prev_dw = flat.copy()

    def _ledger_event(self, event, site, params_added):
        rec = {"event": event, "site": site,
               "params_added": int(params_added),
               "E_before": self._E, "E_after": None, "gain": None,
               "step": self._step_count,
               "due": self._step_count + self.gain_horizon}
        self.gain_ledger.append(rec)
        self._pending_gain.append(len(self.gain_ledger) - 1)
        return rec

    def _cost_fields(self, rec, t0):
        """FR-16 OPTION B (58 D-1): explicit cost columns on
        every growth/removal record. step_at_event anchors the
        chain INSIDE the ledger (I-8: zero new state keys — the
        anchor serializes with the ledger and reverts with it
        on rollback, D-1.3b); steps_since_prev attributes the
        training compute to the preceding configuration;
        wall_ms is THIS operation's own duration (run-varying;
        the ledger sits outside all bit gates, SR-22)."""
        import time
        prev = 0
        for r in reversed(self.gain_ledger):
            if r is not rec and "step_at_event" in r:
                prev = r["step_at_event"]
                break
        rec["step_at_event"] = int(self._step_count)
        rec["steps_since_prev"] = int(self._step_count - prev)
        rec["wall_ms"] = (time.perf_counter() - t0) * 1e3

    def __getstate__(self):
        # artifacts are DEVICE-FREE (DESIGN_BACKEND 3): arrays to
        # numpy, backend handle dropped (serving loads the judge)
        bk = self._bk
        st = self.__dict__.copy()
        st.pop("_bk", None)
        st.pop("_snapshots", None)    # runtime ring (FR-13):
        st.pop("_monitor", None)      # never artifact content
        for k in ("W1", "b1", "W2", "c", "_x_mu", "_x_sd"):
            if st.get(k) is not None and not isinstance(
                    st[k], (int, float)):
                st[k] = bk.to_numpy(st[k])
        st["blocks"] = [{kk: bk.to_numpy(vv) for kk, vv in
                         b.items()} for b in self.blocks]
        if self.loop_block is not None:
            st["loop_block"] = {kk: bk.to_numpy(vv) for kk, vv
                                in self.loop_block.items()}
        return st

    def __setstate__(self, state):
        # backward compatibility: artifacts pickled before S1 lack the
        # instrumentation fields; fill defaults on load.
        self.__dict__.update(state)
        if "_bk" not in self.__dict__:      # device-free artifacts
            self._bk = get_default_backend()
        d = self.__dict__
        d.setdefault("_step_count", 0)
        d.setdefault("_E", None)
        d.setdefault("energy_beta", 0.95)
        d.setdefault("gain_horizon", 200)
        d.setdefault("energy_ring", deque(maxlen=2048))
        d.setdefault("window_ring", deque(maxlen=512))
        d.setdefault("gain_ledger", [])
        d.setdefault("_pending_gain", [])
        d.setdefault("_cos_series", deque(maxlen=32))
        d.setdefault("_prev_dw", None)
        d.setdefault("blocks", [])
        d.setdefault("loop_block", None)
        d.setdefault("_loop_k_last", None)
        d.setdefault("_loop_k_ema", None)
        d.setdefault("_loop_projections", 0)
        # D6 (doc 35): a pre-reform artifact carrying grown bodies
        # but no port field loads AS legacy_scalar — named so audits
        # see the state; serving stays on the verbatim legacy path.
        if d.get("inner") and "_port_site" not in d \
                and "_legacy_port" not in d:
            from .growth_port import LegacyScalarPort
            d["_legacy_port"] = LegacyScalarPort()
        # A4 (doc 52 s7): chain structures under the contract —
        # dict blocks from ANY artifact vintage wrap into the
        # storage-owning structures VALUE-PRESERVING (the
        # kernels keep reading the same tensors through the
        # historical keys; on-disk format stays plain dicts).
        from .bodies import Block, LoopBlock
        self.blocks = [b if isinstance(b, Block) else
                       Block(b["Bin"], b["bb"], b["Bout"],
                             bk=self._bk)
                       for b in d.get("blocks", [])]
        if d.get("loop_block") is not None and not isinstance(
                d["loop_block"], LoopBlock):
            lb = d["loop_block"]
            self.loop_block = LoopBlock(
                lb["L_in"], lb["b_l"], lb["L_out"],
                bk=self._bk)

    def _ingest_state(self):
        """G7 (doc 61 I-A): move every learned tensor onto
        self._bk — trunk, composition blocks, loop block,
        optimizer slots, x-scalers, and the port site's
        couplings/bodies recursively. Identity on the numpy
        judge. Faithful to LIVE construction (probed): the
        EMA instruments (_ema_dw/_ema_adw) STAY numpy (K9
        edge doctrine); _y_mu/_y_sd are python scalars."""
        bk = self._bk
        ing = bk.ingest
        self.W1, self.b1 = ing(self.W1), ing(self.b1)
        self.W2, self.c = ing(self.W2), ing(self.c)
        for blk in self.blocks:
            if hasattr(blk, "_bk"):
                blk._bk = bk
            blk["Bin"] = ing(blk["Bin"])
            blk["bb"] = ing(blk["bb"])
            blk["Bout"] = ing(blk["Bout"])
        if self.loop_block is not None:
            lb = self.loop_block
            if hasattr(lb, "_bk"):
                lb._bk = bk
            lb["L_in"] = ing(lb["L_in"])
            lb["b_l"] = ing(lb["b_l"])
            lb["L_out"] = ing(lb["L_out"])
        opt = getattr(self, "opt", None)
        if opt is not None:
            opt._bk = bk
            opt.m = [ing(a) for a in opt.m]
            opt.v = [ing(a) for a in opt.v]
        if self._x_mu is not None:
            self._x_mu = ing(self._x_mu)
            self._x_sd = ing(self._x_sd)
        site = getattr(self, "_port_site", None)
        if site is not None and hasattr(site, "ingest_to"):
            site.ingest_to(bk)
        for body in getattr(self, "inner", {}).values():
            body._bk = bk
            if hasattr(body, "_ingest_state"):
                body._ingest_state()

    def _std_x(self, X):
        if self._x_mu is None:
            return X
        return self._bk.standardize(X, self._x_mu, self._x_sd)

    def _opt_shapes(self):
        shapes = [self.W1.shape, self.b1.shape,
                  self.W2.shape, self.c.shape]
        for blk in self.blocks:
            shapes += [blk["Bin"].shape, blk["bb"].shape,
                       blk["Bout"].shape]
        if self.loop_block is not None:
            lb = self.loop_block
            shapes += [lb["L_in"].shape, lb["b_l"].shape,
                       lb["L_out"].shape]
        return shapes

    def _rebuild_opt(self):
        self.opt = _Adam(self._opt_shapes(), self.lr, self._bk)

    # ---- 59 R-2: per-region learning-rate scales (ADVANCED,
    # off by default — key absent means these are never
    # called; guard by ABSENCE, never by arithmetic) ----

    def _opt_group_regions(self):
        """Region name per optimizer group, aligned with
        _opt_shapes order (verified: 4 trunk groups, then 3
        per block, then 3 loop)."""
        regs = ["encoder", "encoder", "readout", "readout"]
        for k in range(len(self.blocks)):
            regs += [f"block:{k}"] * 3
        if self.loop_block is not None:
            regs += ["loop"] * 3
        return regs

    def _validate_lr_scales(self, scales):
        validate_lr_scales(scales)      # module-level single
        #                                 implementation point

    def _apply_lr_scales(self, params, new):
        """Blend the update per region (59 D-2.2): the scale
        multiplies the UPDATE (exactly lr x s); s == 0 keeps
        the ORIGINAL object (bitwise stillness, P3-F2).
        Table names are validated against the UNION grammar
        (R-2.3b): typos refuse loudly; grammar-valid names
        not live on this instance are inert (the policy dict
        propagates into grown bodies, S2)."""
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _d
        scales = getattr(self, "_growth_policy", _d)             .get("train_lr_scales")
        if not scales:
            return new
        self._validate_lr_scales(scales)
        regs = self._opt_group_regions()
        out = []
        for p_, n_, r_ in zip(params, new, regs):
            if r_ in scales:
                out.append(blend_scaled(p_, n_,
                                        float(scales[r_])))
            else:
                out.append(n_)
        return out

    def _apply_blocks(self, H0, cache=False):
        """Composition chain (delta blocks) — K2 kernel."""
        return self._bk.block_chain_forward(self.blocks, H0,
                                            cache=cache)

    def predict(self, X):
        X = self._bk.ingest(X)
        if self._y_mu is None and self._zero_out:
            return self._bk.zeros((len(X),
                                   getattr(self, "out_width", 1)))
        Xs = self._std_x(X)
        _, Hact = self._hidden(Xs)
        # readability: a scope without composition blocks never even
        # calls the block chain (pure-widen mode and any zero-block
        # scope take the straight path)
        HL = self._apply_blocks(Hact) if self.blocks else Hact
        if self.loop_block is not None:
            # K11 relaxation; k_used DISCARDED here — serving
            # purity: predict records nothing (red-line test)
            from .growthpolicy import DEFAULT_GROWTH_POLICY as _gp
            _gp = getattr(self, "_growth_policy", _gp)  # S9.5 D-N1
            HL, _, _ = self._bk.loop_forward(
                HL, self.loop_block["L_in"], self.loop_block["b_l"],
                self.loop_block["L_out"],
                float(_gp.get("loop_tol", 1e-6)),
                int(_gp.get("loop_K_max", 32)))
        raw = self._bk.readout(HL, self.W2, self.c)
        if self._y_mu is None:
            return raw
        return raw * self._y_sd + self._y_mu

    # ---------- one coupled training step (recursive) ----------
    def _grads(self, Xs, ys):
        """Raw within-scale gradients on ALREADY-standardized data.
        Returns (grads, aux); grads in Adam order. Body moved
        VERBATIM to the K3 kernel (B1); the composite-inner
        constants rule and the 2/n-in-lr convention unchanged."""
        A, H0 = self._hidden(Xs)
        if self.loop_block is not None:
            from .growthpolicy import DEFAULT_GROWTH_POLICY as _gp
            _gp = getattr(self, "_growth_policy", _gp)  # S9.5 D-N1
            grads, tail = self._bk.scope_backward(
                self.W1, self.b1, self.W2, self.c, self.blocks,
                Xs, ys, A, H0, loop_block=self.loop_block,
                loop_tol=float(_gp.get("loop_tol", 1e-6)),
                loop_K_max=int(_gp.get("loop_K_max", 32)))
        else:
            grads, tail = self._bk.scope_backward(
                self.W1, self.b1, self.W2, self.c, self.blocks,
                Xs, ys, A, H0)
        return grads, {"A": A, "H0": H0, "dH0": tail["dH0"],
                       "mse": tail["mse"],
                       "loop_k": tail.get("loop_k")}

    def train_step(self, X, y, sgd_lr=None):
        """One step toward targets y (shape (n,1)). Returns MSE before step.
        sgd_lr: if given, use plain SGD at that rate instead of Adam —
        required for consolidation-style updates near zero error, where
        Adam\'s stale-moment ratio m/sqrt(v) walks blind O(lr) steps and
        destroys the function (measured)."""
        _sc = getattr(self, "_growth_policy",
                      {}).get("train_lr_scales")
        if _sc:
            self._validate_lr_scales(_sc)   # 59 R-2.3b: refuse
        #                                     BEFORE any mutation
        X = self._bk.ingest(X)               # device edge first:
        y = self._bk.ingest(y)               # the walk sees device
        # SPU integration seam (DESIGN_SPU_UNIT_HOOK v2.0): if this
        # scope HOLDS an spu policy, run the pre-forward walk over its
        # subtree BEFORE the released body — process -> participate ->
        # learn at every depth. Costs one getattr when no policy is
        # installed; children never hold the policy, so no re-walk.
        _pol = getattr(self, "_spu_policy", None)
        if (_pol is not None and _pol["spu_enabled"]
                and self._x_mu is not None):
            from .spu.spu_network import spu_pre_forward  # lazy
            spu_pre_forward(self, X, y, _pol)
        n = len(X)
        if self._x_mu is None:                      # fit scalers once
            self._x_mu, self._x_sd = self._bk.fit_x_stats(X)
            ym, ys_ = self._bk.y_stats(y)
            self._y_mu = 0.0 if self._zero_out else ym
            self._y_sd = ys_
        else:
            # self-shaping scaler refresh (curriculum widens the target
            # range): rescale y-standardization AND compensate W2/c with
            # the exact inverse affine map -- function preserved.
            mb, sb = self._bk.y_stats(y)
            if sb > 1.5 * self._y_sd or abs(mb - self._y_mu) > 2 * self._y_sd:
                mu_n = 0.0 if self._zero_out else mb
                sd_n = max(sb, self._y_sd)
                self.W2, self.c = self._bk.refresh_readout(
                    self.W2, self.c, self._y_mu, self._y_sd,
                    mu_n, sd_n)
                self._y_mu, self._y_sd = mu_n, sd_n
                # Adam moments are in the OLD scale; stale second moments
                # take oversized steps on the rescaled weights and destroy
                # the function (measured). Reset optimizer state (rare op).
                self._rebuild_opt()
        X_in = self._bk.to_numpy(X)          # as received (S1 ring;
        y_in = self._bk.to_numpy(y)          # rings stay numpy)
        X = self._std_x(X)
        y = (y - self._y_mu) / self._y_sd
        grads, aux = self._grads(X, y)
        A, Hact = aux["A"], aux["H0"]
        dH0, mse = aux["dH0"], aux["mse"]
        old_W1 = self._bk.copy(self.W1)
        params = [self.W1, self.b1, self.W2, self.c]
        for blk in self.blocks:
            params += [blk["Bin"], blk["bb"], blk["Bout"]]
        if self.loop_block is not None:
            lb = self.loop_block
            params += [lb["L_in"], lb["b_l"], lb["L_out"]]
        if sgd_lr is None:
            new = self.opt.step(params, grads)
        else:
            new = self._bk.sgd_step(params, grads, sgd_lr)
        if getattr(self, "_growth_policy",
                   {}).get("train_lr_scales"):   # 59 R-2
            new = self._apply_lr_scales(params, new)
        self.W1, self.b1, self.W2, self.c = new[:4]
        i = 4
        for blk in self.blocks:
            blk["Bin"], blk["bb"], blk["Bout"] = new[i:i + 3]
            i += 3
        if self.loop_block is not None:
            lb = self.loop_block
            lb["L_in"], lb["b_l"], lb["L_out"] = new[i:i + 3]
            i += 3
            # certificate enforcement (DESIGN §4): post-step scalar
            # projection restores rho_hat <= cap exactly; audited
            from engine.loop_ops import loop_rho_hat
            from .growthpolicy import DEFAULT_GROWTH_POLICY as _gp3
            _gp3 = getattr(self, "_growth_policy", _gp3)  # S9.5 D-N1
            rho_cap = float(_gp3.get("loop_rho_max", 0.6))
            rho = loop_rho_hat(self._bk.to_numpy(lb["L_in"]),
                               self._bk.to_numpy(lb["L_out"]))
            if rho > rho_cap:
                lb["L_out"] = lb["L_out"] * (rho_cap / rho)
                self._loop_projections += 1
            # k_used stats: TRAINING PATH ONLY (serving purity)
            lk = aux.get("loop_k")
            if lk is not None:
                self._loop_k_last = int(lk)
                self._loop_k_ema = (float(lk) if self._loop_k_ema
                                    is None else 0.95 *
                                    self._loop_k_ema + 0.05 * lk)
        # instability stats from actual updates
        # instability instrumentation stays numpy (signals are the
        # CPU world; K9 edge — identity on the judge)
        dw = self._bk.to_numpy(self.W1 - old_W1)
        self._ema_dw = 0.95 * self._ema_dw + 0.05 * dw
        self._ema_adw = 0.95 * self._ema_adw + 0.05 * np.abs(dw)
        # instrumentation (S1): observation only
        self._record_update_direction(dw)
        self._observe_step(X_in, y_in, mse)
        # across scales: hand each composite node a target; recurse
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gpe
        _gpe = getattr(self, "_growth_policy", _gpe)  # S9.5 D-N1
        _eta = float(_gpe.get("eta_target", ETA_TARGET))
        from .growth_port import legacy_handoff
        legacy_handoff(self.inner, X, Hact, A, dH0,
                       self._bk.gelu, _eta, sgd_lr=sgd_lr)
        port = getattr(self, "_port_site", None)
        if port is not None:
            # TRUE gradient scale (pin P-2: dU = the raw chain-rule
            # gradient): this kernel's documented objective divides
            # param grads by n at the end; the port receives dH on
            # the SAME scale so A_g/body updates are consistent
            # with the host's own (Part B T-14/T-14b adjudicated)
            port.backward_step(dH0 / max(len(X), 1), X, self.lr,
                               sgd_lr=sgd_lr)
        from .instrument import monitor_tick
        monitor_tick(self)   # FR-16 (no-op unless armed)
        return mse

    def train_from_grad(self, X, dU, sgd_lr=None):
        """Growth-port training entry (doc 35 D4, plan B —
        EXACT): one step against the RAW output gradient dU
        (shape (n, out_width)), with no target fitting and no
        eta. Mechanism: the backward kernel computes
        err = pred - ys internally and scales every gradient
        by 1/n, so feeding ys = pred - n*dU makes err = n*dU
        and every returned gradient EXACTLY dU^T-chained
        (audited: numpy_backend.scope_backward). pred is
        recomputed here through the SAME kernels the backward
        uses, so the subtraction is bitwise-consistent.
        Intended for PORT-OWNED bodies (identity scalers);
        refuses otherwise. Recursion into legacy inner nodes
        follows the same handoff as train_step."""
        _sc = getattr(self, "_growth_policy",
                      {}).get("train_lr_scales")
        if _sc:
            self._validate_lr_scales(_sc)   # 59 R-2.3b
        if not getattr(self, "_port_owned", False):
            raise ValueError(
                "train_from_grad serves port-owned bodies "
                "(identity scalers); construct via "
                "growth_port.make_port_body")
        bk = self._bk
        X = bk.ingest(X)
        dU = bk.ingest(dU)
        n = len(X)
        Xs = self._std_x(X)                # identity (port-owned)
        A, H0 = self._hidden(Xs)
        HL = self._apply_blocks(H0) if self.blocks else H0
        pred = bk.readout(HL, self.W2, self.c)
        ys = pred - float(n) * dU          # err := n*dU exactly
        grads, aux = self._grads(Xs, ys)
        A_, Hact = aux["A"], aux["H0"]
        dH0 = aux["dH0"]
        old_W1 = bk.copy(self.W1)
        params = [self.W1, self.b1, self.W2, self.c]
        for blk in self.blocks:
            params += [blk["Bin"], blk["bb"], blk["Bout"]]
        if self.loop_block is not None:
            lb = self.loop_block
            params += [lb["L_in"], lb["b_l"], lb["L_out"]]
        if sgd_lr is None:
            new = self.opt.step(params, grads)
        else:
            new = bk.sgd_step(params, grads, sgd_lr)
        if getattr(self, "_growth_policy",
                   {}).get("train_lr_scales"):   # 59 R-2
            new = self._apply_lr_scales(params, new)
        self.W1, self.b1, self.W2, self.c = new[:4]
        i = 4
        for blk in self.blocks:
            blk["Bin"], blk["bb"], blk["Bout"] = new[i:i + 3]
            i += 3
        if self.loop_block is not None:
            lb = self.loop_block
            lb["L_in"], lb["b_l"], lb["L_out"] = new[i:i + 3]
            i += 3
        dw = bk.to_numpy(self.W1 - old_W1)
        self._ema_dw = 0.95 * self._ema_dw + 0.05 * dw
        self._ema_adw = 0.95 * self._ema_adw + 0.05 * np.abs(dw)
        self._record_update_direction(dw)
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gpe
        _gpe = getattr(self, "_growth_policy", _gpe)
        _eta = float(_gpe.get("eta_target", ETA_TARGET))
        from .growth_port import legacy_handoff
        legacy_handoff(self.inner, Xs, Hact, A_, dH0,
                       bk.gelu, _eta, sgd_lr=sgd_lr)
        port = getattr(self, "_port_site", None)
        if port is not None:
            # same TRUE-scale convention as train_step (P-2)
            port.backward_step(dH0 / max(n, 1), Xs, self.lr,
                               sgd_lr=sgd_lr)
        # instrumentation (OBSERVATION ONLY — no training-path
        # change): the gradient step's Reading-B target-equivalent
        # stream is t = pred - eta*dU (the exact target this step
        # chases, same eta convention as the legacy handoff), so
        # the standing energy/window/gain instruments and the
        # growth adjudicator keep serving gradient-trained scopes.
        # Disclosed in-batch addition (doc 38).
        t_eq = bk.to_numpy(pred - _eta * dU)
        mse_eq = float(np.mean((_eta * bk.to_numpy(dU)) ** 2))
        self._observe_step(np.asarray(bk.to_numpy(Xs)), t_eq,
                           mse_eq)
        return float(aux["mse"])

    # ---------- instability (owner's signal) ----------
    def instability(self):
        """u_j in [0,1]: 1 = steady drift, 0 = oscillation. Returns 1-u
        (higher = more unstable) per node."""
        num = np.linalg.norm(self._ema_dw, axis=1)
        den = np.linalg.norm(self._ema_adw, axis=1) + 1e-12
        return 1.0 - num / den

    # ---------- growth support (A2 preset seams, doc 52 s5/6.1) ----------
    def _scale_guard(self, body, j):
        """L2 scale-hierarchy guard at the post-build/pre-couple
        seam (doc 52 SR-13) — A5: VERBATIM motion to
        method/gates.py; this method resolves the policy and
        delegates (hosts own policy resolution, gates never
        import it)."""
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gp2
        _gp2 = getattr(self, "_growth_policy", _gp2)  # S9.5 D-N1
        from .method.gates import scale_hierarchy_guard
        scale_hierarchy_guard(self, body, j, _gp2)

    def _ensure_port_site(self):
        """The host's shared coupling site, created on first
        growth — VERBATIM code motion from the historical grow
        body."""
        from .growth_port import PortSite
        port = getattr(self, "_port_site", None)
        if port is None:
            port = PortSite(self.H, backend=self._bk)
            self._port_site = port
        return port

    # ---------- growth (single refinement operator, any depth) ----------
    def grow(self, j: int, hidden: int = 16, body_type=None,
             force=False):
        """rho: growth triggered at node j (j = provenance/site
        key). Growth Interface Reform (doc 35, doc 36 W3): the new
        body is a FULLWIDTH port citizen — vector output u_g
        (grow_body_out_width), zero-born trainable assembly A_g
        across the whole hidden width, exact chain-rule training.
        `body_type` overrides the policy's grow_body_type;
        `hidden` parameterizes the REFERENCE body only (attention
        sizes come from the grow_attention_* policy keys).
        legacy_scalar is load-only and REFUSED here."""
        import time
        t0 = time.perf_counter()
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gpp
        _gpp = getattr(self, "_growth_policy", _gpp)  # S9.5 D-N1
        from .growth_store import auto_snapshot
        auto_snapshot(self, _gpp)     # FR-13: pre-event capture
        from . import growth_port as _gp  # noqa: F401 (A2/AL-6:
        # importing the host-tier module registers the structure-
        # kind builders into the foundation socket)
        from .method.presets import preset_rho
        body = preset_rho(self, j, hidden, body_type, _gpp,
                          force=force)
        rec = self.gain_ledger[-1]
        self._cost_fields(rec, t0)    # FR-16
        if force:                     # C-5 (58 D-8): the
            rec["forced"] = True      # override is RECORDED
        return body

    def grown_body(self, j):
        """The body grown at node j — fullwidth (port slot keyed
        by j) or legacy (loaded artifact inner dict). None if node
        j never grew."""
        if j in self.inner:
            return self.inner[j]
        port = getattr(self, "_port_site", None)
        return port.body_by_key(j) if port is not None else None

    def remove_grown(self, j, force=False):
        """Remove the body grown at node j (removability contract;
        the port analog of the legacy `del net.inner[j]`).
        FR-18: shrink governance — pre-event snapshot
        (policy-gated, default ON), reversible in-window."""
        import time
        t0 = time.perf_counter()
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gps
        _gps = getattr(self, "_growth_policy", _gps)
        from .growth_store import auto_snapshot
        auto_snapshot(self, _gps)
        if j in self.inner:
            n = self.inner[j].n_params()
            del self.inner[j]
            rec = self._ledger_event("remove_grown", j, -n)
            rec.setdefault("trigger", "caller")   # FR-18/R11
            self._cost_fields(rec, t0)            # FR-16
            if force:                             # C-5
                rec["forced"] = True
            return
        removed = self._port_site.remove_by_key(j)
        self._port_js.discard(j)
        # R11 (55 B5 REMOVAL LEDGERING): negative mirror of the
        # grow event — Coupling.n_params() = assembly A + body,
        # exactly the grow record's positive accounting.
        rec = self._ledger_event("remove_grown", j,
                                 -removed.n_params())
        rec.setdefault("trigger", "caller")
        self._cost_fields(rec, t0)                # FR-16
        if force:                                 # C-5
            rec["forced"] = True

    def deepen(self, m: int | None = None, position=None,
               recipe=None, recipe_params=None, scope=None,
               zero_side=None, force=False):
        """Append one ZERO-INITIALIZED residual composition block to
        this scope (the delta operator). Exact at application: Bout
        is zeros, so the block contributes nothing until trained.
        Serial depth inside the scope grows by one; nothing outside
        the scope is touched."""
        import time
        t0 = time.perf_counter()
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gpd
        _gpd = getattr(self, "_growth_policy", _gpd)
        # G-ASPECT (60D): PRE-snapshot, pre-mutation — a
        # refusal must not leave a junk auto-snapshot. ONE seam
        # covers the default AND the preset_layer paths; scoped
        # deepen is judged on the GLOBAL shape (D-2 ruling).
        from .method.gates import gate_aspect
        gate_aspect(_gpd, self.H, 1 + len(self.blocks) + 1,
                    force=force)
        from .growth_store import auto_snapshot
        auto_snapshot(self, _gpd)     # FR-13: pre-event capture
        from . import growth_port as _gp  # noqa: F401 (builders)
        from . import bodies as _bodies    # noqa: F401 (A4:
        # registers the "block"/"loop" chain-structure builders)
        if (position is None and recipe is None
                and recipe_params is None and scope is None
                and zero_side is None):
            # DEFAULT CALL: the historical delta path, bitwise
            # (B1 carrier rule, doc 55 s3 B1)
            from .method.presets import preset_delta
            out = preset_delta(self, m)
        else:
            from .method.presets import preset_layer
            out = preset_layer(self, m, position=position,
                               recipe=recipe,
                               recipe_params=recipe_params,
                               scope=scope, zero_side=zero_side,
                               gpp=_gpd, force=force)
        rec = self.gain_ledger[-1]
        self._cost_fields(rec, t0)    # FR-16
        if force:                     # C-5 (58 D-8)
            rec["forced"] = True
        return out

    def remove_block(self, k: int, force=False):
        """Remove composition block k (audited structural edit).
        FR-18: shrink stands under the growth governance — a
        pre-event snapshot is taken (policy-gated, default ON)
        so the removal is reversible within the window."""
        import time
        t0 = time.perf_counter()
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gpr
        _gpr = getattr(self, "_growth_policy", _gpr)
        from .growth_store import auto_snapshot
        auto_snapshot(self, _gpr)
        blk = self.blocks.pop(k)
        m = blk["bb"].size
        self._rebuild_opt()
        rec = self._ledger_event("prune_block", "scope",
                                 -(m * self.H + m + self.H * m))
        rec.setdefault("trigger", "caller")   # FR-18/R11
        self._cost_fields(rec, t0)            # FR-16
        if force:                             # C-5
            rec["forced"] = True
        return k

    def loop(self, m: int | None = None, force=False):
        """lambda: grow the governed directed cycle at the END of
        this scope's chain (DESIGN_LOOP_V2 v2.3). Exact at
        application: L_out is ZEROS, so z* == H_L bitwise until
        trained. Opt-in: refuses unless loop_enabled."""
        import time
        t0 = time.perf_counter()
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gp
        from engine.loop_ops import validate_loop_policy
        _gp = getattr(self, "_growth_policy", _gp)   # S9.5 D-N1
        eff = validate_loop_policy(_gp)          # loud, incl. switch
        if self.loop_block is not None:
            raise ValueError("this scope already has a loop block "
                             "(budget is ONE per scope): "
                             "remove_loop first")
        m = eff["loop_m"] if m is None else int(m)
        if not (1 <= m <= 256):
            raise ValueError(f"loop m must be in [1, 256], got {m}")
        # timing guard: a cycle is a topology change — it comes
        # after the base has taken shape (the grow() ruling verbatim)
        min_steps = int(_gp.get("grow_min_host_steps", 100))
        if self._step_count < min_steps:
            msg = (f"scale-hierarchy violation: host trained only "
                   f"{self._step_count} steps < {min_steps} "
                   f"(grow_min_host_steps): topology changes must "
                   f"come after the base has taken shape, never "
                   f"from the start")
            if _gp.get("grow_scale_guard", "warn") == "refuse":
                raise ValueError(msg)
            import warnings
            warnings.warn(msg + " (recorded)", stacklevel=2)
            if not hasattr(self, "_scale_events"):
                self._scale_events = []
            self._scale_events.append(
                {"site": "loop", "step": self._step_count,
                 "problems": [msg]})
        # surgery per 2a: numpy creation, ingest to the device;
        # init family matches the released delta Bin convention
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gpl
        _gpl = getattr(self, "_growth_policy", _gpl)
        from .growth_store import auto_snapshot
        auto_snapshot(self, _gpl)     # FR-13: pre-event capture
        from . import growth_port as _gp  # noqa: F401 (builders)
        from . import bodies as _bodies    # noqa: F401 (A4)
        from .method.presets import preset_loop
        out = preset_loop(self, m)
        rec = self.gain_ledger[-1]
        self._cost_fields(rec, t0)    # FR-16
        if force:                     # C-5 (58 D-8)
            rec["forced"] = True
        return out

    def remove_loop(self, force=False):
        """Remove the lambda block (audited structural edit).
        Bitwise-restoring when the block was never trained past
        L_out = 0. FR-18: shrink governance — pre-event
        snapshot (policy-gated, default ON)."""
        import time
        t0 = time.perf_counter()
        from .growthpolicy import DEFAULT_GROWTH_POLICY as _gps
        _gps = getattr(self, "_growth_policy", _gps)
        from .growth_store import auto_snapshot
        auto_snapshot(self, _gps)
        if self.loop_block is None:
            raise ValueError("no loop block on this scope")
        lb = self.loop_block
        m = int(self._bk.numel(lb["b_l"]))
        self.loop_block = None
        self._loop_k_last = None
        self._loop_k_ema = None
        self._rebuild_opt()
        rec = self._ledger_event("remove_loop", "scope",
                                 -(2 * m * self.H + m))
        rec.setdefault("trigger", "caller")   # FR-18/R11
        self._cost_fields(rec, t0)            # FR-16
        if force:                             # C-5
            rec["forced"] = True
        return m

    def layer_depth(self):
        """Layer depth of this scope's own body: 1 (the base hidden
        layer) + the number of composition blocks. Distinct from
        tree_height(), the height of the inclusion tree."""
        return 1 + len(self.blocks)

    serial_depth = layer_depth   # back-compat alias (pre-rename)

    # ---------- introspection ----------
    def n_params(self):
        nel = self._bk.numel
        own = (nel(self.W1) + nel(self.b1) + nel(self.W2)
               + nel(self.c))
        own += sum(nel(b["Bin"]) + nel(b["bb"]) + nel(b["Bout"])
                   for b in self.blocks)
        if self.loop_block is not None:
            own += sum(nel(v) for v in self.loop_block.values())
        total = own + sum(n.n_params() for n in self.inner.values())
        port = getattr(self, "_port_site", None)
        if port is not None:
            total += port.n_params()     # bodies + assemblies A_g
        return total

    def depth(self):
        """Height of the inclusion tree (rooted-tree sense): how deep
        the network-inside-a-node nesting goes. Reports and figures
        label this quantity "tree height". Distinct from
        layer_depth(), one scope's composed-stage count."""
        port = getattr(self, "_port_site", None)
        kids = [n.depth() for n in self.inner.values()]
        if port is not None:
            kids += [s["body"].depth() for s in port.bodies]
        return 1 + max(kids, default=0)

    def structure(self, path=""):
        port = getattr(self, "_port_site", None)
        grown_js = sorted(getattr(self, "_port_js", set()))
        rows = [{"path": path or "root", "H": self.H,
                 "composite": sorted(self.inner) + grown_js,
                 "blocks": len(self.blocks),
                 "loop": self.loop_block is not None}]
        for j, net in self.inner.items():
            rows += net.structure(f"{path}/{j}" if path else str(j))
        if port is not None:
            base = path or ""
            for g, s in enumerate(port.bodies):
                rows += s["body"].structure(
                    f"{base}/port[{g}]" if base else f"port[{g}]")
        return rows


Network.tree_height = Network.depth   # authoritative-name alias
