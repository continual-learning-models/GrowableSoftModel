"""Growth Port — the ONE shared coupling between grown
structures and their hosts (Growth Interface Reform,
docs/system/35 v1.3).

Normative equations (doc 35 3):
    PORT-FWD    H' = H + sum_g  u_g(X_b) @ A_g
    PORT-BWD-A  dA_g = u_g(X_b)^T @ dH
    PORT-BWD-U  dU_g = dH @ A_g^T
with H (N x C) the host's attach-site activations (positions
folded into rows), X_b the host-designated body input, u_g
the body's raw vector output (N x k_g), and A_g (k_g x C) a
trainable assembly matrix born ZERO (exact function
preservation — the standing zero-side doctrine).

Port types (registry; replaceable-parts doctrine):
  fullwidth      — the reform. Default for all NEW growth.
  legacy_scalar  — DEPRECATED-DEFECTIVE, load-only: exists
                   solely so pre-reform artifacts remain
                   readable. REFUSED for new growth and never
                   a verification reference (owner ruling).

Training: bodies receive the TRUE task gradient (PORT-BWD-U)
via Network.train_from_grad; A_g steps with Adam in the
family's mul association (doc 35 D4b) or plain SGD when the
host trains with sgd_lr.
"""
from __future__ import annotations

import numpy as np

PORT_TYPES = ("fullwidth", "legacy_scalar")


def make_port_body(d_in, hidden, out_width, lr, seed,
                   backend=None, body_type="reference",
                   policy=None):
    """Construct a PORT-OWNED body of any registered body type
    (doc 36 W2(e)): born LIVE (u != 0 — the coupling's ONE zero
    side is the assembly A_g, doc 35 R4 / the head_add zero-side
    doctrine; a zero body AND a zero assembly would deadlock the
    chain-rule gradients dA = u^T dH, dU = dH A^T), identity
    scalers on both faces (doc 35 pin P-1 — the assembly owns
    output scaling), gradient-entry training enabled."""
    from .bodies import make_body
    body = make_body(body_type, policy, d_in, hidden, lr, seed,
                     backend=backend, out_width=int(out_width),
                     zero_out=False)
    bk = body._bk
    body._x_mu = bk.ingest(np.zeros(d_in))
    body._x_sd = bk.ingest(np.ones(d_in))
    body._y_mu, body._y_sd = 0.0, 1.0
    body._port_owned = True
    return body


# ---------- structure-kind builders (A2, doc 52 AL-6) ----------
# foundation holds only the SOCKET (compose.BUILDERS); this
# host-tier module — the owner of make_port_body — plugs the
# port-body builders in at import time. bodies.py remains THE
# single kind registry underneath (make_port_body delegates to
# bodies.make_body); no second registry exists.
from .foundation.compose import BUILDERS as _BUILDERS  # noqa: E402


def _port_body_builder(host, spec):
    policy = getattr(host, "_growth_policy", None)
    if policy is None:
        from .growthpolicy import DEFAULT_GROWTH_POLICY as policy
    return make_port_body(spec.params["d_in"],
                          spec.params["hidden"],
                          spec.params["out_width"], spec.lr,
                          spec.seed, backend=host._bk,
                          body_type=spec.kind, policy=policy)


for _kind in ("reference", "attention"):
    _BUILDERS.setdefault(_kind, _port_body_builder)


def shift_layer_maps(host, p, key_roots):
    """B2 (doc 55 s3-B2a): the ONE implementation point of the
    per-layer RENUMBERING surgery (doc 35 R2 doctrine) — every
    layer-indexed container shifts l -> l+1 for l >= p, in
    DESCENDING order (no collisions):
      - P dict keys  root_{l}  (key_roots per host)
      - _adam slots for the same keys
      - _adam_h keys (l, h, name)        (GA head slots)
      - per-layer instrument dicts keyed by int l
      - _port_sites {l: site}, inner {(l, j)}, _port_js {(l, j)}
    Grown FFN port bodies KEEP their identity — only the layer
    coordinate of their key shifts (B-2-2 boxes)."""
    L = host.L
    for l in range(L - 1, p - 1, -1):
        for root in key_roots:
            old, new = f"{root}_{l}", f"{root}_{l + 1}"
            if old in host.P:
                host.P[new] = host.P.pop(old)
            if old in getattr(host, "_adam", {}):
                host._adam[new] = host._adam.pop(old)
    if hasattr(host, "_adam_h"):
        host._adam_h = {
            ((k[0] + 1,) + tuple(k[1:]) if k[0] >= p else k): v
            for k, v in host._adam_h.items()}
    for attr in ("_ema_dw", "_ema_adw"):
        d = getattr(host, attr, None)
        if isinstance(d, dict):
            setattr(host, attr,
                    {(l + 1 if isinstance(l, int) and l >= p
                      else l): v for l, v in d.items()})
    if hasattr(host, "_port_sites"):
        host._port_sites = {
            (l + 1 if l >= p else l): site
            for l, site in host._port_sites.items()}
    if getattr(host, "inner", None):
        host.inner = {
            ((k[0] + 1, k[1]) if isinstance(k, tuple)
             and k[0] >= p else k): v
            for k, v in host.inner.items()}
    if hasattr(host, "_port_js"):
        host._port_js = {
            ((k[0] + 1, k[1]) if isinstance(k, tuple)
             and k[0] >= p else k) for k in host._port_js}


def layer_census(host, key_roots):
    """Audit helper (RB-1): the LIVE key walk — every
    layer-indexed name currently present. Generated from the
    real containers, never a hand list."""
    out = {"P": sorted(k for k in host.P
                       if any(k.startswith(f"{r}_")
                              for r in key_roots)),
           "adam": sorted(k for k in getattr(host, "_adam", {})
                          if any(k.startswith(f"{r}_")
                                 for r in key_roots)),
           "adam_h": sorted(map(str,
                                getattr(host, "_adam_h", {}))),
           "port_sites": sorted(getattr(host, "_port_sites",
                                        {})),
           "port_js": sorted(map(str,
                                 getattr(host, "_port_js",
                                         set())))}
    return out


def grow_ffn_body(host, l, j, hidden, site_path,
                  force=False):
    """Shared FFN-site growth routing for the per-layer 3-D hosts
    (growable_attention / transformer) — doc 36 W3(a), one
    implementation point (doc 35 R2). Creates a FULLWIDTH port
    body: input = the post-LN FFN input rows (the hosts' X_b
    designation), output assembled across the whole FFN hidden
    width m by a zero-born trainable A_g. legacy_scalar is
    load-only and REFUSED for new growth."""
    from .growthpolicy import DEFAULT_GROWTH_POLICY as _gp
    _gp = getattr(host, "_growth_policy", _gp)
    from .method.presets import preset_site

    def _site():
        if not hasattr(host, "_port_sites"):
            host._port_sites = {}
        site = host._port_sites.get(l)
        if site is None:
            site = PortSite(host.m, backend=host._bk)
            host._port_sites[l] = site
        return site

    return preset_site(host, l, j, hidden, _gp, _site,
                       site_path, force=force)


# ---------- legacy quarantine (A5, doc 52 s6.4) ----------
# VERBATIM code motion to legacy_compat.py; re-exported here so
# every existing import path stays valid (facade, doc 52 s6.3).
from .legacy_compat import (          # noqa: E402, F401
    LegacyScalarPort, legacy_attach, legacy_attach_layer,
    legacy_collect_layer, legacy_handoff)


class PortSite:
    """All grown bodies attached at ONE host site (one attach
    activation H of channel width C). The host calls forward()
    inside its forward pass and backward_step() inside its
    training step with dH = dL/dH taken pre-host-step."""

    PORT_TYPE = "fullwidth"

    def __init__(self, channels, backend=None):
        from engine.backends import current_backend
        self._bk = backend or current_backend()
        self.C = int(channels)
        self.bodies = []          # [Coupling] (slot-mapping compatible)
        self._t = 0               # SITE-level Adam counter (52 SR-20)
        self._u_cache = None

    # ---------- growth ----------
    def add_body(self, body, port_type="fullwidth", key=None,
                 span_in=None, span_out=None):
        """key: the host's site provenance (node j / (layer, j))
        so callers can address the body via host.grown_body().

        A1 (doc 52 v1.7 section 3.2/6.3): the slot is now a
        foundation.Coupling — same zero-born construction, same
        mapping surface (slot["A"] etc. keep working)."""
        if port_type != "fullwidth":
            raise ValueError(
                f"port type {port_type!r} is refused for new "
                "growth: legacy_scalar is a load-only artifact-"
                "compat path (deprecated-defective, doc 35 R3); "
                "new growth uses 'fullwidth'")
        if not getattr(body, "_port_owned", False):
            raise ValueError("bodies must be constructed via "
                             "make_port_body (port ownership)")
        from .foundation.coupling import Coupling
        c = Coupling.born_zero(
            body, body.out_width, self.C, key, self._bk)
        if span_in is not None or span_out is not None:
            # A3 span honoring: A reshaped to
            # (|span_in| x |span_out|) per 52 SR-2
            k = len(span_in) if span_in is not None \
                else int(body.out_width)
            w = len(span_out) if span_out is not None else self.C
            c.A = self._bk.ingest(np.zeros((k, w)))
            c.mA = self._bk.zeros_like(c.A)
            c.vA = self._bk.zeros_like(c.A)
            c.edw = self._bk.zeros_like(c.A)
            c.eadw = self._bk.zeros_like(c.A) + 1e-12
            c.span_in = span_in
            c.span_out = span_out
        self.bodies.append(c)

    def body_by_key(self, key):
        for s in self.bodies:
            if s.get("key") == key:
                return s["body"]
        return None

    def remove_by_key(self, key):
        """Remove the body grown at `key` (removability is part of
        the body contract). Returns the removed slot; raises if the
        key is unknown."""
        for i, s in enumerate(self.bodies):
            if s.get("key") == key:
                return self.bodies.pop(i)
        raise KeyError(f"no port body with key {key!r}")

    # ---------- forward (PORT-FWD) ----------
    def forward(self, H, X_b):
        """Returns H + sum_g u_g(X_b) @ A_g. With no bodies the
        INPUT OBJECT is returned unchanged (structural no-op:
        zero arithmetic on the fixed-net path — doc 37 T-4)."""
        if not self.bodies:
            return H
        self._u_cache = []
        out = H
        for slot in self.bodies:
            u = slot.source(X_b)               # raw (n, k)
            self._u_cache.append(u)
            c = slot.contribution(u)
            if slot.span_out is None:
                out = out + c
            else:                     # A3: span_out honoring —
                pad = self._bk.ingest(         # scatter columns
                    np.zeros((np.asarray(c).shape[0], self.C)))
                pad[:, slot.span_out] = c
                out = out + pad
        return out

    # ---------- training coupling (PORT-BWD) ----------
    def backward_step(self, dH, X_b, lr, sgd_lr=None):
        """dH = dL/dH' (N x C), pre-host-step (pin P-4). Steps
        every A_g (Adam mul association / SGD) and hands every
        body its exact chain-rule gradient dU_g.

        A1 orchestration exactness (52 SR-21): the per-slot
        sequence — dA from cached u; dU from PRE-update A;
        counter incremented per slot (SITE-level, SR-20);
        A stepped; instability EMAs of the applied update;
        body trained LAST — is the historical order verbatim."""
        if not self.bodies:
            return
        bk = self._bk
        dH = bk.ingest(dH)
        for slot, u in zip(self.bodies, self._u_cache):
            dA, dU = slot.gradients(dH, u)
            if sgd_lr is None:
                self._t += 1
            slot.step(dA, lr, sgd_lr, self._t)
            slot.body.train_from_grad(X_b, dU, sgd_lr=sgd_lr)
        self._u_cache = None

    # ---------- bookkeeping ----------
    def n_params(self):
        return sum(s.n_params() for s in self.bodies)

    def loading(self, X_b):
        """||u_g @ A_g||_F per body over a batch (read-only
        instrument, the head_loading analog)."""
        n_ = self._bk.to_numpy
        out = []
        for s in self.bodies:
            contrib = np.asarray(
                n_(s["body"].predict(X_b) @ s["A"]))
            out.append(float(np.linalg.norm(contrib)))
        return out

    def instability(self):
        """Per-body assembly instability in [0, 1] from the
        update EMAs (doc 35 D5) — the Network.instability
        convention: 1 - ||EMA(dw)||_F / ||EMA(|dw|)||_F
        (0 = steady drift, 1 = pure oscillation)."""
        n_ = self._bk.to_numpy
        out = []
        for s in self.bodies:
            num = float(np.linalg.norm(np.asarray(n_(s["edw"]))))
            den = float(np.linalg.norm(
                np.asarray(n_(s["eadw"])))) + 1e-12
            out.append(1.0 - num / den)
        return out

    # ---------- serialization (backend-portable) ----------
    def __getstate__(self):
        """On-disk format UNCHANGED (A1, doc 52 section 6.3):
        couplings serialize as the HISTORICAL device-free slot
        dicts, so pre-adjustment artifacts load here and
        artifacts written here load in pre-adjustment code."""
        st = dict(self.__dict__)
        bk = st.pop("_bk", None)
        st.pop("_u_cache", None)
        if bk is not None:
            st["bodies"] = [s.to_slot_dict(bk.to_numpy)
                            for s in st["bodies"]]
        return st

    def __setstate__(self, st):
        from .foundation.coupling import Coupling
        self.__dict__.update(st)
        # G7 (doc 61 I-A, SV-6c): ONE artifact doctrine —
        # unpickling lands on the JUDGE like the owning
        # Network's __setstate__ (the historical
        # current_backend() here produced MIXED graphs under
        # an active torch policy: numpy body u @ torch A ->
        # TypeError). The live backend is preserved only via
        # the in-session _ingest_state hook (rollback) or the
        # owner class's own load path.
        from engine.backends import get_default_backend
        self._bk = get_default_backend()
        self._u_cache = None
        bk = self._bk
        self.bodies = [Coupling.from_slot_dict(s, bk)
                       for s in self.bodies]

    def ingest_to(self, bk):
        """G7: move the whole site — couplings' tensors and
        grown BODIES (recursively) — onto backend bk; caches
        cleared. Identity on the judge."""
        self._bk = bk
        self._u_cache = None
        for c in self.bodies:
            c._bk = bk
            c.A = bk.ingest(c.A)
            c.mA = bk.ingest(c.mA)
            c.vA = bk.ingest(c.vA)
            c.edw = bk.ingest(c.edw)
            c.eadw = bk.ingest(c.eadw)
            body = c.body
            if body is not None:
                body._bk = bk
                if hasattr(body, "_ingest_state"):
                    body._ingest_state()
