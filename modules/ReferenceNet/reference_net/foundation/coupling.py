"""P-2, the COUPLING PRIMITIVE (next-dev-docs/52 v1.7
section 3.2) — the second of the two L1 root primitives.

One Coupling = one zero-born trainable link between a
designated source value and a designated target value:

    contribution = u[:, span_in] @ A   scattered to span_out
    dA = u^T dH ;  dU = dH @ A^T       (span-restricted)

This is a VERBATIM extraction of the verified slot mathematics
of growth_port.PortSite (growthport-v1, growth_port.py
155-316): with the default full spans, every floating-point
operation and its order are IDENTICAL to the historical slot
code — the A1 bitwise gates adjudicate that, not this
docstring.

State ownership (52 SR-20, bit-critical): the Adam
bias-correction counter t is SITE-level, incremented once per
coupling per backward call by the counter's OWNER (the
PortSite view / host). A Coupling never owns t — a per-link
counter would silently drift every multi-body site.

Compatibility (52 SR-1): Coupling speaks the historical slot
MAPPING protocol — c["A"], c.get("key"), c["mA"] = ... — so
every existing walker, test and instrument runs unmodified.

Neutrality (PR-0): no notion of layer, organ, depth, width,
position or time lives here; the only numeric constants are
the family optimizer-kernel constants already resident in the
extracted code (Adam 0.9/0.999/1e-8 with the 0.1/0.001
mul-association; instability EMAs 0.95/0.05), listed in the
PR-4 constants ledger.
"""
import numpy as np

_FIELDS = ("body", "A", "key", "mA", "vA", "edw", "eadw")


class IdentitySource:
    """Neutral value source for COUPLING-ONLY events
    (StructureSpec kind="none", 52 s4: "skip = one more
    coupling"): predict returns its input verbatim — the
    coupling connects an EXISTING designated value to the
    target through the zero-born trainable A. No interior
    parameters; trivially contract-conformant."""

    _port_owned = True      # scaler-free by construction

    def __init__(self, width):
        self.out_width = int(width)

    def predict(self, X):
        return X

    def n_params(self):
        return 0

    def train_from_grad(self, X, dU, sgd_lr=None):
        return None

    def __getstate__(self):
        return {"out_width": self.out_width}

    def __setstate__(self, st):
        self.out_width = st["out_width"]


class Coupling:
    """One zero-born trainable link (slot semantics preserved).

    span_in / span_out are OPTIONAL index sets on the source /
    target faces (52 SR-2: A's shape is |span_in| x |span_out|,
    reducing to (k x C) under the default full spans). The
    default path is a STRUCTURAL no-op: no masking arithmetic
    executes (span attributes are None)."""

    def __init__(self, body, A, key=None, bk=None,
                 span_in=None, span_out=None,
                 mA=None, vA=None, edw=None, eadw=None):
        self.body = body
        self.A = A
        self.key = key
        self._bk = bk
        self.span_in = span_in
        self.span_out = span_out
        self.mA = mA if mA is not None else bk.zeros_like(A)
        self.vA = vA if vA is not None else bk.zeros_like(A)
        self.edw = edw if edw is not None else bk.zeros_like(A)
        self.eadw = eadw if eadw is not None \
            else bk.zeros_like(A) + 1e-12

    # ---------- construction ----------
    @classmethod
    def born_zero(cls, body, out_width, channels, key, bk):
        """The historical add_body birth: A zero-born at
        (out_width x channels), fresh optimizer/EMA slots."""
        A = bk.ingest(np.zeros((int(out_width), int(channels))))
        return cls(body, A, key=key, bk=bk)

    # ---------- forward ----------
    def source(self, X_b):
        """u = body.predict(X_b), span_in-restricted if set."""
        u = self.body.predict(X_b)
        if self.span_in is not None:
            u = u[:, self.span_in]
        return u

    def contribution(self, u):
        """u @ A (the caller supplies the cached u; scattering
        to span_out, when set, is the caller's placement of the
        returned columns)."""
        return u @ self.A

    # ---------- backward ----------
    def gradients(self, dH, u):
        """(dA, dU) from the exact chain rule; dU uses the
        PRE-update A (52 SR-21 orchestration exactness)."""
        if self.span_out is not None:
            dH = dH[:, self.span_out]
        dA = u.swapaxes(0, 1) @ dH if u.ndim == 2 \
            else np.asarray(u).T @ dH          # (k, C)
        dU = dH @ self.A.swapaxes(0, 1)        # (n, k)
        if self.span_in is not None:
            # scatter back to the body's own output coordinates
            # (unused outputs receive exact zeros)
            full = self._bk.ingest(np.zeros(
                (np.asarray(dU).shape[0],
                 int(self.body.out_width))))
            full[:, self.span_in] = dU
            dU = full
        return dA, dU

    def step(self, dA, lr, sgd_lr, t):
        """Apply one update to A; returns the applied update.
        t is SUPPLIED by the counter's owner (SR-20). The Adam
        mul-association (0.1 = 1-0.9; 0.001 = 1-0.999) and the
        instability EMAs of the APPLIED update reproduce the
        slot code bitwise."""
        if sgd_lr is not None:
            upd = sgd_lr * dA
        else:
            m_, v_ = self.mA, self.vA
            m_[:] = 0.9 * m_ + 0.1 * dA
            v_[:] = 0.999 * v_ + 0.001 * dA * dA
            b1c = 1 - 0.9 ** t
            b2c = 1 - 0.999 ** t
            upd = lr * (m_ / b1c) / ((v_ / b2c) ** 0.5 + 1e-8)
        self.A = self.A - upd
        self.edw = 0.95 * self.edw + 0.05 * upd
        self.eadw = 0.95 * self.eadw + 0.05 * abs(upd)
        return upd

    # ---------- bookkeeping ----------
    def n_params(self):
        return int(np.prod(self.A.shape)) + self.body.n_params()

    # ---------- historical slot mapping protocol (SR-1) ----------
    def __getitem__(self, k):
        if k in _FIELDS:
            return getattr(self, k)
        raise KeyError(k)

    def get(self, k, default=None):
        return getattr(self, k) if k in _FIELDS else default

    def __setitem__(self, k, v):
        if k not in _FIELDS:
            raise KeyError(k)
        setattr(self, k, v)

    def __contains__(self, k):
        return k in _FIELDS

    def keys(self):
        return _FIELDS

    # ---------- serialization ----------
    def to_slot_dict(self, to_numpy):
        """Device-free HISTORICAL slot-dict form — the on-disk
        format is unchanged (old artifacts load here; artifacts
        written here load in pre-adjustment code)."""
        n_ = to_numpy
        out = {"body": self.body, "A": np.asarray(n_(self.A)),
               "key": self.key,
               "mA": np.asarray(n_(self.mA)),
               "vA": np.asarray(n_(self.vA)),
               "edw": np.asarray(n_(self.edw)),
               "eadw": np.asarray(n_(self.eadw))}
        # ADDITIVE keys, present only when spans are set — a
        # default coupling's on-disk bytes are unchanged
        if self.span_in is not None:
            out["span_in"] = np.asarray(self.span_in)
        if self.span_out is not None:
            out["span_out"] = np.asarray(self.span_out)
        return out

    @classmethod
    def from_slot_dict(cls, s, bk):
        return cls(s["body"], bk.ingest(s["A"]),
                   key=s.get("key"), bk=bk,
                   span_in=s.get("span_in"),
                   span_out=s.get("span_out"),
                   mA=bk.ingest(s["mA"]), vA=bk.ingest(s["vA"]),
                   edw=bk.ingest(s["edw"]),
                   eadw=bk.ingest(s["eadw"]))
