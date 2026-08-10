"""Body registry — what kind of inner network rho grows
(DESIGN_GROW_BODY_TYPE v1.2, T2).

Additive-only registry; "reference" reproduces the EXACT
constructor call net.py's grow() has always made, so the default
world is bitwise-identical. The seam in grow() resolves through
here (lazy import on the net.py side)."""
from .attention_body import AttentionBody
from . import net as _net

BODY_TYPES = ("reference", "attention")


def make_body(body_type, policy, d_in, hidden, lr, seed,
              backend=None, out_width: int = 1, zero_out=True):
    """Construct an inner body of the selected type. `hidden`
    parameterizes the reference body ONLY (ignored for attention
    bodies — their sizes come from the grow_attention_* policy
    keys). out_width (doc 35 D2, W2(e)): readout width; the
    default 1 is the pre-reform shape (bitwise for old callers).
    zero_out=True is the legacy eta-handoff convention; PORT
    bodies pass False — they are born LIVE (u != 0) because the
    zero side of the coupling is the assembly A_g (doc 35 R4,
    the head_add zero-side doctrine; double-zero would deadlock
    the chain-rule gradients)."""
    if body_type == "reference":
        return _net.Network(d_in, hidden, lr=lr, seed=seed,
                            zero_out=zero_out, backend=backend,
                            out_width=int(out_width))
    if body_type == "attention":
        pol = policy or {}
        dm = int(pol.get("grow_attention_d_model", 8))
        nl = int(pol.get("grow_attention_layers", 1))
        nh = int(pol.get("grow_attention_heads", 2))
        fn = int(pol.get("grow_attention_ffn", 16))
        problems = []
        if not 2 <= dm <= 64:
            problems.append(f"grow_attention_d_model {dm} not in 2..64")
        if not 1 <= nl <= 4:
            problems.append(f"grow_attention_layers {nl} not in 1..4")
        if not 1 <= nh <= 8:
            problems.append(f"grow_attention_heads {nh} not in 1..8")
        if dm % max(nh, 1) != 0:
            problems.append(f"grow_attention_heads {nh} must divide "
                            f"d_model {dm}")
        if not 2 <= fn <= 256:
            problems.append(f"grow_attention_ffn {fn} not in 2..256")
        if problems:
            raise ValueError("attention body policy refused: "
                             + "; ".join(problems))
        return AttentionBody(d_in, lr=lr, seed=seed,
                             zero_out=zero_out,
                             d_model=dm, n_layers=nl, n_heads=nh,
                             ffn=fn, backend=backend,
                             out_width=int(out_width))
    raise ValueError(f"unknown grow_body_type {body_type!r}; "
                     f"valid: {BODY_TYPES}")


# ---------- A4: chain structures under the contract ----------
# (doc 52 v1.7 s7, SR-14 HOT-PATH RULE: the chain mathematics
# STAYS in the fused backend kernels — block_chain_forward and
# the scope backward — which keep operating on this SAME
# storage through the historical mapping keys. The standalone
# contract methods exist for conformance and are judged by
# their own boxes; scope-integrated training remains the
# kernels' job, verified by the replay gates.)
import numpy as _np
from .foundation.compose import register_builder as _register


class _ChainStruct:
    """Storage-owning structure speaking the historical dict
    protocol for its tensor keys (SR-5): every kernel, walker,
    optimizer-shape builder and serializer runs unmodified."""

    _KEYS = ()

    def __getitem__(self, k):
        if k in self._KEYS:
            return getattr(self, k)
        raise KeyError(k)

    def __setitem__(self, k, v):
        if k not in self._KEYS:
            raise KeyError(k)
        setattr(self, k, v)

    def get(self, k, default=None):
        return getattr(self, k) if k in self._KEYS else default

    def __contains__(self, k):
        return k in self._KEYS

    def __iter__(self):
        return iter(self._KEYS)     # dict semantics: iterate keys

    def keys(self):
        return self._KEYS

    def items(self):
        return [(k, getattr(self, k)) for k in self._KEYS]

    def values(self):
        return [getattr(self, k) for k in self._KEYS]

    def n_params(self):
        return int(sum(_np.asarray(v).size
                       for v in self.values()))

    def shape_map(self):
        return {k: tuple(_np.asarray(getattr(self, k)).shape)
                for k in self._KEYS}

    def __getstate__(self):
        return {k: _np.asarray(getattr(self, k))
                for k in self._KEYS}

    def __setstate__(self, st):
        for k in self._KEYS:
            setattr(self, k, st[k])
        self._bk = None


class Block(_ChainStruct):
    """The delta composition block (kind "block"): residual
    contribution gelu(H Bin^T + bb) Bout^T on the scope's
    stream. predict() returns the CONTRIBUTION (the kernel adds
    it to the stream); train_from_grad() applies the exact
    chain rule with plain SGD for STANDALONE use — in-scope
    training remains the fused kernels' (SR-14)."""

    _KEYS = ("Bin", "bb", "Bout")

    def __init__(self, Bin, bb, Bout, bk=None):
        self.Bin, self.bb, self.Bout = Bin, bb, Bout
        self._bk = bk

    def _gelu(self, Z):
        return self._bk.gelu(Z)

    def predict(self, H):
        Z = H @ self.Bin.T + self.bb
        return self._gelu(Z) @ self.Bout.T

    def train_from_grad(self, H, dU, sgd_lr=None):
        lr = 0.01 if sgd_lr is None else float(sgd_lr)
        Z = H @ self.Bin.T + self.bb
        G = self._gelu(Z)
        dBout = dU.T @ G
        dG = dU @ self.Bout
        dZ = dG * self._bk.gelu_d(Z)
        dBin = dZ.T @ H
        dbb = dZ.sum(axis=0)
        self.Bout = self.Bout - lr * dBout
        self.Bin = self.Bin - lr * dBin
        self.bb = self.bb - lr * dbb


class LoopBlock(_ChainStruct):
    """The lambda loop block (kind "loop"): same storage/
    mapping/serialization pattern; its recurrence semantics are
    SCOPE-OWNED (engine loop kernels) — predict() is the
    single-application contribution, documented as the
    structure's local map only."""

    _KEYS = ("L_in", "b_l", "L_out")

    def __init__(self, L_in, b_l, L_out, bk=None):
        self.L_in, self.b_l, self.L_out = L_in, b_l, L_out
        self._bk = bk

    def predict(self, H):
        Z = H @ self.L_in.T + self.b_l
        return self._bk.gelu(Z) @ self.L_out.T

    def train_from_grad(self, H, dU, sgd_lr=None):
        lr = 0.01 if sgd_lr is None else float(sgd_lr)
        Z = H @ self.L_in.T + self.b_l
        G = self._bk.gelu(Z)
        dLout = dU.T @ G
        dG = dU @ self.L_out
        dZ = dG * self._bk.gelu_d(Z)
        self.L_out = self.L_out - lr * dLout
        self.L_in = self.L_in - lr * (dZ.T @ H)
        self.b_l = self.b_l - lr * dZ.sum(axis=0)


def _block_builder(host, spec):
    """EXACT historical delta construction (preset_delta order:
    rng from the already-incremented seed; He Bin; zero bb;
    zero Bout)."""
    m = int(spec.params["m"])
    rng = _np.random.default_rng(spec.seed)
    bk = host._bk
    return Block(
        bk.ingest(rng.normal(0, _np.sqrt(2.0 / host.H),
                             (m, host.H))),
        bk.ingest(_np.zeros(m)),
        bk.ingest(_np.zeros((host.H, m))), bk=bk)


def _loop_builder(host, spec):
    """EXACT historical loop construction (net.loop order)."""
    m = int(spec.params["m"])
    rng = _np.random.default_rng(spec.seed)
    bk = host._bk
    return LoopBlock(
        bk.ingest(rng.normal(0, _np.sqrt(2.0 / host.H),
                             (m, host.H))),
        bk.ingest(_np.zeros(m)),
        bk.ingest(_np.zeros((host.H, m))), bk=bk)


for _k, _b in (("block", _block_builder), ("loop", _loop_builder)):
    from .foundation.compose import BUILDERS as _B
    _B.setdefault(_k, _b)
