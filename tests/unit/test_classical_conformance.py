"""Part A boxes (doc 37 v1.3): T-7 golden replay — the fixed-net
before/after comparison. Reruns the EXACT capture procedure on
the CURRENT code and compares every recorded number BITWISE
against the W0 goldens (captured from the pristine pre-reform
tree). Any fixed-net behavior change anywhere in the reform
trips this box. T-8/T-9 (matched-state textbook oracles) and
VM-1/VM-2 live in this file too as they land with W2."""
import importlib.util
import pickle   # safe: locally captured organ states only
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))

GOLD = REPO / "tests" / "unit" / "fixtures" / \
    "classical_goldens"

_spec = importlib.util.spec_from_file_location(
    "capture_classical_goldens",
    REPO / "scripts" / "capture_classical_goldens.py")
_cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cap)


# Attributes the reform ADDS to state (design-sanctioned, doc 35
# D2): permitted on the post-reform side ONLY at their documented
# fixed-net default — any other value or any other new key fails.
_REFORM_NEW_ATTRS = {"out_width": 1}


def _tree_equal(a, b, path):
    """Recursive BITWISE comparison of two state trees (arrays
    exact, scalars exact, objects via their state). a = golden
    (pre-reform), b = post-reform."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(np.asarray(a), np.asarray(b)), path
    elif isinstance(a, dict):
        assert not set(a) - set(b), path
        for k in set(b) - set(a):
            assert k in _REFORM_NEW_ATTRS \
                and b[k] == _REFORM_NEW_ATTRS[k], (path, k)
        for k in a:
            _tree_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple, deque)):
        assert len(a) == len(b), path
        for i, (x, z) in enumerate(zip(a, b)):
            _tree_equal(x, z, f"{path}[{i}]")
    elif isinstance(a, (int, float, complex, str, bool,
                        type(None))):
        if isinstance(a, float) and np.isnan(a):
            assert isinstance(b, float) and np.isnan(b), path
        else:
            assert type(a) is type(b) and a == b, path
    elif hasattr(a, "__getstate__") and a.__getstate__() \
            is not None:
        assert type(a) is type(b), path
        _tree_equal(a.__getstate__(), b.__getstate__(),
                    path + ">")
    else:
        assert type(a) is type(b), path
        _tree_equal(a.__dict__, b.__dict__, path + ">")


@pytest.mark.parametrize("name", [f"F{i}" for i in range(1, 9)])
def test_t7_fixed_net_bitwise_vs_golden(name):
    spec = _cap.build_configs()[name]
    rec = _cap.run_config(name, spec)
    gold = np.load(GOLD / f"{name}.npz")
    # losses: every one of the 100 adam + 20 sgd steps, bitwise
    assert np.array_equal(rec["losses_adam"],
                          gold["losses_adam"]), name
    assert np.array_equal(rec["losses_sgd"],
                          gold["losses_sgd"]), name
    # outputs at steps 0/50/100, bitwise
    for k in ("out0", "out50", "out100"):
        assert np.array_equal(rec[k], gold[k]), (name, k)
    # final behavior: the golden organ (unpickled) and the
    # fresh-run organ must serve identical outputs, bitwise
    m_gold = pickle.loads(gold["final_state"].tobytes())
    m_new = pickle.loads(rec["final_state"].tobytes())
    X = gold["X"]
    out_g = (m_gold.predict(X) if spec["kind"] == "numeric"
             else m_gold.predict_proba(X))
    out_n = (m_new.predict(X) if spec["kind"] == "numeric"
             else m_new.predict_proba(X))
    assert np.array_equal(np.asarray(out_g),
                          np.asarray(out_n)), name
    # EVERY parameter tensor (doc 37 T-7): the full final
    # state trees compare bitwise, not just served behavior
    _tree_equal(m_gold.__getstate__(), m_new.__getstate__(),
                name)


# ================= T-8 / T-9 / VM-1 / VM-2 / T-17 ==============
# INDEPENDENT textbook implementations, written from the
# documented equations (doc 35 5). They import NOTHING from the
# production forward paths; parameters are read as plain numpy.

def _np(x):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return a          # preserve T-9 complex-step perturbations
    return np.asarray(a, dtype=float)


def _gelu(a):
    """Documented activation (tanh form), restated here."""
    return 0.5 * a * (1.0 + np.tanh(
        0.7978845608 * (a + 0.044715 * a ** 3)))


def _ln(x, g, b, eps=1e-5):
    # variance written analytically (mean of squared deviation,
    # not np.var) so the SAME textbook formula is complex-
    # differentiable for the T-9 complex-step gradients
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    sd = np.sqrt(var + eps)
    return ((x - mu) / sd) * g + b


def _softmax_rows(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def textbook_mlp_forward(m, X, kind):
    """y = psi(W2 gelu(W1 x_hat + b1) + c) — the classical MLP
    equation (fixed net: no blocks, no bodies)."""
    Xh = (_np(X) - _np(m._x_mu)) / _np(m._x_sd) \
        if m._x_mu is not None else _np(X)
    Z = _gelu(Xh @ _np(m.W1).T + _np(m.b1))
    raw = Z @ _np(m.W2).T + _np(m.c)
    if kind == "numeric":
        return raw * m._y_sd + m._y_mu
    return _softmax_rows(raw)


def textbook_ga_forward(m, X, kind):
    """Pre-LN block equations with the two PROVEN-identity
    documented deviations (birth-absorbed scale; additive
    multi-head form)."""
    P = {k: _np(v) for k, v in m.P.items()}
    Xh = (_np(X) - _np(m._x_mu)) / _np(m._x_sd) \
        if m._x_mu is not None else _np(X)
    if m.CAUSAL:
        T = Xh @ P["Wv"] + P["Bf"][None, :Xh.shape[1], :]
    else:
        T = Xh[:, :, None] * P["Wv"][None] + P["Bf"][None]
    for l in range(m.L):
        Tn = _ln(T, P[f"g1_{l}"], P[f"b1n_{l}"])
        O = np.zeros_like(T)
        for HS in m.heads[l]:
            Wq, Wk, Wv_, Wo = (_np(HS.Wq), _np(HS.Wk),
                               _np(HS.Wv), _np(HS.Wo))
            S = (Tn @ Wq) @ (Tn @ Wk).swapaxes(1, 2)
            if m.CAUSAL:
                Tl = S.shape[-1]
                S = S + np.triu(np.full((Tl, Tl), -1e9), k=1)
            A = _softmax_rows(S)
            O = O + (A @ (Tn @ Wv_)) @ Wo
        T1 = T + O
        Tn2 = _ln(T1, P[f"g2_{l}"], P[f"b2n_{l}"])
        H = _gelu(Tn2 @ P[f"W1_{l}"] + P[f"b1_{l}"])
        T = T1 + (H @ P[f"W2_{l}"] + P[f"b2_{l}"])
    Pool = T[:, -1, :] if m.CAUSAL else T.mean(1)
    logits = Pool @ P["Wh"] + P["bh"]
    if kind == "numeric":
        return logits * m._y_sd + m._y_mu
    return _softmax_rows(logits)


def textbook_transformer_forward(m, X, kind):
    """Packed multi-head pre-LN block with explicit 1/sqrt(dh)."""
    P = {k: _np(v) for k, v in m.P.items()}
    n = len(X)
    Xh = (_np(X) - _np(m._x_mu)) / _np(m._x_sd) \
        if m._x_mu is not None else _np(X)
    hh = m.h
    d = m.d
    dh = d // hh
    if m.CAUSAL:
        T = Xh @ P["Wv"] + P["Bf"][None, :Xh.shape[1], :]
    else:
        T = Xh[:, :, None] * P["Wv"][None] + P["Bf"][None]
    for l in range(m.L):
        Tn = _ln(T, P[f"g1_{l}"], P[f"b1n_{l}"])

        def heads_(M):
            return M.reshape(n, -1, hh, dh).transpose(0, 2, 1, 3)
        Qh = heads_(Tn @ P[f"Wq_{l}"])
        Kh = heads_(Tn @ P[f"Wk_{l}"])
        Vh = heads_(Tn @ P[f"Wk2_{l}"])
        S = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(dh)
        if m.CAUSAL:
            Tl = S.shape[-1]
            S = S + np.triu(np.full((Tl, Tl), -1e9), k=1)
        A = _softmax_rows(S)
        O = (A @ Vh).transpose(0, 2, 1, 3).reshape(n, -1, d) \
            @ P[f"Wo_{l}"]
        T1 = T + O
        Tn2 = _ln(T1, P[f"g2_{l}"], P[f"b2n_{l}"])
        H = _gelu(Tn2 @ P[f"W1_{l}"] + P[f"b1_{l}"])
        T = T1 + (H @ P[f"W2_{l}"] + P[f"b2_{l}"])
    Pool = T[:, -1, :] if m.CAUSAL else T.mean(axis=1)
    logits = Pool @ P["Wh"] + P["bh"]
    if kind == "numeric":
        return logits * m._y_sd + m._y_mu
    return _softmax_rows(logits)


TEXTBOOK = {"F1": textbook_mlp_forward,
            "F2": textbook_mlp_forward,
            "F3": textbook_ga_forward,
            "F4": textbook_ga_forward,
            "F5": textbook_ga_forward,
            "F6": textbook_transformer_forward,
            "F7": textbook_transformer_forward,
            "F8": textbook_mlp_forward}


def _err(a, b):
    a, b = _np(a), _np(b)
    return float(np.abs(a - b).max() / max(1.0, np.abs(a).max()))


def _book_loss(m, X, y, spec, name):
    """The host's DOCUMENTED training objective evaluated from
    the INDEPENDENT textbook forward at the current state:
    numeric = mean((raw - y_std)^2) (all numeric hosts return
    this mse); categorical = -mean(log(p_true + 1e-12)) (the
    kernels' cat_ce formula, restated)."""
    ref = TEXTBOOK[name]
    if spec["kind"] == "numeric":
        raw = (_np(ref(m, X, "numeric")) - m._y_mu) / m._y_sd
        ys = (_np(y) - m._y_mu) / m._y_sd
        return float(((raw - ys) ** 2).mean())
    probs = _np(ref(m, X, "categorical"))
    idx = [m.vocab.index(v) for v in np.asarray(y).ravel()]
    return float(-np.mean(np.log(
        probs[np.arange(len(idx)), idx] + 1e-12)))


@pytest.mark.parametrize("name", [f"F{i}" for i in range(1, 9)])
def test_t8_matched_state_textbook_oracle(name):
    """T-8: at steps {0, 50, 100} the code's own state is
    evaluated by the INDEPENDENT textbook implementation; the
    served OUTPUTS and the training LOSSES must agree within
    1e-12 (matched-state oracle — references are never
    co-trained). The loss check uses the loss train_step
    RETURNS at that same state (evaluated pre-update)."""
    spec = _cap.build_configs()[name]
    m = spec["make"]()
    X, y = spec["data"]()
    ref = TEXTBOOK[name]
    checks = loss_checks = 0
    for i in range(101):
        at_ck = i in (0, 50, 100) and (
            getattr(m, "_x_mu", None) is not None)
        if at_ck:
            code_out = (m.predict(X) if spec["kind"] == "numeric"
                        else m.predict_proba(X))
            book_out = ref(m, X, spec["kind"])
            assert _err(code_out, book_out) < 1e-12, (name, i)
            checks += 1
            book_l = _book_loss(m, X, y, spec, name)
        if i < 100:
            ret = m.train_step(X, y)
            if at_ck:                    # ret = loss AT state i
                assert abs(float(ret) - book_l) \
                    <= 1e-12 * max(1.0, abs(book_l)), (name, i)
                loss_checks += 1
            if spec.get("add_class_at") == i + 1:
                m.add_class("t_new")
    assert checks >= 2 and loss_checks >= 1, name


@pytest.mark.parametrize("name", ["F1", "F2"])
def test_t9_matched_state_gradients_mlp_family(name):
    """T-9 (closed-form, numeric MLP family — F1 mlp host and
    F2 reference Network share the documented kernel): the FULL
    backward is hand-derivable; code gradients (sgd_lr=1 delta)
    must equal the closed forms at 1e-10."""
    spec = _cap.build_configs()[name]
    m = spec["make"]()
    X, y = spec["data"]()
    for _ in range(10):
        m.train_step(X, y)
    before = {k: _np(getattr(m, k))
              for k in ("W1", "b1", "W2", "c")}
    y_mu, y_sd = m._y_mu, m._y_sd
    x_mu, x_sd = _np(m._x_mu), _np(m._x_sd)
    m.train_step(X, y, sgd_lr=1.0)
    applied = {k: before[k] - _np(getattr(m, k)) for k in before}
    # closed forms for L = mean((pred_std - y_std)^2) with the
    # kernel's 2/n-in-lr convention: err_hat = 2*(pred-y)/n is
    # NOT used by the kernel — it uses err/n scaling internally;
    # derive from the kernel-documented loss: gW2 = err^T Z / n
    Xh = (_np(X) - x_mu) / x_sd
    A = Xh @ before["W1"].T + before["b1"]
    Z = _gelu(A)
    pred = Z @ before["W2"].T + before["c"]
    ys = (_np(y) - y_mu) / y_sd
    err = pred - ys
    n = len(X)
    gW2 = err.T @ Z / n
    gc = err.mean(0)
    t = np.tanh(0.7978845608 * (A + 0.044715 * A ** 3))
    dgelu = 0.5 * (1.0 + t) + 0.5 * A * (1.0 - t ** 2) \
        * 0.7978845608 * (1.0 + 3 * 0.044715 * A ** 2)
    dZ = (err @ before["W2"]) * dgelu / n
    gW1 = dZ.T @ Xh
    gb1 = dZ.sum(0)
    for nm, g in (("W2", gW2), ("c", gc), ("W1", gW1),
                  ("b1", gb1)):
        assert _err(applied[nm], g) < 1e-10, nm


def test_vm1_textbook_reference_anchored_by_hand():
    """VM-1: the independent MLP reference itself is checked
    against a pencil case — W1 = [[1,0],[0,1]], b1 = 0,
    W2 = [[1,1]], c = 0.5, x = [2,-1] (identity scalers):
      A = [2,-1]; gelu(2) = 1.9545977..., gelu(-1) =
      -0.15880801..., derived from the documented tanh form
      evaluated by hand below as literals; y = their sum + 0.5."""
    class Tiny:
        W1 = np.eye(2)
        b1 = np.zeros(2)
        W2 = np.array([[1.0, 1.0]])
        c = np.array([0.5])
        _x_mu = None
        _y_mu, _y_sd = 0.0, 1.0
    x = np.array([[2.0, -1.0]])
    got = float(textbook_mlp_forward(Tiny, x, "numeric")[0, 0])
    g2 = 0.5 * 2.0 * (1.0 + np.tanh(
        0.7978845608 * (2.0 + 0.044715 * 8.0)))
    g1 = 0.5 * -1.0 * (1.0 + np.tanh(
        0.7978845608 * (-1.0 - 0.044715)))
    assert abs(got - (g2 + g1 + 0.5)) < 1e-14


def test_t17_hosts_own_no_private_attach_or_handoff():
    """T-17 (R2 enforcement): no host retains a private attach
    loop or eta-handoff block; the ONLY implementations live in
    growth_port.py."""
    retired = ("net.predict(X)[:, 0]",
               "net.predict(inner_in)[:, 0]",
               "t_j = Hact[:, j]",
               "t_j = H[:, :, j]")
    hosts = ("modules/ReferenceNet/reference_net/net.py",
             "core/substrate.py",
             "core/substrates/growable_attention.py",
             "core/substrates/transformer.py")
    for h in hosts:
        src = (REPO / h).read_text()
        for pat in retired:
            assert pat not in src, (h, pat)
    # A5 (doc 52 s6.4, FR-8): the single implementation home
    # moved by verbatim code motion to legacy_compat.py;
    # growth_port keeps only the re-export (facade). The box's
    # INTENT is unchanged: exactly ONE implementation home.
    legacy_src = (REPO / "modules/ReferenceNet/reference_net/"
                         "legacy_compat.py").read_text()
    assert "net.predict(X)[:, 0]" in legacy_src
    port_src = (REPO / "modules/ReferenceNet/reference_net/"
                       "growth_port.py").read_text()
    assert "net.predict(X)[:, 0]" not in port_src
    assert "from .legacy_compat import" in port_src


# ================= T-10 / T-11 / VM-2 (audit completion) ========

def test_t10_fixed_net_pickle_round_trip_bitwise():
    """T-10: fixed nets serialize/deserialize with identical
    served behavior (bitwise), across all four host families."""
    for name in ("F1", "F2", "F5", "F6"):
        spec = _cap.build_configs()[name]
        m = spec["make"]()
        X, y = spec["data"]()
        for _ in range(5):
            m.train_step(X, y)
        m2 = pickle.loads(pickle.dumps(m))
        a = (m.predict(X) if spec["kind"] == "numeric"
             else m.predict_proba(X))
        b = (m2.predict(X) if spec["kind"] == "numeric"
             else m2.predict_proba(X))
        assert np.array_equal(np.asarray(a), np.asarray(b)), name


def test_t11_legacy_grown_artifact_load_smoke():
    """T-11 (informational, NEVER a correctness reference): a
    pre-flip LEGACY-GROWN artifact loads and serves identically
    to its captured outputs — archived results stay readable."""
    fx = REPO / "tests" / "unit" / "fixtures" / \
        "legacy_grown_artifact"
    m = pickle.loads((fx / "net_legacy_grown.pkl").read_bytes())
    exp = np.load(fx / "expected.npz")
    assert list(m.inner), "fixture must contain a legacy body"
    assert np.array_equal(np.asarray(m.predict(exp["X"])),
                          exp["pred"])


def test_vm2_backward_pencil_literals():
    """VM-2: the closed-form backward used by T-9 is itself
    anchored by a pencil case. Setup (identity scalers):
      W1 = [[1,0],[0,1]], b1 = 0, W2 = [[1,1]], c = 0,
      x = [2,-1], target y = 0, n = 1.
    Forward: A = [2,-1]; Z = [gelu(2), gelu(-1)];
      pred = gelu(2) + gelu(-1); err = pred - 0.
    Hand gradients (n = 1):
      gW2 = err * Z            (row vector)
      gc  = err
      dZ  = err * W2 * gelu'(A) = err * [gelu'(2), gelu'(-1)]
      gW1 = outer(dZ, x); gb1 = dZ."""
    g2, g1 = _gelu(np.array(2.0)), _gelu(np.array(-1.0))
    t2 = np.tanh(0.7978845608 * (2.0 + 0.044715 * 8.0))
    d2 = 0.5 * (1 + t2) + 0.5 * 2.0 * (1 - t2 ** 2) \
        * 0.7978845608 * (1 + 3 * 0.044715 * 4.0)
    t1 = np.tanh(0.7978845608 * (-1.0 - 0.044715))
    d1 = 0.5 * (1 + t1) + 0.5 * -1.0 * (1 - t1 ** 2) \
        * 0.7978845608 * (1 + 3 * 0.044715 * 1.0)
    err = float(g2 + g1)
    hand_gW2 = np.array([[err * float(g2), err * float(g1)]])
    hand_gW1 = np.array([[err * float(d2) * 2.0,
                          err * float(d2) * -1.0],
                         [err * float(d1) * 2.0,
                          err * float(d1) * -1.0]])
    # the same closed forms as T-9, evaluated at the pencil point
    Xh = np.array([[2.0, -1.0]])
    W1 = np.eye(2)
    W2 = np.array([[1.0, 1.0]])
    A = Xh @ W1.T
    Z = _gelu(A)
    pred = Z @ W2.T
    e = pred - 0.0
    gW2 = e.T @ Z / 1
    t = np.tanh(0.7978845608 * (A + 0.044715 * A ** 3))
    dg = 0.5 * (1 + t) + 0.5 * A * (1 - t ** 2) \
        * 0.7978845608 * (1 + 3 * 0.044715 * A ** 2)
    dZ = (e @ W2) * dg / 1
    gW1 = dZ.T @ Xh
    assert np.abs(gW2 - hand_gW2).max() < 1e-14
    assert np.abs(gW1 - hand_gW1).max() < 1e-14


# ---- T-12 mapping note (box index, doc 37 4): T-12 = the
# suite-wide default-path green gate. It is a GATE-level box
# (the full lib suite run recorded in each commit's gate log),
# not a unit test in this file — recorded here so the doc 37
# box index resolves completely.


def test_d6_legacy_artifacts_load_as_legacy_scalar_marker():
    """D6 (doc 35): artifacts with grown bodies but NO port
    field load AS legacy_scalar — the audit-visible marker is
    attached on setstate for all three host setstate paths
    (Network incl. mlp substrate via inheritance; GA;
    transformer). Fixed nets (empty inner) get NO marker."""
    from reference_net.growth_port import LegacyScalarPort
    fx = REPO / "tests" / "unit" / "fixtures" / \
        "legacy_grown_artifact"
    m = pickle.loads((fx / "net_legacy_grown.pkl").read_bytes())
    mk = getattr(m, "_legacy_port", None)
    assert isinstance(mk, LegacyScalarPort)
    assert mk.PORT_TYPE == "legacy_scalar"
    # GA host: a PRE-REFORM artifact state = legacy inner dict
    # and NO port field (constructed the way old pickles look;
    # W3 migration note: grow_site now produces FULLWIDTH organs,
    # which correctly get NO legacy marker — asserted below)
    from reference_net.net import Network as _Net
    spec = _cap.build_configs()["F3"]
    ga = spec["make"]()
    X, y = spec["data"]()
    for _ in range(3):
        ga.train_step(X, y)
    ga.inner[(0, 1)] = _Net(ga.d, 6, lr=1e-3, seed=9,
                            zero_out=True)      # legacy shape
    ga2 = pickle.loads(pickle.dumps(ga))
    assert isinstance(getattr(ga2, "_legacy_port", None),
                      LegacyScalarPort)
    # a fullwidth-grown organ is NOT legacy: no marker
    gaf = spec["make"]()
    for _ in range(3):
        gaf.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gaf.grow_site("layer0/ffn[1]", hidden=6)
    gaf2 = pickle.loads(pickle.dumps(gaf))
    assert getattr(gaf2, "_legacy_port", None) is None
    assert gaf2._port_sites[0].bodies       # fullwidth body there
    # fixed net: NO marker (red line untouched)
    fixed = pickle.loads(pickle.dumps(spec["make"]()))
    assert getattr(fixed, "_legacy_port", None) is None


# ======== T-9 PER CONFIG (doc 37: gradient equality per
# config, >= 4 coordinates per parameter family, 1e-10) ========
# Reference for the attention configs: COMPLEX-STEP exact
# differentiation of the SAME independent textbook formulas
# used by T-8 (h = 1e-30: truncation O(h^2) is far below f64
# resolution, and there is no subtractive cancellation — the
# derivative is exact to roundoff). This differentiates the
# textbook MATH with machine precision and carries no
# transcription risk; the hand-derived anchors stay in
# VM-1/VM-2 and the MLP-family closed forms.

_H_STEP = 1e-30


class _Shim:
    """Plain-attribute stand-in evaluated by the textbook
    forwards; carries (possibly complex-perturbed) numpy
    parameters extracted from the real organ."""


def _ga_extract(m):
    P = {k: _np(v) for k, v in m.P.items()}
    H = {}
    for l, layer in enumerate(m.heads):
        for hh, HS in enumerate(layer):
            for w in ("Wq", "Wk", "Wv", "Wo"):
                H[(l, hh, w)] = _np(getattr(HS, w))
    return P, H


def _ga_shim(m, P, H):
    s = _Shim()
    s.P = P
    s.CAUSAL, s.L = m.CAUSAL, m.L
    s._x_mu = _np(m._x_mu) if m._x_mu is not None else None
    s._x_sd = _np(m._x_sd) if m._x_mu is not None else None
    s._y_mu, s._y_sd = 0.0, 1.0          # raw logits head
    heads = []
    for l, layer in enumerate(m.heads):
        row = []
        for hh, _ in enumerate(layer):
            hs = _Shim()
            for w in ("Wq", "Wk", "Wv", "Wo"):
                setattr(hs, w, H[(l, hh, w)])
            row.append(hs)
        heads.append(row)
    s.heads = heads
    return s


def _tr_extract(m):
    return {k: _np(v) for k, v in m.P.items()}, {}


def _tr_shim(m, P, H):
    s = _Shim()
    s.P = P
    s.CAUSAL, s.L, s.h, s.d = m.CAUSAL, m.L, m.h, m.d
    s._x_mu = _np(m._x_mu) if m._x_mu is not None else None
    s._x_sd = _np(m._x_sd) if m._x_mu is not None else None
    s._y_mu, s._y_sd = 0.0, 1.0
    return s


def _att_objective(m, shim, ref, X, y, kind):
    """The host's DOCUMENTED objective (gradient convention:
    numeric dlog = 2 err/n <-> L = mean(err^2); categorical
    dlog = (probs - onehot)/n <-> L = -mean(log p_true),
    the 1e-12 guard being report-only)."""
    if kind == "numeric":
        raw = ref(shim, X, "numeric")     # shim head is raw
        ys = (_np(y) - m._y_mu) / m._y_sd
        return ((raw - ys) ** 2).mean()
    probs = ref(shim, X, "categorical")
    idx = [m.vocab.index(v) for v in np.asarray(y).ravel()]
    return -(np.log(probs[np.arange(len(idx)), idx])).mean()


def _family_coords(P, H, k=4):
    """>= k coordinates per parameter FAMILY (family = base
    tensor name, pooled across layers/heads), spread across
    members and positions."""
    fams = {}
    for key in P:
        fams.setdefault(key.split("_")[0], []).append(("P", key))
    for key in H:
        fams.setdefault("head." + key[2], []).append(("H", key))
    coords = []
    for fam in sorted(fams):
        members = sorted(fams[fam], key=str)
        picked = set()
        for j in range(k):
            kindt, key = members[j % len(members)]
            size = int(np.prod((P if kindt == "P"
                                else H)[key].shape))
            idx = 0 if size == 1 else (j * (size - 1)) // (k - 1)
            picked.add((kindt, key, idx))
        coords += sorted(picked, key=str)
    return coords


def _complex_step_grad(loss_of_tree, P, H, coord):
    kindt, key, idx = coord
    P2 = dict(P)
    H2 = dict(H)
    tgt = P2 if kindt == "P" else H2
    arr = tgt[key].astype(complex).copy()
    arr.flat[idx] += 1j * _H_STEP
    tgt[key] = arr
    return float(np.imag(loss_of_tree(P2, H2)) / _H_STEP)


def _t9_attention_config(name, extract, shim_of, ref):
    spec = _cap.build_configs()[name]
    m = spec["make"]()
    X, y = spec["data"]()
    for _ in range(10):
        m.train_step(X, y)
    P0, H0 = extract(m)
    m.train_step(X, y, sgd_lr=1.0)      # delta trick: raw grads
    P1, H1 = extract(m)
    coords = _family_coords(P0, H0)

    def loss_of_tree(P, H):
        return _att_objective(m, shim_of(m, P, H), ref, X, y,
                              spec["kind"])

    fams_checked = set()
    for coord in coords:
        kindt, key, idx = coord
        g_code = float((P0 if kindt == "P" else H0)[key].flat[idx]
                       - (P1 if kindt == "P" else H1)[key]
                       .flat[idx])
        g_book = _complex_step_grad(loss_of_tree, P0, H0, coord)
        scale = max(1.0, abs(g_book))
        assert abs(g_code - g_book) / scale < 1e-10, (name, coord)
        fams_checked.add(key.split("_")[0] if kindt == "P"
                         else "head." + key[2])
    # every parameter family of the host was covered
    assert fams_checked == {k.split("_")[0] for k in P0} \
        | {"head." + k[2] for k in H0}, name


@pytest.mark.parametrize("name", ["F3", "F4", "F5"])
def test_t9_per_config_gradients_ga(name):
    """T-9 for the growable_attention configs: code gradients
    (sgd_lr = 1 delta) vs complex-step exact derivatives of the
    independent textbook formulas, >= 4 coordinates for EVERY
    parameter family (embed, LN gains/biases, FFN mats, every
    head matrix, readout), 1e-10 scaled."""
    _t9_attention_config(name, _ga_extract, _ga_shim,
                         textbook_ga_forward)


@pytest.mark.parametrize("name", ["F6", "F7"])
def test_t9_per_config_gradients_transformer(name):
    """T-9 for the transformer configs (packed heads incl. the
    Wk2 V-projection), same method and tolerance."""
    _t9_attention_config(name, _tr_extract, _tr_shim,
                         textbook_transformer_forward)


def test_t9_per_config_gradients_f8_categorical_closed_form():
    """T-9 for F8 (mlp host, categorical): FULL closed-form
    backward of the documented cat kernel —
      probs = softmax(gelu(W1 x_hat + b1) W2^T + c)
      err = probs - onehot
      gW2 = err^T Hact / n;  gc = err.mean(0)
      dA = (err W2) * gelu'(A);  gW1 = dA^T x_hat / n;
      gb1 = dA.mean(0)
    vs sgd_lr = 1 deltas at 1e-10 (all coordinates)."""
    spec = _cap.build_configs()["F8"]
    m = spec["make"]()
    X, y = spec["data"]()
    for _ in range(10):
        m.train_step(X, y)
    before = {k: _np(getattr(m, k))
              for k in ("W1", "b1", "W2", "c")}
    x_mu, x_sd = _np(m._x_mu), _np(m._x_sd)
    idx = [m.vocab.index(v) for v in np.asarray(y).ravel()]
    m.train_step(X, y, sgd_lr=1.0)
    applied = {k: before[k] - _np(getattr(m, k)) for k in before}
    n = len(X)
    Xh = (_np(X) - x_mu) / x_sd
    A = Xh @ before["W1"].T + before["b1"]
    Z = _gelu(A)
    logits = Z @ before["W2"].T + before["c"]
    probs = _softmax_rows(logits)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), idx] = 1.0
    err = probs - onehot
    gW2 = err.T @ Z / n
    gc = err.mean(0)
    t = np.tanh(0.7978845608 * (A + 0.044715 * A ** 3))
    dgelu = 0.5 * (1.0 + t) + 0.5 * A * (1.0 - t ** 2) \
        * 0.7978845608 * (1.0 + 3 * 0.044715 * A ** 2)
    dA = (err @ before["W2"]) * dgelu
    gW1 = dA.T @ Xh / n
    gb1 = dA.mean(0)
    for nm, g in (("W2", gW2), ("c", gc), ("W1", gW1),
                  ("b1", gb1)):
        assert _err(applied[nm], g) < 1e-10, nm
