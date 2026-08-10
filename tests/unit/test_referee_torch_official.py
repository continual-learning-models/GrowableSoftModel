"""60C: THIRD-PARTY REFEREE verification of every self-written
attention computation, against PyTorch official components
(the industry-standard authority). Owner ruling: self-written
tests alone are self-verification; the referee here is
EXTERNAL — torch.nn.functional components for the forward,
torch autograd for every gradient. A red in this file is a
REAL FINDING about our math (defect protocol; tolerances are
never loosened to pass). Multiple computation cases per box
(configurations x seeds x probe batches). Conventions table:
doc 60C s1c."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip(
    "torch", reason="the third-party referee REQUIRES torch — "
    "without it this verification cannot run (loud skip)")
import torch.nn.functional as TF                     # noqa: E402

from engine.backends import set_compute_policy       # noqa: E402
from core.substrates.transformer import \
    TransformerSubstrate                             # noqa: E402
from core.substrates.transformer_plus import \
    TransformerPlus                                  # noqa: E402
from core.substrates.growable_attention import \
    GrowableAttentionSubstrate                       # noqa: E402
from core.substrates.sequence import \
    SequenceSubstrate                                # noqa: E402
from reference_net.attention_body import \
    AttentionBody                                    # noqa: E402

F64 = torch.float64
# 60C s2c ADJUDICATION (first-run record): the a-priori 1e-12
# absolute forward line was breached at 1.07e-12..1.38e-12 by
# the larger configs with digit-identical values — cross-
# library op-order accumulation (arithmetic-physics regime),
# not computation error. Adjudicated lines, recorded in the
# doc with the evidence — never quietly tuned:
FWD_TOL = 1e-11
GRAD_TOL = 1e-10
TRAJ_TOL = 1e-9      # 5 co-evolved SGD steps (60C U-2)


@pytest.fixture(autouse=True)
def _numpy_judge():
    set_compute_policy("numpy", "cpu", None)
    yield
    set_compute_policy("numpy", "cpu", None)


def _t(x, grad=False):
    a = torch.tensor(np.asarray(x, dtype=np.float64), dtype=F64)
    if grad:
        a.requires_grad_(True)
    return a


def _np(x):
    return np.asarray(x, dtype=np.float64)


# =====================================================
# H-1/H-3: the torch-official twin for the STANDARD scheme
# (TransformerSubstrate, SequenceSubstrate, AttentionBody —
# scaled dot-product attention, contiguous head split).
# Reads shapes dynamically: pristine/inserted/widened/dist
# instants all transplant through this ONE builder.
# =====================================================

def twin_std_forward(P, L, hh, X, causal=False,
                     head_names=("Wh", "bh"), ports=None,
                     m_ffn=None):
    """P: dict of torch tensors; official components only:
    F.layer_norm (eps 1e-5 = ours), F.scaled_dot_product_
    attention (scale 1/sqrt(dh) = ours), F.gelu tanh."""
    d = P["Wv"].shape[1]
    n = X.shape[0]
    if causal:
        Tlen = X.shape[1]
        T = X @ P["Wv"] + P["Bf"][None, :Tlen, :]
    else:
        T = X[:, :, None] * P["Wv"][None] + P["Bf"][None]
    for l in range(L):
        Tn = TF.layer_norm(T, (d,), P[f"g1_{l}"],
                           P[f"b1n_{l}"], eps=1e-5)
        Q, K, V = Tn @ P[f"Wq_{l}"], Tn @ P[f"Wk_{l}"], \
            Tn @ P[f"Wk2_{l}"]
        dh = d // hh
        Fn = Q.shape[1]

        def heads_(M):
            return M.reshape(n, Fn, hh, dh).permute(0, 2, 1, 3)
        Qh, Kh, Vh = heads_(Q), heads_(K), heads_(V)
        mask = None
        if causal:
            mask = _t(np.triu(np.full((Fn, Fn), -1e9), k=1))
        Oh = TF.scaled_dot_product_attention(
            Qh, Kh, Vh, attn_mask=mask)
        O = Oh.permute(0, 2, 1, 3).reshape(n, Fn, d) \
            @ P[f"Wo_{l}"]
        T1 = T + O
        Tn2 = TF.layer_norm(T1, (d,), P[f"g2_{l}"],
                            P[f"b2n_{l}"], eps=1e-5)
        H = TF.gelu(Tn2 @ P[f"W1_{l}"] + P[f"b1_{l}"],
                    approximate="tanh")
        if ports and ports.get(l):
            m = m_ffn if m_ffn is not None else H.shape[-1]
            Hf = H.reshape(-1, m)
            xin = Tn2.reshape(-1, d)
            for body_P, A in ports[l]:
                u = TF.gelu(xin @ body_P["W1"].T
                            + body_P["b1"],
                            approximate="tanh") \
                    @ body_P["W2"].T + body_P["c"]
                Hf = Hf + u @ A
            H = Hf.reshape(n, Fn, m)
        T = T1 + (H @ P[f"W2_{l}"] + P[f"b2_{l}"])
    Pool = T[:, -1, :] if causal else T.mean(dim=1)
    Wh, bh = head_names
    return Pool @ P[Wh] + P[bh]


def transplant_std(organ, grad=False, extra_elems=0):
    P = {k: _t(organ._bk.to_numpy(v), grad=grad)
         for k, v in organ.P.items()}
    _census(organ, sum(int(v.numel()) for v in P.values())
            + extra_elems)
    return P


def _census(organ, transplanted_elems):
    """60C U-3 (R-6): the referee must SEE every parameter —
    summed transplanted element count == the organ's own
    n_params(). An untransplanted tensor breaks this loudly."""
    assert transplanted_elems == int(organ.n_params()), (
        transplanted_elems, int(organ.n_params()))


# =====================================================
# H-2: the twin for GrowableAttention — a RESEARCH operator
# (SCALE-FREE, ragged per-head widths, additive per-head Wo:
# doc 60C s1b) — assembled from torch OFFICIAL PRIMITIVES
# (layer_norm, softmax, tanh-gelu, matmul); autograd referees
# every gradient.
# =====================================================

def transplant_ga(organ, grad=False):
    P = {k: _t(organ._bk.to_numpy(v), grad=grad)
         for k, v in organ.P.items()}
    heads = [[{nm: _t(organ._bk.to_numpy(getattr(HS, nm)),
                      grad=grad)
               for nm in ("Wq", "Wk", "Wv", "Wo")}
              for HS in layer] for layer in organ.heads]
    n = sum(int(v.numel()) for v in P.values()) \
        + sum(int(t.numel()) for layer in heads
              for HS in layer for t in HS.values())
    _census(organ, n)
    return P, heads


def twin_ga_forward(P, heads, L, X):
    d = P["Wv"].shape[1]
    T = X[:, :, None] * P["Wv"][None] + P["Bf"][None]
    for l in range(L):
        Tn = TF.layer_norm(T, (d,), P[f"g1_{l}"],
                           P[f"b1n_{l}"], eps=1e-5)
        O = torch.zeros_like(T)
        for HS in heads[l]:
            Qh = Tn @ HS["Wq"]
            Kh = Tn @ HS["Wk"]
            Vh = Tn @ HS["Wv"]
            S = Qh @ Kh.transpose(1, 2)        # SCALE-FREE (ours)
            A = torch.softmax(S, dim=-1)       # official softmax
            O = O + (A @ Vh) @ HS["Wo"]        # additive form
        T1 = T + O
        Tn2 = TF.layer_norm(T1, (d,), P[f"g2_{l}"],
                            P[f"b2n_{l}"], eps=1e-5)
        H = TF.gelu(Tn2 @ P[f"W1_{l}"] + P[f"b1_{l}"],
                    approximate="tanh")
        T = T1 + (H @ P[f"W2_{l}"] + P[f"b2_{l}"])
    Pool = T.mean(dim=1)
    return Pool @ P["Wh"] + P["bh"]


# =====================================================
# fixtures: multiple computation cases
# =====================================================

TR_CASES = [
    dict(d_in=3, hidden=8, d_model=8, n_layers=1, n_heads=1,
         seed=3),
    dict(d_in=4, hidden=6, d_model=16, n_layers=2, n_heads=2,
         seed=11),
    dict(d_in=3, hidden=8, d_model=12, n_layers=1, n_heads=3,
         seed=7),
]
GA_CASES = [
    dict(d_in=3, hidden=8, d_model=8, n_layers=1,
         heads_spec=[[1]], seed=3),
    dict(d_in=4, hidden=6, d_model=8, n_layers=1,
         heads_spec=[[2, 1]], seed=11),
    dict(d_in=3, hidden=8, d_model=12, n_layers=2,
         heads_spec=[[1, 3], [2]], seed=7),
]
PROBE_SEEDS = (0, 1)


def _data(d_in, n=24, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_in))
    y = (2.0 * X[:, 0] - X[:, 1]).reshape(-1, 1)
    return X, y


def _mk_tr(case, mode="numeric", cls=TransformerSubstrate,
           steps=30):
    kw = dict(case)
    o = cls(kw.pop("d_in"), kw.pop("hidden"), mode=mode, **kw)
    X, y = _data(case["d_in"], seed=case["seed"])
    for _ in range(steps):
        o.train_step(X, y)
    return o, X, y


def _mk_ga(case, mode="numeric", steps=30):
    kw = dict(case)
    o = GrowableAttentionSubstrate(kw.pop("d_in"),
                                   kw.pop("hidden"),
                                   mode=mode, **kw)
    X, y = _data(case["d_in"], seed=case["seed"])
    for _ in range(steps):
        o.train_step(X, y)
    return o, X, y


def _fwd_ours(organ, Xs):
    return _np(organ._bk.to_numpy(
        organ._forward(organ._bk.ingest(Xs))))


def _probes(organ, d_in, k=2):
    for ps in PROBE_SEEDS[:k]:
        Xq = np.random.default_rng(100 + ps).normal(
            size=(8, d_in))
        yield _np(organ._bk.to_numpy(organ._stdx(Xq)))


# =====================================================
# R-1a/R-1b/R-1c: FORWARD referee, multiple cases
# =====================================================

@pytest.mark.parametrize("case", TR_CASES,
                         ids=[f"tr{i}" for i in
                              range(len(TR_CASES))])
def test_r1a_transformer_forward_vs_torch_official(case):
    organ, X, _ = _mk_tr(case)
    P = transplant_std(organ)
    for Xs in _probes(organ, case["d_in"]):
        ours = _fwd_ours(organ, Xs)
        ref = twin_std_forward(P, organ.L, organ.h,
                               _t(Xs)).detach().numpy()
        assert np.abs(ours - ref).max() <= FWD_TOL
    # end-to-end served value once (scalers included)
    v_ours = _np(organ._bk.to_numpy(organ.predict(X[:4])))
    Xs = _np(organ._bk.to_numpy(organ._stdx(X[:4])))
    z = twin_std_forward(P, organ.L, organ.h,
                         _t(Xs)).detach().numpy()
    v_ref = z * float(organ._y_sd) + float(organ._y_mu)
    assert np.abs(v_ours - v_ref).max() <= 1e-10


@pytest.mark.parametrize("case", GA_CASES,
                         ids=[f"ga{i}" for i in
                              range(len(GA_CASES))])
def test_r1b_ga_forward_vs_torch_primitives(case):
    organ, X, _ = _mk_ga(case)
    P, heads = transplant_ga(organ)
    for Xs in _probes(organ, case["d_in"]):
        ours = _fwd_ours(organ, Xs)
        ref = twin_ga_forward(P, heads, organ.L,
                              _t(Xs)).detach().numpy()
        assert np.abs(ours - ref).max() <= FWD_TOL


def test_r1c_sequence_causal_forward():
    o = SequenceSubstrate(1, 8, mode="numeric", d_model=8,
                          n_layers=1, n_heads=1, seed=5)
    assert o.CAUSAL
    rng = np.random.default_rng(2)
    X = rng.normal(size=(16, 4, 1))
    y = X[:, -1, 0] + 0.5
    for _ in range(20):
        o.train_step(X, y)
    Xs = _np(o._bk.to_numpy(o._stdx(X[:6])))
    ours = _fwd_ours(o, Xs)
    ref = twin_std_forward(transplant_std(o), o.L, o.h,
                           _t(Xs), causal=True)
    assert np.abs(ours - ref.detach().numpy()).max() <= FWD_TOL


# =====================================================
# R-2: GRADIENT referee — torch autograd vs our hand-written
# backward (one pure-SGD step delta / eta == our gradient)
# =====================================================

def _our_grads_sgd(organ, X, y, eta=1e-3, heads=False):
    o2 = copy.deepcopy(organ)
    before = {k: _np(o2._bk.to_numpy(v)).copy()
              for k, v in o2.P.items()}
    hb = []
    if heads:
        hb = [[{nm: _np(o2._bk.to_numpy(getattr(HS, nm))).copy()
                for nm in ("Wq", "Wk", "Wv", "Wo")}
               for HS in layer] for layer in o2.heads]
    o2.train_step(X, y, sgd_lr=eta)
    G = {k: (before[k] - _np(o2._bk.to_numpy(o2.P[k]))) / eta
         for k in before}
    Gh = None
    if heads:
        Gh = [[{nm: (hb[l][h][nm] - _np(o2._bk.to_numpy(
                    getattr(o2.heads[l][h], nm)))) / eta
                for nm in ("Wq", "Wk", "Wv", "Wo")}
               for h in range(len(o2.heads[l]))]
              for l in range(len(o2.heads))]
    return G, Gh


def _rel(a, b):
    d = np.abs(a - b).max()
    s = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    return d / s


def _std_targets(organ, y):
    return (np.asarray(y, float).reshape(-1)
            - float(organ._y_mu)) / float(organ._y_sd)


@pytest.mark.parametrize("case", TR_CASES[:2],
                         ids=["tr0", "tr1"])
def test_r2a_transformer_gradients_vs_autograd(case):
    organ, X, y = _mk_tr(case)
    G_ours, _ = _our_grads_sgd(organ, X, y)
    P = transplant_std(organ, grad=True)
    Xs = _t(_np(organ._bk.to_numpy(organ._stdx(X))))
    ys = _t(_std_targets(organ, y))
    logits = twin_std_forward(P, organ.L, organ.h, Xs)
    loss = ((logits.reshape(-1) - ys) ** 2).mean()
    loss.backward()
    for k, v in P.items():
        assert v.grad is not None, k
        assert _rel(G_ours[k],
                    v.grad.detach().numpy()) <= GRAD_TOL, k


@pytest.mark.parametrize("case", GA_CASES[:2],
                         ids=["ga0", "ga1"])
def test_r2b_ga_gradients_vs_autograd(case):
    organ, X, y = _mk_ga(case)
    G_ours, Gh_ours = _our_grads_sgd(organ, X, y, heads=True)
    P, heads = transplant_ga(organ, grad=True)
    Xs = _t(_np(organ._bk.to_numpy(organ._stdx(X))))
    ys = _t(_std_targets(organ, y))
    logits = twin_ga_forward(P, heads, organ.L, Xs)
    loss = ((logits.reshape(-1) - ys) ** 2).mean()
    loss.backward()
    for k, v in P.items():
        assert _rel(G_ours[k],
                    v.grad.detach().numpy()) <= GRAD_TOL, k
    for l, layer in enumerate(heads):
        for h, HS in enumerate(layer):
            for nm, v in HS.items():
                assert _rel(Gh_ours[l][h][nm],
                            v.grad.detach().numpy()) \
                    <= GRAD_TOL, (l, h, nm)


# =====================================================
# R-3: GROWN INSTANTS (quasi-static: every instant is a
# fixed standard net and must transplant)
# =====================================================

def test_r3a_transformer_post_insert_layer_instant():
    case = TR_CASES[0]
    organ, X, y = _mk_tr(case)
    organ.insert_layer(1)
    for _ in range(15):
        organ.train_step(X, y)          # TRAINED instant
    P = transplant_std(organ)
    for Xs in _probes(organ, case["d_in"]):
        ours = _fwd_ours(organ, Xs)
        ref = twin_std_forward(P, organ.L, organ.h, _t(Xs))
        assert np.abs(ours -
                      ref.detach().numpy()).max() <= FWD_TOL


def test_r3b_ga_post_head_add_instant():
    case = GA_CASES[0]
    organ, X, y = _mk_ga(case)
    organ.head_add(0)
    for _ in range(15):
        organ.train_step(X, y)
    P, heads = transplant_ga(organ)
    for Xs in _probes(organ, case["d_in"]):
        ours = _fwd_ours(organ, Xs)
        ref = twin_ga_forward(P, heads, organ.L, _t(Xs))
        assert np.abs(ours -
                      ref.detach().numpy()).max() <= FWD_TOL


def test_r3c_transformer_post_grow_site_instant():
    case = TR_CASES[0]
    organ, X, y = _mk_tr(case)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        organ.grow_site("layer0/ffn[0]", hidden=4)
    for _ in range(15):
        organ.train_step(X, y)
    extra = sum(int(np.asarray(organ._bk.to_numpy(t)).size)
                for site in organ._port_sites.values()
                for slot in site.bodies
                for t in (slot["body"].W1, slot["body"].b1,
                          slot["body"].W2, slot["body"].c,
                          slot["A"]))
    P = transplant_std(organ, extra_elems=extra)
    ports = {}
    for l, site in getattr(organ, "_port_sites", {}).items():
        rows = []
        for slot in site.bodies:
            b = slot["body"]
            body_P = {"W1": _t(b._bk.to_numpy(b.W1)),
                      "b1": _t(b._bk.to_numpy(b.b1)),
                      "W2": _t(b._bk.to_numpy(b.W2)),
                      "c": _t(b._bk.to_numpy(b.c))}
            rows.append((body_P, _t(slot["A"])))
        ports[l] = rows
    for Xs in _probes(organ, case["d_in"]):
        ours = _fwd_ours(organ, Xs)
        ref = twin_std_forward(P, organ.L, organ.h, _t(Xs),
                               ports=ports, m_ffn=organ.m)
        assert np.abs(ours -
                      ref.detach().numpy()).max() <= FWD_TOL


@pytest.mark.parametrize("mk,twin", [
    ("tr", None), ("ga", None)], ids=["tr-dist", "ga-dist"])
def test_r3d_dist_instants_forward_and_nll_grads(mk, twin):
    if mk == "tr":
        organ, X, y = _mk_tr(TR_CASES[0], mode="numeric_dist")
        P = transplant_std(organ, grad=True)
        logits_fn = lambda Xs: twin_std_forward(  # noqa: E731
            P, organ.L, organ.h, Xs)
        heads = None
    else:
        organ, X, y = _mk_ga(GA_CASES[0], mode="numeric_dist")
        P, heads = transplant_ga(organ, grad=True)
        logits_fn = lambda Xs: twin_ga_forward(  # noqa: E731
            P, heads, organ.L, Xs)
    Xs_np = _np(organ._bk.to_numpy(organ._stdx(X)))
    # forward: the 2-column (mu, log-v) head
    ours = _fwd_ours(organ, Xs_np)
    ref = logits_fn(_t(Xs_np))
    assert np.abs(ours - ref.detach().numpy()).max() <= FWD_TOL
    # NLL gradient referee (kernel formula from torch ops;
    # clamp +-10.0 == engine NLL_CLAMP; torch clamp's zero
    # outside-gradient == the kernel's masked subgradient)
    G_ours, Gh_ours = _our_grads_sgd(organ, X, y,
                                     heads=(mk == "ga"))
    t = _t(_std_targets(organ, y))
    z = logits_fn(_t(Xs_np))
    mu, v = z[:, 0], torch.clamp(z[:, 1], -10.0, 10.0)
    loss = (0.5 * (v + (t - mu) ** 2 * torch.exp(-v))).mean()
    loss.backward()
    for k, p_ in P.items():
        assert _rel(G_ours[k],
                    p_.grad.detach().numpy()) <= GRAD_TOL, k
    if heads is not None:
        for l, layer in enumerate(heads):
            for h, HS in enumerate(layer):
                for nm, p_ in HS.items():
                    assert _rel(Gh_ours[l][h][nm],
                                p_.grad.detach().numpy()) \
                        <= GRAD_TOL, (l, h, nm)


def test_r3e_transformer_plus_post_widen_instant():
    case = dict(TR_CASES[0])
    organ, X, y = _mk_tr(case, cls=TransformerPlus)
    organ.widen_ffn(2)
    for _ in range(15):
        organ.train_step(X, y)
    P = transplant_std(organ)
    for Xs in _probes(organ, case["d_in"]):
        ours = _fwd_ours(organ, Xs)
        ref = twin_std_forward(P, organ.L, organ.h, _t(Xs))
        assert np.abs(ours -
                      ref.detach().numpy()).max() <= FWD_TOL


def test_r3f_attention_body_forward_and_grad():
    body = AttentionBody(3, d_model=8, n_layers=1, n_heads=2,
                         ffn=8, seed=4, zero_out=False,
                         out_width=2)
    body._x_mu = body._bk.ingest(np.zeros(3))
    body._x_sd = body._bk.ingest(np.ones(3))
    body._y_mu, body._y_sd = 0.0, 1.0
    body._port_owned = True
    rng = np.random.default_rng(6)
    X = rng.normal(size=(12, 3))
    # forward referee (raw serve) — the standard-scheme twin
    P = {k: _t(body._bk.to_numpy(v)) for k, v in body.P.items()}
    _census(body, sum(int(v.numel()) for v in P.values()))
    ours = _np(body._bk.to_numpy(body.predict(X)))
    ref = twin_std_forward(P, body.L, body.h, _t(X),
                           head_names=("Wout", "bout"))
    assert np.abs(ours - ref.detach().numpy()).max() <= FWD_TOL
    # gradient referee through the PORT training entry:
    # loss = <raw, dU> so dL/dP is autograd's; our step delta
    # from train_from_grad(dU) equals it
    dU = rng.normal(size=(12, 2))
    b2 = copy.deepcopy(body)
    before = {k: _np(b2._bk.to_numpy(v)).copy()
              for k, v in b2.P.items()}
    eta = 1e-3
    b2.train_from_grad(X, dU, sgd_lr=eta)
    G_ours = {k: (before[k] - _np(b2._bk.to_numpy(b2.P[k])))
              / eta for k in before}
    Pg = {k: _t(body._bk.to_numpy(v), grad=True)
          for k, v in body.P.items()}
    raw = twin_std_forward(Pg, body.L, body.h, _t(X),
                           head_names=("Wout", "bout"))
    loss = (raw * _t(dU)).sum()
    loss.backward()
    for k, p_ in Pg.items():
        assert _rel(G_ours[k],
                    p_.grad.detach().numpy()) <= GRAD_TOL, k


# =====================================================
# R-2c/R-2d: review additions — the causal training path and
# the CATEGORICAL head (mode coverage totality)
# =====================================================

def test_r2c_sequence_causal_gradients_vs_autograd():
    o = SequenceSubstrate(1, 8, mode="numeric", d_model=8,
                          n_layers=1, n_heads=1, seed=5)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(16, 4, 1))
    y = X[:, -1, 0] + 0.5
    for _ in range(20):
        o.train_step(X, y)
    G_ours, _ = _our_grads_sgd(o, X, y)
    P = transplant_std(o, grad=True)
    Xs = _t(_np(o._bk.to_numpy(o._stdx(X))))
    ys = _t(_std_targets(o, y))
    logits = twin_std_forward(P, o.L, o.h, Xs, causal=True)
    loss = ((logits.reshape(-1) - ys) ** 2).mean()
    loss.backward()
    for k, v in P.items():
        assert _rel(G_ours[k],
                    v.grad.detach().numpy()) <= GRAD_TOL, k


@pytest.mark.parametrize("host", ["tr", "ga"])
def test_r2d_categorical_head_gradients_vs_autograd(host):
    d_in = 3
    rng = np.random.default_rng(8)
    X = rng.normal(size=(30, d_in))
    labels = np.where(X[:, 0] > 0, "hi", "lo")
    if host == "tr":
        o = TransformerSubstrate(d_in, 8, mode="categorical",
                                 vocab=["hi", "lo"], d_model=8,
                                 n_layers=1, n_heads=1, seed=3)
    else:
        o = GrowableAttentionSubstrate(
            d_in, 8, mode="categorical", vocab=["hi", "lo"],
            d_model=8, n_layers=1, heads_spec=[[2]], seed=3)
    for _ in range(20):
        o.train_step(X, labels)
    G_ours, Gh_ours = _our_grads_sgd(o, X, labels,
                                     heads=(host == "ga"))
    if host == "tr":
        P = transplant_std(o, grad=True)
        logits = twin_std_forward(
            P, o.L, o.h,
            _t(_np(o._bk.to_numpy(o._stdx(X)))))
        heads = None
    else:
        P, heads = transplant_ga(o, grad=True)
        logits = twin_ga_forward(
            P, heads, o.L,
            _t(_np(o._bk.to_numpy(o._stdx(X)))))
    idx = torch.tensor([o.vocab.index(v) for v in labels])
    # torch OFFICIAL cross-entropy == our softmax-CE formula
    loss = TF.cross_entropy(logits, idx)
    loss.backward()
    for k, v in P.items():
        assert _rel(G_ours[k],
                    v.grad.detach().numpy()) <= GRAD_TOL, k
    if heads is not None:
        for l, layer in enumerate(heads):
            for h, HS in enumerate(layer):
                for nm, v in HS.items():
                    assert _rel(Gh_ours[l][h][nm],
                                v.grad.detach().numpy()) \
                        <= GRAD_TOL, (l, h, nm)


# =====================================================
# R-5: PARAMETER-TRAJECTORY referee (60C U-2) — k co-evolved
# SGD steps; after EVERY step, EVERY tensor, EVERY entry
# =====================================================

def _sgd_twin_step(leaves, loss, eta):
    loss.backward()
    with torch.no_grad():
        for t in leaves:
            t -= eta * t.grad
            t.grad = None


@pytest.mark.parametrize("host,case_i", [
    ("tr", 0), ("tr", 1), ("ga", 0), ("ga", 1)],
    ids=["tr0", "tr1", "ga0", "ga1"])
def test_r5_parameter_trajectory_every_coefficient(host,
                                                   case_i):
    eta, K = 1e-2, 5
    if host == "tr":
        organ, X, y = _mk_tr(TR_CASES[case_i])
        P = transplant_std(organ, grad=True)
        heads = None
        leaves = list(P.values())
    else:
        organ, X, y = _mk_ga(GA_CASES[case_i])
        P, heads = transplant_ga(organ, grad=True)
        leaves = list(P.values()) + [t for layer in heads
                                     for HS in layer
                                     for t in HS.values()]
    o2 = copy.deepcopy(organ)
    Xs = _t(_np(organ._bk.to_numpy(organ._stdx(X))))
    ys = _t(_std_targets(organ, y))
    for step in range(K):
        o2.train_step(X, y, sgd_lr=eta)
        if host == "tr":
            logits = twin_std_forward(P, o2.L, o2.h, Xs)
        else:
            logits = twin_ga_forward(P, heads, o2.L, Xs)
        loss = ((logits.reshape(-1) - ys) ** 2).mean()
        _sgd_twin_step(leaves, loss, eta)
        # EVERY tensor, EVERY entry, at EVERY step
        for k in P:
            d = np.abs(_np(o2._bk.to_numpy(o2.P[k]))
                       - P[k].detach().numpy()).max()
            assert d <= TRAJ_TOL, (step, k, d)
        if heads is not None:
            for l, layer in enumerate(heads):
                for h, HS in enumerate(layer):
                    for nm, t in HS.items():
                        d = np.abs(_np(o2._bk.to_numpy(
                            getattr(o2.heads[l][h], nm)))
                            - t.detach().numpy()).max()
                        assert d <= TRAJ_TOL, (step, l, h,
                                               nm, d)


def test_r2e_sequence_causal_categorical_gradients():
    """Review addition (60C final review): the LAST cell of
    the mode x path matrix — causal + categorical."""
    o = SequenceSubstrate(1, 8, mode="categorical",
                          vocab=["up", "dn"], d_model=8,
                          n_layers=1, n_heads=1, seed=9)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(16, 4, 1))
    labels = np.where(X[:, -1, 0] > 0, "up", "dn")
    for _ in range(20):
        o.train_step(X, labels)
    G_ours, _ = _our_grads_sgd(o, X, labels)
    P = transplant_std(o, grad=True)
    logits = twin_std_forward(
        P, o.L, o.h, _t(_np(o._bk.to_numpy(o._stdx(X)))),
        causal=True)
    idx = torch.tensor([o.vocab.index(v) for v in labels])
    loss = TF.cross_entropy(logits, idx)
    loss.backward()
    # ADJUDICATED line for THIS box (60C s2c, second entry):
    # gradients here are tiny (~1e-4) and the delta-extraction
    # instrument's own floor is eps*|P|/(eta*max|G|) ~ 1e-9 —
    # observed agreement to 9 significant digits (8.3e-10 rel).
    # Instrument physics, not a math error.
    for k, v in P.items():
        assert _rel(G_ours[k],
                    v.grad.detach().numpy()) <= 1e-8, k


def test_r1d_ga_causal_forward_and_grads():
    """Final-sweep addition: the GA CAUSAL construction path
    (raw-constructible; no product substrate uses it, covered
    for constructor-surface totality)."""
    o = GrowableAttentionSubstrate(1, 8, mode="numeric",
                                   d_model=8, n_layers=1,
                                   heads_spec=[[2]], seed=5,
                                   causal=True)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(14, 4, 1))
    y = X[:, -1, 0] * 1.5
    for _ in range(20):
        o.train_step(X, y)
    P, heads = transplant_ga(o)
    Xs = _np(o._bk.to_numpy(o._stdx(X[:6])))
    ours = _fwd_ours(o, Xs)
    # causal GA twin: official primitives + additive -1e9 mask
    Xt = _t(Xs)
    d = P["Wv"].shape[1]
    Tlen = Xt.shape[1]
    T = Xt @ P["Wv"] + P["Bf"][None, :Tlen, :]
    for l in range(o.L):
        Tn = TF.layer_norm(T, (d,), P[f"g1_{l}"],
                           P[f"b1n_{l}"], eps=1e-5)
        O = torch.zeros_like(T)
        mask = _t(np.triu(np.full((Tlen, Tlen), -1e9), k=1))
        for HS in heads[l]:
            S = (Tn @ HS["Wq"]) @ (Tn @ HS["Wk"]).transpose(
                1, 2) + mask
            A = torch.softmax(S, dim=-1)
            O = O + (A @ (Tn @ HS["Wv"])) @ HS["Wo"]
        T1 = T + O
        Tn2 = TF.layer_norm(T1, (d,), P[f"g2_{l}"],
                            P[f"b2n_{l}"], eps=1e-5)
        H = TF.gelu(Tn2 @ P[f"W1_{l}"] + P[f"b1_{l}"],
                    approximate="tanh")
        T = T1 + (H @ P[f"W2_{l}"] + P[f"b2_{l}"])
    ref = (T[:, -1, :] @ P["Wh"] + P["bh"]).detach().numpy()
    assert np.abs(ours - ref).max() <= FWD_TOL


def test_r3g_ga_post_head_widen_instant():
    """Final-sweep addition: the head_widen grown instant."""
    case = GA_CASES[0]
    organ, X, y = _mk_ga(case)
    organ.head_widen(0, 0, m=2)
    for _ in range(15):
        organ.train_step(X, y)
    P, heads = transplant_ga(organ)
    for Xs in _probes(organ, case["d_in"]):
        ours = _fwd_ours(organ, Xs)
        ref = twin_ga_forward(P, heads, organ.L, _t(Xs))
        assert np.abs(ours -
                      ref.detach().numpy()).max() <= FWD_TOL
