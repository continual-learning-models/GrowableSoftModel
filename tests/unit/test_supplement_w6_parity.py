"""W6 (58 v1.3, items 11-13): backend rows + host parity +
autonomy-inventory walk. PURE TEST — expected green; a RED is
a real defect and stops the item (defect protocol).

 T-9  backend rows: layer-insertion preservation and artifact
      round-trip re-proven per backend (torch-cpu-f64 1e-8,
      mps-f32 2e-3, skip-if-unavailable) on the reference
      Network AND the GA host — the B-1/B-2 claims per
      backend, not assumed from numpy.
 T-10 host parity: trial / shrink-by-rollback governance /
      forced override IN FULL on GrowableAttentionSubstrate
      (values, not smoke); the loud host-removal refusal.
 T-11 autonomy walk (C-5): every automatic mechanism's policy
      key has a PARAMETER_REFERENCE row (doc parsed).
"""
import pickle   # safe: locally produced model states only
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from engine.backends import resolve_backend                 # noqa: E402
from reference_net.net import Network                       # noqa: E402
from reference_net.growth_store import (                    # noqa: E402
    rollback, snapshot, trial)
from reference_net.method.gates import propose, run_plan    # noqa: E402
from reference_net.growthpolicy import \
    DEFAULT_GROWTH_POLICY as GP                             # noqa: E402
from core.substrates.growable_attention import \
    GrowableAttentionSubstrate                              # noqa: E402

# doc 32 9.6 device/tolerance matrix (skip-if-unavailable)
DEVICES = [("numpy", None, None, 0.0)]
try:
    import torch
    DEVICES.append(("torch", "cpu", "float64", 1e-8))
    if torch.backends.mps.is_available():
        DEVICES.append(("torch", "mps", "float32", 2e-3))
    if torch.cuda.is_available():
        DEVICES.append(("torch", "cuda", "float32", 2e-3))
except ImportError:
    pass

_IDS = [f"{n}-{d}-{t}" for n, d, t, _ in DEVICES]


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _np(net, a):
    return np.asarray(net._bk.to_numpy(a), dtype=np.float64)


def _err(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.abs(a - b).max()
                 / max(1.0, np.abs(a).max()))


# ---------------- T-9 backend rows ----------------

@pytest.mark.parametrize("name,dev,dt,tol", DEVICES, ids=_IDS)
def test_t9_network_deepen_preservation_per_backend(
        name, dev, dt, tol):
    """B-1 claim per backend: deepen (global AND scoped) is
    preservation-exact at birth within the row tolerance
    (bitwise on the numpy judge)."""
    bk = resolve_backend(name, dev, dt)
    net = Network(3, 8, lr=1e-2, seed=7, backend=bk)
    X, y = _data()
    for _ in range(30):
        net.train_step(X, y)
    base = _np(net, net.predict(X))
    net.deepen()
    net.deepen(scope=[0, 2, 5])
    out = _np(net, net.predict(X))
    if tol == 0.0:
        assert np.array_equal(out, base)      # judge: bitwise
    else:
        assert _err(out, base) <= tol
    net.train_step(X, y)                      # and it trains


@pytest.mark.parametrize("name,dev,dt,tol", DEVICES, ids=_IDS)
def test_t9_ga_insert_layer_preservation_per_backend(
        name, dev, dt, tol):
    """B-2 claim per backend: whole-layer insertion preserves
    the served function at birth; artifact round-trip serves
    identically on reload."""
    bk = resolve_backend(name, dev, dt)
    ga = GrowableAttentionSubstrate(
        3, 8, mode="numeric", d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]], backend=bk)
    X, y = _data(0, 6, 3)
    for _ in range(5):
        ga.train_step(X, y)
    base = np.asarray(ga.predict(X), dtype=np.float64)
    ga.insert_layer(1)
    out = np.asarray(ga.predict(X), dtype=np.float64)
    if tol == 0.0:
        assert np.array_equal(out, base)
    else:
        assert _err(out, base) <= tol
    twin = pickle.loads(pickle.dumps(ga))     # round-trip
    back = np.asarray(twin.predict(X), dtype=np.float64)
    assert _err(back, out) <= max(tol, 0.0) \
        if tol else np.array_equal(back, out)
    twin.train_step(X, y)


# ---------------- T-10 host parity ----------------

def _ga(seed=99):
    ga = GrowableAttentionSubstrate(
        3, 8, mode="numeric", d_model=8, n_layers=2,
        seed=seed, heads_spec=[[4, 4], [4, 4]])
    X, y = _data(0, 6, 3)
    for _ in range(5):
        ga.train_step(X, y)
    return ga, X, y


def test_t10_trial_on_host_values():
    """FR-15 on GA: budgeted probe reports real losses and the
    host comes back BITWISE (values compared, not smoke)."""
    ga, X, y = _ga()
    out0 = np.asarray(ga.predict(X), dtype=np.float64).copy()
    p0 = ga.n_params()
    rep = trial(ga, lambda h: h.insert_layer(1), X, y,
                budget_steps=3)
    assert rep["steps_run"] == 3
    assert len(rep["losses"]) == 3
    assert all(np.isfinite(v) for v in rep["losses"])
    assert rep["wall_ms"] > 0
    assert ga.n_params() == p0                # rolled back
    assert np.array_equal(
        np.asarray(ga.predict(X), dtype=np.float64), out0)


def test_t10_shrink_by_rollback_governance_on_host():
    """FR-18 host path: auto-snapshot -> rollback bitwise ->
    rollback record in growth_events? Hosts carry no gain
    ledger — the record home check + the LOUD removal
    refusal (D-7.2 exact message)."""
    ga, X, y = _ga(seed=141)
    ga.insert_layer(1)                        # auto-snapshot ON
    rec = ga._snapshots[-1]
    assert rec["tag"] == "auto:pre-growth"
    out_pre = None
    twin = pickle.loads(rec["blob"])
    out_pre = np.asarray(twin.predict(X), dtype=np.float64)
    rollback(ga, rec)
    assert np.array_equal(
        np.asarray(ga.predict(X), dtype=np.float64), out_pre)
    # loud capability refusal, exact wording (58 D-7.2)
    with pytest.raises(ValueError,
                       match=r"has no remove_block; host "
                             r"shrink = snapshot rollback "
                             r"\(FR-18\)"):
        propose(ga, "remove_block", GP, k=0)
    plan = {"steps": [{"rule": "schedule",
                       "move": "remove_block",
                       "args": {"k": 0}}]}
    with pytest.raises(ValueError, match="has no remove_block"):
        run_plan(ga, plan, GP, X, y, steps_between=1)


def test_t10_forced_override_on_host_full():
    """C-5 on GA: gate refuse -> forced grow_site executes,
    record carries forced=True and the four-spec bundle."""
    ga, X, y = _ga(seed=143)
    ga._growth_policy = dict(GP, gate_seam_min_width=3,
                             gate_seam_mode="refuse")
    with pytest.raises(ValueError, match="G-SEAM"):
        ga.grow_site("layer1/ffn[1]", hidden=4)
    ga.grow_site("layer1/ffn[1]", hidden=4, force=True)
    rec = [e for e in ga.growth_events
           if e["event"] == "grow_site"][-1]
    assert rec["forced"] is True
    assert rec["n_params"] == 49              # hand count
    assert set(rec["specs"]) == {"structure", "wiring",
                                 "placement", "birth"}
    ga.train_step(X, y)                       # and it trains


# ---------------- T-11 autonomy walk ----------------

def test_t11_autonomy_inventory_documented():
    """C-5: every automatic mechanism's policy key has a
    PARAMETER_REFERENCE row. REGISTRY = auto-snapshot (2),
    G-DEEPEN (1), gate family (6)."""
    registry = ("growth_auto_snapshot", "growth_snapshot_keep",
                "gate_deepen_mode",
                "gate_seam_min_width", "gate_seam_mode",
                "gate_nest_mode",
                "gate_scope_min_width", "gate_scope_mode",
                "gate_widen_mode")
    doc = (REPO / "docs" / "PARAMETER_REFERENCE.md").read_text()
    rows = [ln for ln in doc.splitlines()
            if ln.startswith("| ")]
    for key in registry:
        assert any(f"| {key} " in ln for ln in rows), \
            f"undocumented automatic-mechanism key: {key}"


def test_t10b_tr_host_grow_site_governance_parity():
    """D-W7-1 regression (the W5 text-replace silently missed
    the TR host's flat grow_site path): BOTH hosts must show
    identical governance on grow_site — auto-snapshot,
    four-spec record with wall_ms, forced marker."""
    from core.substrates.transformer import (
        TransformerSubstrate)
    X, y = _data(0, 6, 3)
    tr = TransformerSubstrate(3, 8, d_model=8,
                              n_layers=2, seed=97)
    for _ in range(5):
        tr.train_step(X, y)
    assert getattr(tr, "_snapshots", None) is None
    tr.grow_site("layer0/ffn[1]", hidden=4)
    assert len(tr._snapshots) == 1               # FR-13
    assert tr._snapshots[-1]["tag"] == "auto:pre-growth"
    rec = [e for e in tr.growth_events
           if e["event"] == "grow_site"][-1]
    assert rec["n_params"] == 49                 # hand count
    assert rec["wall_ms"] > 0                    # FR-16
    assert set(rec["specs"]) == {"structure", "wiring",
                                 "placement", "birth"}
    assert "forced" not in rec
    tr.grow_site("layer1/ffn[1]", hidden=4, force=True)
    rec2 = [e for e in tr.growth_events
            if e["event"] == "grow_site"][-1]
    assert rec2.get("forced") is True            # C-5


# ====== T-B7/B8/B10 (59 v1.3 R-2): host + backend rows ======

import copy as _copy


def test_tb7_ga_host_half_speed_judge():
    """R-2.1 on GA: {"layer:0": 0.5, "head:0:0": 0.5} -> those
    tensors land exactly at old + 0.5*(full-old); layer 1 and
    "out"/"embed" tensors bitwise equal the twin's."""
    ga, X, y = _ga(seed=151)
    twin = _copy.deepcopy(ga)
    old_W1 = np.asarray(ga._bk.to_numpy(ga.P["W1_0"])).copy()
    HS = ga.heads[0][0]
    old_Wq = np.asarray(ga._bk.to_numpy(HS.Wq)).copy()
    twin.train_step(X, y)
    ga._growth_policy = dict(GP, train_lr_scales={
        "layer:0": 0.5, "head:0:0": 0.5})
    ga.train_step(X, y)
    f_W1 = np.asarray(twin._bk.to_numpy(twin.P["W1_0"]))
    assert np.array_equal(
        np.asarray(ga._bk.to_numpy(ga.P["W1_0"])),
        old_W1 + 0.5 * (f_W1 - old_W1))
    f_Wq = np.asarray(twin._bk.to_numpy(twin.heads[0][0].Wq))
    assert np.array_equal(
        np.asarray(ga._bk.to_numpy(ga.heads[0][0].Wq)),
        old_Wq + 0.5 * (f_Wq - old_Wq))
    for k in ("W1_1", "Wh", "bh", "Bf", "Wv"):    # untouched
        assert np.array_equal(
            np.asarray(ga._bk.to_numpy(ga.P[k])),
            np.asarray(twin._bk.to_numpy(twin.P[k]))), k
    # head at layer 1: full speed too
    assert np.array_equal(
        np.asarray(ga._bk.to_numpy(ga.heads[1][0].Wq)),
        np.asarray(twin._bk.to_numpy(twin.heads[1][0].Wq)))


def test_tb8_tr_host_zero_still_and_refusal():
    """R-2.6/R-2.3b on TR: layer-1 frozen bitwise over 3
    steps while layer-0/out move; typo refused loudly
    pre-mutation."""
    from core.substrates.transformer import TransformerSubstrate
    X, y = _data(0, 6, 3)
    tr = TransformerSubstrate(3, 8, d_model=8, n_layers=2,
                              seed=157)
    for _ in range(5):
        tr.train_step(X, y)
    tr._growth_policy = dict(GP, train_lr_scales={"layer:1":
                                                  0.0})
    l1 = {k: np.asarray(tr._bk.to_numpy(tr.P[k])).copy()
          for k in tr.P if k.endswith("_1")}
    l0_W1 = np.asarray(tr._bk.to_numpy(tr.P["W1_0"])).copy()
    for _ in range(3):
        tr.train_step(X, y)
    for k, v in l1.items():
        assert np.array_equal(
            np.asarray(tr._bk.to_numpy(tr.P[k])), v), k
    assert not np.array_equal(
        np.asarray(tr._bk.to_numpy(tr.P["W1_0"])), l0_W1)
    # typo refusal, loud and pre-mutation
    tr._growth_policy = dict(GP, train_lr_scales={"layr:0":
                                                  0.5})
    snap = {k: np.asarray(tr._bk.to_numpy(tr.P[k])).copy()
            for k in tr.P}
    with pytest.raises(ValueError, match="layr:0"):
        tr.train_step(X, y)
    for k, v in snap.items():
        assert np.array_equal(
            np.asarray(tr._bk.to_numpy(tr.P[k])), v), k


@pytest.mark.parametrize("name,dev,dt,tol",
                         [d for d in DEVICES
                          if d[0] == "torch"],
                         ids=[i for i in _IDS
                              if i.startswith("torch")])
def test_tb10_backend_rows_half_speed(name, dev, dt, tol):
    """R-2 backend row: the half-speed identity on torch
    backends (cpu-f64 bitwise via to_numpy; mps-f32 ratio to
    2e-3)."""
    from reference_net.net import Network
    bk = resolve_backend(name, dev, dt)
    net = Network(3, 8, lr=1e-2, seed=7, backend=bk)
    X, y = _data()
    for _ in range(5):
        net.train_step(X, y)
    twin = _copy.deepcopy(net)      # stays on the device
    old = np.asarray(bk.to_numpy(net.W1),
                     dtype=np.float64).copy()
    twin.train_step(X, y)
    net._growth_policy = dict(GP,
                              train_lr_scales={"encoder": 0.5})
    net.train_step(X, y)
    got = np.asarray(bk.to_numpy(net.W1), dtype=np.float64)
    full = np.asarray(bk.to_numpy(twin.W1), dtype=np.float64)
    expect = old + 0.5 * (full - old)
    if tol <= 1e-8:
        assert np.array_equal(got, expect)
    else:
        num = np.abs(got - old).sum()
        den = max(np.abs(full - old).sum(), 1e-30)
        assert abs(num / den - 0.5) <= tol


def test_tb13_host_sgd_branch_scaling():
    """R-2.5 on the HOSTS (audit find: the host sgd write-back
    was implemented but未 boxed): half-speed identity on the
    sgd path for TR layer:0 and a GA head."""
    from core.substrates.transformer import TransformerSubstrate
    X, y = _data(0, 6, 3)
    tr = TransformerSubstrate(3, 8, d_model=8, n_layers=2,
                              seed=161)
    for _ in range(3):
        tr.train_step(X, y)
    tw = _copy.deepcopy(tr)
    old = {k: np.asarray(tr._bk.to_numpy(tr.P[k])).copy()
           for k in tr.P if k.endswith("_0")}
    tw.train_step(X, y, sgd_lr=1e-2)
    tr._growth_policy = dict(GP, train_lr_scales={"layer:0":
                                                  0.5})
    tr.train_step(X, y, sgd_lr=1e-2)
    for k, v in old.items():
        f = np.asarray(tw._bk.to_numpy(tw.P[k]))
        assert np.array_equal(
            np.asarray(tr._bk.to_numpy(tr.P[k])),
            v + 0.5 * (f - v)), k
    ga, X, y = _ga(seed=163)
    tw = _copy.deepcopy(ga)
    old_Wq = np.asarray(ga._bk.to_numpy(
        ga.heads[0][0].Wq)).copy()
    tw.train_step(X, y, sgd_lr=1e-2)
    ga._growth_policy = dict(GP, train_lr_scales={"head:0:0":
                                                  0.5})
    ga.train_step(X, y, sgd_lr=1e-2)
    f_Wq = np.asarray(tw._bk.to_numpy(tw.heads[0][0].Wq))
    assert np.array_equal(
        np.asarray(ga._bk.to_numpy(ga.heads[0][0].Wq)),
        old_Wq + 0.5 * (f_Wq - old_Wq))
    # untouched head at the same layer stays bitwise on twin
    assert np.array_equal(
        np.asarray(ga._bk.to_numpy(ga.heads[0][1].Wq)),
        np.asarray(tw._bk.to_numpy(tw.heads[0][1].Wq)))


# ===== G2/G3/G4/G6 (doc 61 I-B; pure test, expected green) ==

def _tr(seed=99, n_layers=2):
    from core.substrates.transformer import TransformerSubstrate
    tr = TransformerSubstrate(3, 8, mode="numeric", d_model=8,
                              n_layers=n_layers, n_heads=2,
                              seed=seed)
    X, y = _data(0, 6, 3)
    for _ in range(5):
        tr.train_step(X, y)
    return tr, X, y


def test_g2_tr_monitor_and_getstate_guard():
    """G2: monitor on TransformerSubstrate — cadence=5
    window=3, 26 steps -> ring length 3, last step %5==0,
    off-by-default; AND pickle bytes identical with vs
    without populated _snapshots/_monitor (byte-guard).
    [origin: 58 T-6 said "a host"; GA-only box; I-6]"""
    from reference_net.instrument import monitor_configure
    from reference_net.growth_store import snapshot
    tr, X, y = _tr(seed=137)
    assert getattr(tr, "_monitor", None) is None   # off
    monitor_configure(tr, cadence=5, window=3)
    for _ in range(26):
        tr.train_step(X, y)
    recs = tr._monitor["records"]
    assert len(recs) == 3                          # ring
    assert recs[-1]["step"] % 5 == 0               # cadence
    tr2, _, _ = _tr(seed=139)
    b1 = pickle.dumps(tr2)
    snapshot(tr2, tag="x")
    monitor_configure(tr2, cadence=5, window=2)
    b2 = pickle.dumps(tr2)
    assert b1 == b2                # observer state invisible
    twin = pickle.loads(b2)
    assert getattr(twin, "_snapshots", None) is None
    assert getattr(twin, "_monitor", None) is None


@pytest.mark.parametrize("name,dev,dt,tol", DEVICES, ids=_IDS)
def test_t9_tr_insert_layer_preservation_per_backend(
        name, dev, dt, tol):
    """G3 (judges corrected by IN-8): TR insert_layer
    preservation numpy bitwise / torch within the row
    tolerance; RELOAD serve numpy bitwise / torch within
    few-ulp (1e-12) — bitwise reload on device backends is
    NOT a valid expectation (family-wide F-B fact); TR
    predict via bk.to_numpy (F-A fact)."""
    from core.substrates.transformer import TransformerSubstrate
    bk = resolve_backend(name, dev, dt)
    tr = TransformerSubstrate(3, 8, mode="numeric", d_model=8,
                              n_layers=2, n_heads=2, seed=99,
                              backend=bk)
    X, y = _data(0, 6, 3)
    for _ in range(5):
        tr.train_step(X, y)
    base = _np(tr, tr.predict(X))
    tr.insert_layer(1)
    out = _np(tr, tr.predict(X))
    if tol == 0.0:
        assert np.array_equal(out, base)     # numpy bitwise
    else:
        assert _err(out, base) <= tol
    twin = pickle.loads(pickle.dumps(tr))    # round-trip
    back = np.asarray(twin._bk.to_numpy(twin.predict(X)),
                      dtype=np.float64)
    if tol == 0.0:
        assert np.array_equal(back, out)     # numpy bitwise
    else:                                    # per-precision
        reload_tol = 1e-12 if dt == "float64" else 1e-6
        assert _err(back, out) <= reload_tol
    twin.train_step(X, y)                    # serves + trains


def test_g4_host_body_nested_birth():
    """G4: grow_site -> the grown body's gain_ledger[0] is
    the nested_birth stamp with mouth=1 (value-exact); on
    GA AND TR. [origin: 58 D-5.2 stamps in BOTH presets;
    only the trunk path witnessed]"""
    import warnings as _w
    ga = GrowableAttentionSubstrate(
        3, 8, mode="numeric", d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]])
    X, y = _data(0, 6, 3)
    for _ in range(5):
        ga.train_step(X, y)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        ga.grow_site(ga.growth_sites()[0][0], hidden=4)
    for host in (ga,):
        pass
    tr, Xt, yt = _tr(seed=99)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        tr.grow_site(tr.growth_sites()[0][0], hidden=4)
    for host in (ga, tr):
        sites = host._port_sites
        site = (list(sites.values())[0]
                if isinstance(sites, dict) else sites[0])
        body = site.bodies[0]["body"]
        rec = body.gain_ledger[0]
        assert rec["event"] == "nested_birth", rec
        assert rec["mouth"] == 1, rec


def test_tb15_unclassified_key_loud_error():
    """G6 (route corrected by the direct probe): the region
    mapper called DIRECTLY — _lr_region_of("Zx") raises
    RuntimeError naming "Zx" and "unclassified"; legal keys
    map correctly (unit-level check of the R-2.3 promise)."""
    from core.substrates.transformer import TransformerSubstrate
    for cls in (TransformerSubstrate,
                GrowableAttentionSubstrate):
        with pytest.raises(RuntimeError) as ei:
            cls._lr_region_of("Zx")
        assert "Zx" in str(ei.value)
        assert "unclassified" in str(ei.value)
        assert cls._lr_region_of("Wh") == "out"
        assert cls._lr_region_of("Bf") == "embed"
        assert cls._lr_region_of("W1_0") == "layer:0"
