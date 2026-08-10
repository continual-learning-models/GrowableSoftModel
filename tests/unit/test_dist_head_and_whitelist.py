"""60A: uncertainty (dist) heads for the transformer and
growable_attention hosts + the whitelist anti-crash scheme.
Boxes written FIRST from the design text (strict TDD); RED
verified at the CURRENT failure signatures. Judges transcribed
from the GSM-I3 mlp boxes wherever the design says so."""
import json
import os
import pickle
import subprocess
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

from engine.backends import (set_compute_policy,     # noqa: E402
                             resolve_backend)
from core.facade import System                       # noqa: E402
from core.wiring import Config                       # noqa: E402
from core.substrates import REGISTRY                 # noqa: E402
from core.substrates.transformer import \
    TransformerSubstrate                             # noqa: E402
from core.substrates.growable_attention import \
    GrowableAttentionSubstrate                       # noqa: E402
from core.substrates.sequence import \
    SequenceSubstrate                                # noqa: E402
from core.substrates.mlp import MLPSubstrate         # noqa: E402

RNG = np.random.default_rng(0)

HOSTS = [
    ("transformer",
     lambda mode: TransformerSubstrate(
         3, 8, mode=mode, d_model=8, n_layers=1, n_heads=1,
         seed=3)),
    ("growable_attention",
     lambda mode: GrowableAttentionSubstrate(
         3, 8, mode=mode, d_model=8, n_layers=1,
         heads_spec=[[1]], seed=3)),
]
HOST_IDS = [h[0] for h in HOSTS]

ATT_SP = {"d_model": 8, "n_layers": 1, "heads_spec": [[1]],
          "seed": 3}
TR_SP = {"d_model": 8, "n_layers": 1, "n_heads": 1, "seed": 3}


def _homo_data(n=200, eps=0.3, y_scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (2.0 * X[:, 0] - X[:, 1]
         + eps * rng.normal(size=n)) * y_scale
    return X, y


def _hetero_data(n=400, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    noise = np.where(X[:, 0] > 0, 1.0, 0.1)
    y = 2.0 * X[:, 0] - X[:, 1] + noise * rng.normal(size=n)
    return X, y


ROWS = [{"input": {"a": float(x[0]), "b": float(x[1]),
                   "c": float(x[2])}, "target": float(t)}
        for x, t in zip(*_homo_data(48))]

SEQ_ROWS = [{"input": [[float(i + j)] for j in range(4)],
             "target": float(i + 4)} for i in range(24)]


def _sys(tmp_path):
    return System(Config.from_env(backend="mlp",
                                  models_root=tmp_path / "ws"))


@pytest.fixture(autouse=True)
def _numpy_judge():
    set_compute_policy("numpy", "cpu", None)
    yield
    set_compute_policy("numpy", "cpu", None)


# ------------- T-1 whitelist guard (L2) -------------

def test_t1a_sequence_plus_dist_refuses(tmp_path):
    s = _sys(tmp_path)
    out = s.create_model("q", holdout=SEQ_ROWS[:4],
                         policy={"numeric_head": "dist"})
    assert out.get("substrate") == "sequence", out
    r = s.study("q", SEQ_ROWS, steps=2)
    assert isinstance(r, dict) and "refusal" in r, r
    assert "dist" in r["refusal"]


def test_t1b_undeclared_substrate_refuses_not_crashes(
        tmp_path, monkeypatch):
    class BareVector(MLPSubstrate):
        NAME = "barevec"
        # no dist declaration of its own beyond the default
    BareVector.SUPPORTED_HEADS = ("point",)
    monkeypatch.setitem(REGISTRY, "barevec", BareVector)
    s = _sys(tmp_path)
    s.create_model("b", substrate="barevec",
                   policy={"numeric_head": "dist"})
    r = s.study("b", ROWS, steps=2)
    assert isinstance(r, dict) and "refusal" in r, r
    assert "dist" in r["refusal"] and "barevec" in r["refusal"]
    assert "point" in r["refusal"]           # names supported


def test_t1d_teach_lane_door_and_sequence_ctor(tmp_path,
                                               monkeypatch):
    """Review additions: (a) the teach/factory birth lane
    carries the SAME L2 whitelist check (it was a second door
    without one; note — that lane currently ALWAYS builds its
    class from factory.substrate_name, recorded separately in
    doc 61 as the teach substrate-fidelity finding); (b) the
    sequence host constructor enforces ITS OWN whitelist (it
    inherits the transformer ctor which now accepts
    numeric_dist)."""
    class BareVector(MLPSubstrate):
        NAME = "barevec2"
    BareVector.SUPPORTED_HEADS = ("point",)
    monkeypatch.setitem(REGISTRY, "barevec2", BareVector)
    s = _sys(tmp_path)
    s.create_model("b", policy={"numeric_head": "dist"})
    s.f.trainer.substrate_name = "barevec2"   # the lane's
    #                                           class knob
    with pytest.raises(ValueError, match="dist"):
        s.f.teach("b", list(ROWS))      # the door fires
    # sequence ctor: own-whitelist enforcement
    with pytest.raises(ValueError, match="sequence"):
        SequenceSubstrate(1, 8, mode="numeric_dist")


def test_t1c_whitelist_declarations():
    """The L2 declarations themselves, incl. the inheritance
    trap: sequence subclasses transformer and must OVERRIDE."""
    assert "dist" in MLPSubstrate.SUPPORTED_HEADS
    assert "dist" in TransformerSubstrate.SUPPORTED_HEADS
    assert "dist" in GrowableAttentionSubstrate.SUPPORTED_HEADS
    assert SequenceSubstrate.SUPPORTED_HEADS == ("point",)


# ------------- T-2 constructor whitelist (L3) -------------

@pytest.mark.parametrize("name,mk", HOSTS, ids=HOST_IDS)
def test_t2_constructor_refuses_unknown_mode(name, mk):
    with pytest.raises(ValueError) as e:
        mk("bogus_mode")
    assert "numeric" in str(e.value)         # names valid set


# ------------- T-3 birth honesty (mlp judge transcribed) ----

@pytest.mark.parametrize("name,mk", HOSTS, ids=HOST_IDS)
def test_t3_birth_zero_head_and_first_loss_half(name, mk):
    org = mk("numeric_dist")
    assert np.all(np.asarray(
        org._bk.to_numpy(org.P["Wh"])) == 0.0)
    assert np.all(np.asarray(
        org._bk.to_numpy(org.P["bh"])) == 0.0)
    X, y = _homo_data()
    first = org.train_step(X, y)
    assert abs(first - 0.5) < 1e-6           # closed-form birth


# ------------- T-4 training: NLL falls, sigma finite --------

def _train(mk, X, y, steps=300):
    org = mk("numeric_dist")
    losses = [org.train_step(X, y) for _ in range(steps)]
    return org, losses


@pytest.mark.parametrize("name,mk", HOSTS, ids=HOST_IDS)
def test_t4_nll_falls_and_predict_matches_dist(name, mk):
    X, y = _homo_data()
    org, losses = _train(mk, X, y)
    assert losses[-1] < losses[0]
    value, std = org.predict_dist(X)
    assert np.isfinite(value).all() and (std > 0).all()
    # predict == dist value bitwise (mlp judge transcribed)
    assert np.array_equal(np.asarray(org.predict(X)).ravel(),
                          np.asarray(value).ravel())


# ------------- T-5 sigma accuracy (formula angle) -----------

@pytest.mark.parametrize("name,mk", HOSTS, ids=HOST_IDS)
def test_t5_calibration_band_and_hetero_shape(name, mk):
    eps = 0.3
    X, y = _homo_data(eps=eps)
    org, _ = _train(mk, X, y, steps=400)
    _, std = org.predict_dist(X)
    m = float(np.asarray(std).mean())
    assert 0.5 * eps < m < 2.0 * eps, m      # mlp band judge
    Xh, yh = _hetero_data()
    orgh, _ = _train(mk, Xh, yh, steps=400)
    _, sh = orgh.predict_dist(Xh)
    sh = np.asarray(sh)
    hi = float(sh[Xh[:, 0] > 0].mean())
    lo = float(sh[Xh[:, 0] <= 0].mean())
    assert hi > lo                            # noise ordering


# ------------- T-6/T-6b served chain, both arms -------------

@pytest.mark.parametrize("sub,sp", [
    ("transformer", TR_SP), ("growable_attention", ATT_SP)])
def test_t6_served_chain_working_and_committed(tmp_path, sub,
                                               sp):
    s = _sys(tmp_path)
    out = s.create_model("m", holdout=ROWS[:8],
                         substrate=sub,
                         policy={"numeric_head": "dist",
                                 "substrate_params": sp})
    assert "refusal" not in out, out
    r = s.study("m", ROWS, steps=30)
    assert "refusal" not in r, r
    probe = ROWS[0]["input"]
    dw = s.predict_dist("m", probe, working=True)
    assert dw["kind"] == "numeric_dist"
    assert np.isfinite(dw["value"]) and dw["std"] > 0
    c = s.commit("m", note="60a")
    assert c.get("promoted"), c
    dc = s.predict_dist("m", probe, working=False)
    assert dc["kind"] == "numeric_dist"
    assert np.isfinite(dc["value"]) and dc["std"] > 0


# ------------- T-7 backend parity (dist path) ---------------

@pytest.mark.parametrize("name,mk", HOSTS, ids=HOST_IDS)
def test_t7_torch_cpu_f64_parity(name, mk):
    try:
        import torch                          # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    X, y = _homo_data(n=60)
    org_n, _ = _train(mk, X, y, steps=40)
    vn, sn = org_n.predict_dist(X)
    bk = resolve_backend("torch", device="cpu",
                         dtype="float64")

    def mk_t(mode):
        if name == "transformer":
            return TransformerSubstrate(
                3, 8, mode=mode, d_model=8, n_layers=1,
                n_heads=1, seed=3, backend=bk)
        return GrowableAttentionSubstrate(
            3, 8, mode=mode, d_model=8, n_layers=1,
            heads_spec=[[1]], seed=3, backend=bk)
    org_t, _ = _train(mk_t, X, y, steps=40)
    vt, st_ = org_t.predict_dist(X)
    assert np.abs(np.asarray(vn) - np.asarray(vt)).max() < 1e-8
    assert np.abs(np.asarray(sn) - np.asarray(st_)).max() < 1e-8


# ------------- T-8 combination-matrix sweep (L5) ------------

def test_t8_matrix_every_cell_works_or_refuses(tmp_path):
    heads = ("point", "dist")
    vector_subs = sorted(n for n, c in REGISTRY.items()
                         if c.DATA_FORM == "vector")
    outcomes = {}
    for i, sub in enumerate(vector_subs):
        for head in heads:
            s = _sys(tmp_path / f"{sub}_{head}")
            pol = {"numeric_head": head}
            if sub in ("growable_attention",):
                pol["substrate_params"] = ATT_SP
            out = s.create_model("m", substrate=sub, policy=pol)
            if isinstance(out, dict) and "refusal" in out:
                outcomes[(sub, head)] = "refused"
                continue
            r = s.study("m", ROWS, steps=2)
            if isinstance(r, dict) and "refusal" in r:
                outcomes[(sub, head)] = "refused"
            else:
                assert "loss" in r, (sub, head, r)
                if head == "dist":
                    d = s.predict_dist("m", ROWS[0]["input"],
                                       working=True)
                    assert d["std"] > 0, (sub, head, d)
                outcomes[(sub, head)] = "works"
    # sequence rows of the matrix
    for head in heads:
        s = _sys(tmp_path / f"seq_{head}")
        out = s.create_model("q", holdout=SEQ_ROWS[:4],
                             policy={"numeric_head": head})
        assert out.get("substrate") == "sequence", out
        r = s.study("q", SEQ_ROWS, steps=2)
        outcomes[("sequence", head)] = (
            "refused" if isinstance(r, dict) and "refusal" in r
            else "works")
    # every cell landed in the two legal outcomes (no crash
    # survived to here); dist cells of the three main vector
    # substrates WORK; sequence+dist refuses
    for cell, res in outcomes.items():
        assert res in ("works", "refused"), (cell, res)
    for sub in ("mlp", "transformer", "growable_attention",
                "mlp_plus", "transformer_plus"):
        # the _plus variants INHERIT their parents' dist support
        assert outcomes[(sub, "dist")] == "works", sub
    assert outcomes[("sequence", "dist")] == "refused"
    assert outcomes[("sequence", "point")] == "works"


# ------------- T-9 CLI last line (L4) -------------

def test_t9_cli_structured_error_not_raw_crash(tmp_path):
    env = dict(os.environ,
               SOFTMODEL_MODELS_ROOT=str(tmp_path / "ws"))
    p = subprocess.run(
        [sys.executable, "-m", "cli.cli", "study",
         json.dumps({"model_id": "m", "examples": 123})],
        cwd=REPO, env=env, capture_output=True, text=True,
        timeout=120)
    assert p.returncode == 1
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert "error" in out                     # structured
    assert "Traceback" in p.stderr            # information kept


# ------------- T-10 growth x dist axis -------------

def test_t10_growth_verbs_on_dist_models(tmp_path):
    # mlp-dist: every growth verb lands in {works, loud
    # refusal}; deepen's existing loud refusal is EXPECTED
    s = _sys(tmp_path)
    s.create_model("m", policy={"numeric_head": "dist",
                                "max_params_mult": 50})
    s.study("m", ROWS, steps=20)
    r = s.deepen("m", m=4)
    assert "refusal" in r and "numeric" in r["refusal"]
    for verb in (lambda: s.widen("m", k=2),
                 lambda: s.grow("m", k_nodes=1, hidden=4)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = verb()
        assert isinstance(out, dict), out     # no crash
    # host-dist: insert_layer must WORK with (mu, sigma)
    # preserved bitwise at birth
    for sub, sp in (("transformer", TR_SP),
                    ("growable_attention", ATT_SP)):
        st = _sys(tmp_path / sub)
        st.create_model("h", substrate=sub,
                        policy={"numeric_head": "dist",
                                "substrate_params": sp,
                                "max_params_mult": 50})
        st.study("h", ROWS, steps=20)
        probe = ROWS[0]["input"]
        pre = st.predict_dist("h", probe, working=True)
        r2 = st.deepen("h")                   # whole layer
        assert "refusal" not in r2, (sub, r2)
        post = st.predict_dist("h", probe, working=True)
        assert pre["value"] == post["value"], sub
        assert pre["std"] == post["std"], sub
        # review addition: EVERY facade growth verb on the dist
        # host lands in {works, loud refusal} — never a crash
        for verb in (lambda: st.widen("h", k=2),
                     lambda: st.grow("h", k_nodes=1, hidden=4)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = verb()
            assert isinstance(out, dict), (sub, out)
    # mlp-dist loop verbs: same contract
    out = s.set_policy("m", growth_params={"loop_enabled": True})
    assert "refusal" not in out, out
    r_loop = s.loop("m")
    assert isinstance(r_loop, dict), r_loop
    if "refusal" not in r_loop:
        assert isinstance(s.remove_loop("m"), dict)


# ------------- T-11 artifact round-trip -------------

@pytest.mark.parametrize("name,mk", HOSTS, ids=HOST_IDS)
def test_t11_artifact_roundtrip_bitwise(name, mk):
    X, y = _homo_data(n=60)
    org, _ = _train(mk, X, y, steps=40)
    v0, s0 = org.predict_dist(X)
    # pickle is safe here: round-tripping an object THIS test
    # just built in-process (the repo's artifact doctrine and
    # the existing parity tests use the same idiom)
    clone = pickle.loads(pickle.dumps(org))
    v1, s1 = clone.predict_dist(X)
    assert np.array_equal(np.asarray(v0), np.asarray(v1))
    assert np.array_equal(np.asarray(s0), np.asarray(s1))
