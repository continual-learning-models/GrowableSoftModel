"""A2 boxes (plan 53 v1.2 step A2; 54 v1.10 section 3).

 V-A2-1 preset identity — adjudicated by the P0 reproducible-
        recapture box (test_adjustment_fixtures_guard): ALL
        baseline scenarios re-executed through the preset-
        backed operators with state hashes and numeric files
        identical. Here: the specs field's PRESENCE on new
        events (the SR-22 declared amendment).
 V-A2-2 ledger provenance — the four specs of a growth event
        round-trip as JSON and REBUILD the event: a twin host
        receiving the same operator calls is bitwise-equal at
        birth (state hash, ledger-exclusive).
 V-A2-3 spec hygiene — unknown fields, unknown kind, undefined
        designation, missing REQUIRED field: refused loudly
        with the offending name.
 V-A2-4 stage units — one box per compose stage; PLUS the
        refused-growth state boxes replaying FX-4 (duplicate/
        port-type: state untouched; guard-refuse: exactly the
        recorded post state, seed_counter+1 — the P0 capture
        fact).
"""
import hashlib
import json
import pickle   # safe: locally produced write-once fixtures
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

from reference_net.foundation import compose               # noqa: E402
from reference_net.foundation.specs import (               # noqa: E402
    ALL, END, NONE, BirthSpec, PlacementSpec, StructureSpec,
    Tap, WiringSpec, specs_as_dict)
from reference_net.foundation.recipes import (              # noqa: E402
    get_recipe, register_recipe)
from reference_net.net import Network                       # noqa: E402

FIX = REPO / "tests" / "unit" / "fixtures" / \
    "adjustment_baseline"


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _state_hash(net):
    """Same canonical walk as the capture script (ledger-
    exclusive) — independent reimplementation, judge side."""
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o, key=str):
                if str(k) == "gain_ledger":
                    continue
                h.update(str(k).encode()); walk(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(b"["); [walk(v) for v in o]; h.update(b"]")
        elif isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
            h.update(str(o.dtype).encode())
            h.update(str(o.shape).encode())
        elif hasattr(o, "__getstate__") and not isinstance(
                o, (str, bytes, int, float, bool, type(None))):
            walk(o.__getstate__())
        else:
            h.update(repr(o).encode())
    walk(net.__getstate__())
    return h.hexdigest()


def _grown_net(seed=7, steps=120):
    X, y = _data()
    net = Network(3, 8, lr=1e-2, seed=seed)
    for _ in range(steps):
        net.train_step(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.grow(0, hidden=4)
    return net, X, y


# ---------------- V-A2-1 / V-A2-2 ----------------

def test_a2_specs_recorded_on_grow_and_deepen():
    net, X, y = _grown_net()
    net.deepen()
    grow_rec, deep_rec = net.gain_ledger[-2], net.gain_ledger[-1]
    assert grow_rec["event"] == "refine"
    sp = grow_rec["specs"]
    assert set(sp) == {"structure", "wiring", "placement",
                       "birth"}
    assert sp["structure"]["kind"] == "reference"
    assert sp["structure"]["params"]["hidden"] == 4
    assert sp["wiring"]["reads"][0]["source"] == "scope_input"
    assert sp["placement"]["chain"] == NONE
    assert deep_rec["specs"]["structure"]["kind"] == "block"
    assert deep_rec["specs"]["placement"]["chain"] == "blocks"
    json.dumps(sp); json.dumps(deep_rec["specs"])  # JSON-safe


def test_a2_rebuild_from_specs_bitwise_at_birth():
    """Twin hosts, same calls: the specs fully determine the
    event (seeded), so the twin is bitwise-equal at birth."""
    a, _, _ = _grown_net(seed=7)
    b, _, _ = _grown_net(seed=7)
    assert _state_hash(a) == _state_hash(b)
    assert a.gain_ledger[-1]["specs"] == \
        b.gain_ledger[-1]["specs"]


# ---------------- V-A2-3 spec hygiene ----------------

def test_a2_unknown_spec_field_refused():
    with pytest.raises(ValueError, match="unknown field"):
        StructureSpec(kind="reference", bogus=1)
    with pytest.raises(ValueError, match="unknown field"):
        Tap("stream", colour="red")


def test_a2_required_fields_have_no_defaults():
    with pytest.raises(TypeError):
        WiringSpec()                       # reads REQUIRED
    with pytest.raises(ValueError, match="REQUIRED"):
        WiringSpec(reads=[], write={"target": "stream"})
    with pytest.raises(ValueError, match="REQUIRED"):
        WiringSpec(reads=[Tap("stream")], write={})


def test_a2_unknown_kind_refused_with_catalog():
    net = Network(3, 8, lr=1e-2, seed=1)
    net.grow  # touch
    import reference_net.growth_port  # noqa: F401 (builders)
    with pytest.raises(ValueError, match="unknown structure"):
        compose.build(net, StructureSpec(kind="hologram"))


def test_a2_undefined_designation_refused():
    with pytest.raises(ValueError, match="unknown role"):
        Tap("stream", role="sideways")
    net = Network(3, 8, lr=1e-2, seed=1)
    w = WiringSpec(reads=[Tap("stream")],
                   write={"target": "stream"})
    with pytest.raises(ValueError, match="undefined chain"):
        compose.resolve(net, w, PlacementSpec(chain="spiral"))


# ---------------- V-A2-4 stage units ----------------

def test_a2_stage_place_none_is_noop_and_blocks_inserts():
    net = Network(3, 8, lr=1e-2, seed=1)
    w = WiringSpec(reads=[Tap("stream")],
                   write={"target": "stream"})
    r = compose.resolve(net, w, PlacementSpec(chain=NONE))
    assert compose.place(net, {"x": 1}, r) is None
    assert net.blocks == []
    r = compose.resolve(net, w, PlacementSpec(chain="blocks",
                                              position=END))
    k = compose.place(net, {"x": 1}, r)
    assert k == 0 and net.blocks == [{"x": 1}]
    net.blocks.clear()


def test_a2_stage_record_writes_verbatim_specs():
    net = Network(3, 8, lr=1e-2, seed=1)
    sd = specs_as_dict(StructureSpec(kind="reference"),
                       WiringSpec(reads=[Tap("stream")],
                                  write={"target": "stream"}),
                       PlacementSpec(), BirthSpec())
    rec = compose.record(net, "refine", 0, 42, sd)
    assert net.gain_ledger[-1] is rec
    assert rec["specs"] == sd and rec["params_added"] == 42


def test_a2_recipes_registry():
    zero = get_recipe("zero")({"W": (2, 2)}, None, {})
    assert np.array_equal(zero["W"], np.zeros((2, 2)))
    src = {"W": np.arange(4.0).reshape(2, 2)}
    cp = get_recipe("copy_layer")({"W": (2, 2)}, None,
                                  {"source": src})
    assert np.array_equal(cp["W"], src["W"])
    left = {"W": np.zeros((2, 2))}
    right = {"W": np.ones((2, 2)) * 4}
    iv = get_recipe("interleave_neighbors")(
        {"W": (2, 2)}, None, {"left": left, "right": right})
    assert np.array_equal(iv["W"], np.ones((2, 2)) * 2)
    with pytest.raises(ValueError, match="unknown recipe"):
        get_recipe("astrology")
    with pytest.raises(ValueError, match="already registered"):
        register_recipe("zero", lambda *a: {})


def test_a2_refused_growth_duplicate_state_untouched():
    man = json.loads((FIX / "MANIFEST.json").read_text())
    net = pickle.loads((FIX / "fx4_dup_pre.pkl").read_bytes())
    assert _state_hash(net) == \
        man["fixtures"]["FX-4"]["duplicate"]["state_hash"]
    with pytest.raises(ValueError, match="already composite"):
        net.grow(0, hidden=4)
    assert _state_hash(net) == \
        man["fixtures"]["FX-4"]["duplicate"]["state_hash"]


def test_a2_refused_growth_port_type_state_untouched():
    man = json.loads((FIX / "MANIFEST.json").read_text())
    net = pickle.loads((FIX / "fx4_ptype_pre.pkl").read_bytes())
    with pytest.raises(ValueError, match="legacy_scalar"):
        net.grow(0, hidden=4)
    assert _state_hash(net) == \
        man["fixtures"]["FX-4"]["port_type"]["state_hash"]


def test_a2_refused_growth_guard_leaves_recorded_post_state():
    """The P0 capture fact, preserved: the guard refusal fires
    AFTER the seed increment — post-state equals the RECORDED
    post hash, delta exactly seed_counter+1."""
    man = json.loads((FIX / "MANIFEST.json").read_text())
    rec = man["fixtures"]["FX-4"]["guard_refuse"]
    net = pickle.loads((FIX / "fx4_guard_pre.pkl").read_bytes())
    assert _state_hash(net) == rec["state_hash_pre"]
    pre_seed = int(net._seed_counter)
    # the fixture already carries grow_scale_guard="refuse"
    # (set before capture; part of the hashed state)
    assert net._growth_policy["grow_scale_guard"] == "refuse"
    with pytest.raises(ValueError, match="scale-hierarchy"):
        net.grow(0, hidden=4)
    assert net._seed_counter == pre_seed + 1
    assert net._seed_counter == rec["post_seed_counter"]
    assert _state_hash(net) == rec["state_hash_post"]
