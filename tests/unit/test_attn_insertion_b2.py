"""B2 boxes (plan 56 v1.3 stage B2; 54 v1.10 section 7):
whole-layer insertion on the attention hosts.

 B-2-1 insertion preservation — BOTH hosts, p in {0, mid, top}:
       outputs BITWISE unchanged at birth (zero attention
       out-projections + zero FFN second matrix + LN identity
       under the residual stream — survey 49 s2.2, the
       formally proven form); PLUS the full-copy case
       (recipe="copy_layer", zero_side="none"): inserted
       layer's tensors equal the designated source BITWISE
       (out-projections and, on GA, every head tensor
       included) and the function CHANGES — by declared
       choice.
 B-2-2 renumbering audit — the live-walk census gains exactly
       one layer's keys; grown FFN port sites BELOW the seam
       keep their keys, sites AT/ABOVE the seam shift by one;
       the host trains after insertion.
 B-2-3 artifact round-trip after insertion — save/load/serve
       bitwise.
 Provenance — deepen_layer[attn] events carry verbatim specs
       (JSON-safe); these hosts have no gain ledger (recorded
       fact), events live in host.growth_events.
 Verdict (PR-2, math-research/02 T8): whole-layer insertion =
 delta form at host scale (+1 degree exponent, rank cap
 unchanged); seam width = full model width (junction law
 satisfied by construction).
"""
import json
import pickle   # safe: locally produced objects only
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.substrates.growable_attention import \
    GrowableAttentionSubstrate                              # noqa: E402
from core.substrates.transformer import TransformerSubstrate  # noqa: E402
from reference_net.growth_port import layer_census          # noqa: E402


def _data(seed=0, n=6, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return X, y


def _ga():
    X, y = _data()
    ga = GrowableAttentionSubstrate(
        3, 8, d_model=8, n_layers=2, seed=99,
        heads_spec=[[4, 4], [4, 4]])
    for _ in range(5):
        ga.train_step(X, y)
    return ga, X, y


def _tr():
    X, y = _data()
    tr = TransformerSubstrate(3, 8, d_model=8, n_layers=2,
                              seed=11)
    for _ in range(5):
        tr.train_step(X, y)
    return tr, X, y


# ---------------- B-2-1 preservation ----------------

@pytest.mark.parametrize("mk", [_ga, _tr], ids=["GA", "TR"])
@pytest.mark.parametrize("pos", [0, 1, 2])
def test_b2_insertion_preserves_bitwise(mk, pos):
    host, X, y = mk()
    base = np.asarray(host.predict(X))
    p = host.insert_layer(pos)
    assert p == pos and host.L == 3
    out = np.asarray(host.predict(X))
    assert np.array_equal(out, base)
    host.train_step(X, y)                 # trains afterwards
    ev = host.growth_events[-1]
    assert ev["event"] == "deepen_layer[attn]"
    assert ev["specs"]["placement"]["position"] == pos
    json.dumps(ev["specs"])               # JSON-safe


def test_b2_full_copy_non_preserving_tr():
    tr, X, y = _tr()
    base = np.asarray(tr.predict(X))
    src = {r: np.asarray(tr._bk.to_numpy(tr.P[f"{r}_0"])).copy()
           for r in tr._LAYER_KEY_ROOTS}
    tr.insert_layer(1, recipe="copy_layer", source_index=0,
                    zero_side="none")
    for r in tr._LAYER_KEY_ROOTS:
        assert np.array_equal(
            np.asarray(tr._bk.to_numpy(tr.P[f"{r}_1"])),
            src[r]), r
    assert not np.array_equal(np.asarray(tr.predict(X)), base)


def test_b2_full_copy_non_preserving_ga_heads_included():
    ga, X, y = _ga()
    base = np.asarray(ga.predict(X))
    src_heads = [{nm: getattr(h, nm).copy()
                  for nm in ("Wq", "Wk", "Wv", "Wo")}
                 for h in ga.heads[0]]
    ga.insert_layer(1, recipe="copy_layer", source_index=0,
                    zero_side="none")
    for h, ref in zip(ga.heads[1], src_heads):
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            assert np.array_equal(getattr(h, nm), ref[nm]), nm
    assert not np.array_equal(np.asarray(ga.predict(X)), base)


# ---------------- B-2-2 renumbering audit ----------------

def test_b2_census_and_port_site_shift():
    ga, X, y = _ga()
    ga.grow_site("layer0/ffn[1]", hidden=4)
    ga.grow_site("layer1/ffn[2]", hidden=4)
    for _ in range(3):
        ga.train_step(X, y)
    pre = layer_census(ga, ga._LAYER_KEY_ROOTS)
    ga.insert_layer(1)
    post = layer_census(ga, ga._LAYER_KEY_ROOTS)
    assert len(post["P"]) == len(pre["P"]) + 8
    # below the seam: unchanged; at/above: shifted by one
    assert (0, 1) in ga._port_js and (2, 2) in ga._port_js
    assert 0 in ga._port_sites and 2 in ga._port_sites
    assert 1 not in ga._port_sites        # the newborn layer
    ga.train_step(X, y)                   # everything trains


# ---------------- B-2-3 artifact round-trip ----------------

@pytest.mark.parametrize("mk", [_ga, _tr], ids=["GA", "TR"])
def test_b2_artifact_roundtrip_after_insertion(mk):
    host, X, y = mk()
    host.insert_layer(1)
    for _ in range(2):
        host.train_step(X, y)
    out = np.asarray(host.predict(X))
    twin = pickle.loads(pickle.dumps(host))
    assert np.array_equal(np.asarray(twin.predict(X)), out)
    twin.train_step(X, y)


def test_b2_position_out_of_range_refused():
    ga, X, y = _ga()
    with pytest.raises(ValueError, match="out of range"):
        ga.insert_layer(5)
