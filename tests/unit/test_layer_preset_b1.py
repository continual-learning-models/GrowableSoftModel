"""B1 boxes (plan 56 v1.3 stage B1; 54 v1.10 section 7):
PRESET_LAYER — whole-layer deepening at any designated scope.

 B-1-1 worked examples per recipe/position class; FULL-COPY
       (zero_side=NONE: every tensor equals the designated
       source bitwise, Bout included, function NOT preserved
       — by declared choice); CUSTOM-VALUES (a caller-supplied
       array lands verbatim: user-controlled initial values,
       FR-4/C-5).
 B-1-2 birth preservation at EVERY seam of a 3-block chain
       (zero-Bout insertion leaves predict() bitwise
       unchanged at any p).
 B-1-3 layer-event provenance (position/recipe/scope in the
       ledgered specs; JSON round-trip).
 B-1-4 arbitrary-scope cases (a half-width set AND a
       non-contiguous set — scope is a FREE parameter):
       out-of-scope read columns and write rows born EXACTLY
       ZERO; contribution confined to the scope at birth.
 deepen() default box — no new arguments => the historical
       delta path (adjudicated bitwise by the P0 recapture,
       reasserted here on the ledger event shape).
 Verdicts (PR-2, math-research/02 T8): layer insertion =
 delta form (+1 degree exponent, rank cap unchanged); scoped
 events are delta within the scope (scope-relative).
"""
import json
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

from reference_net.bodies import Block                      # noqa: E402
from reference_net.net import Network                       # noqa: E402
from reference_net.foundation.recipes import (              # noqa: E402
    RECIPES, register_recipe)


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def _net_with_blocks(nblocks=3, seed=7, steps=40):
    net = Network(3, 8, lr=1e-2, seed=seed)
    X, y = _data()
    for _ in range(steps):
        net.train_step(X, y)
    for _ in range(nblocks):
        net.deepen()
    for _ in range(10):
        net.train_step(X, y)     # blocks carry trained values
    return net, X, y


def _np_of(net, a):
    return np.asarray(net._bk.to_numpy(a))


# ---------------- deepen() default box ----------------

def test_b1_default_deepen_is_the_historical_path():
    net = Network(3, 8, lr=1e-2, seed=7)
    net.deepen()
    rec = net.gain_ledger[-1]
    assert rec["event"] == "deepen"
    assert rec["specs"]["placement"]["position"] == "end"
    assert rec["specs"]["birth"]["recipe"] == "random"


# ---------------- B-1-2 every-seam preservation ----------------

def test_b1_insertion_preserves_function_at_every_seam():
    for p in (0, 1, 2, 3):
        net, X, y = _net_with_blocks(3, seed=30 + p)
        base = _np_of(net, net.predict(X))
        k = net.deepen(position=p)
        assert k == p and len(net.blocks) == 4
        assert isinstance(net.blocks[p], Block)
        out = _np_of(net, net.predict(X))
        assert np.array_equal(out, base), f"seam {p}"
        net.train_step(X, y)          # trains after insertion


# ---------------- B-1-1 recipes ----------------

def test_b1_copy_layer_recipe_preserving():
    net, X, y = _net_with_blocks(2, seed=41)
    base = _np_of(net, net.predict(X))
    src = net.blocks[0]
    src_Bin = _np_of(net, src.Bin).copy()
    net.deepen(position=1, recipe="copy_layer")   # copies p-1=0
    blk = net.blocks[1]
    assert np.array_equal(_np_of(net, blk.Bin), src_Bin)
    assert np.array_equal(_np_of(net, blk.Bout),
                          np.zeros((8, 8)))       # zero side kept
    assert np.array_equal(_np_of(net, net.predict(X)), base)
    sp = net.gain_ledger[-1]["specs"]
    assert sp["birth"]["recipe"] == "copy_layer"
    assert sp["placement"]["position"] == 1


def test_b1_full_copy_zero_side_none_not_preserving():
    """Insert between blocks 2 and 3 copying block 2 IN FULL
    (the owner's worked case): every tensor equals the source
    bitwise, Bout included; the function CHANGES at birth."""
    net, X, y = _net_with_blocks(3, seed=42)
    # make the source block's Bout genuinely nonzero first
    assert float(np.abs(_np_of(net, net.blocks[1].Bout)).max()) > 0
    base = _np_of(net, net.predict(X))
    src = net.blocks[1]
    srcs = {k: _np_of(net, src[k]).copy() for k in src}
    net.deepen(position=2, recipe="copy_layer",
               recipe_params={"source_index": 1},
               zero_side="none")
    blk = net.blocks[2]
    for k in ("Bin", "bb", "Bout"):
        assert np.array_equal(_np_of(net, blk[k]), srcs[k]), k
    out = _np_of(net, net.predict(X))
    assert not np.array_equal(out, base)   # declared non-preserving


def test_b1_interleave_neighbors_recipe():
    net, X, y = _net_with_blocks(2, seed=43)
    left = {k: _np_of(net, net.blocks[0][k]).copy()
            for k in net.blocks[0]}
    right = {k: _np_of(net, net.blocks[1][k]).copy()
             for k in net.blocks[1]}
    base = _np_of(net, net.predict(X))
    net.deepen(position=1, recipe="interleave_neighbors")
    blk = net.blocks[1]
    assert np.array_equal(_np_of(net, blk.Bin),
                          0.5 * (left["Bin"] + right["Bin"]))
    assert np.array_equal(_np_of(net, blk.Bout),
                          np.zeros((8, 8)))       # zero side
    assert np.array_equal(_np_of(net, net.predict(X)), base)


def test_b1_custom_values_recipe_lands_verbatim():
    W = {"Bin": np.arange(64.0).reshape(8, 8),
         "bb": np.arange(8.0),
         "Bout": np.zeros((8, 8))}
    if "b1_custom" not in RECIPES:
        register_recipe("b1_custom",
                        lambda shapes, rng, ctx: {
                            k: v.copy() for k, v in W.items()})
    net, X, y = _net_with_blocks(1, seed=44)
    net.deepen(position=0, recipe="b1_custom")
    blk = net.blocks[0]
    assert np.array_equal(_np_of(net, blk.Bin), W["Bin"])
    assert np.array_equal(_np_of(net, blk.bb), W["bb"])


# ---------------- B-1-4 arbitrary scope ----------------

@pytest.mark.parametrize("scope", [
    [0, 1, 2, 3],                 # half width
    [0, 2, 5],                    # non-contiguous free set
])
def test_b1_scoped_deepening_birth_topology(scope):
    net, X, y = _net_with_blocks(1, seed=50 + len(scope))
    # NON-TRIVIAL confinement: a custom recipe supplies a
    # nonzero FULL Bout; the scope structuring must zero the
    # out-of-scope write rows, leaving a genuinely nonzero
    # in-scope contribution (zero Bout would make the
    # confinement assertion vacuous).
    name = f"b1_scope_probe_{len(scope)}"
    if name not in RECIPES:
        register_recipe(
            name, lambda shapes, rng, ctx: {
                "Bout": np.ones((8, 8)) * 0.1})
    net.deepen(scope=scope, zero_side="none", recipe=name)
    blk = net.blocks[-1]
    Bin = _np_of(net, blk.Bin); Bout = _np_of(net, blk.Bout)
    outside = [i for i in range(8) if i not in scope]
    # out-of-scope read columns and write rows born EXACTLY zero
    assert np.array_equal(Bin[:, outside],
                          np.zeros((8, len(outside))))
    assert np.array_equal(Bout[outside, :],
                          np.zeros((len(outside), 8)))
    # in-scope read columns carry the He draw (nonzero)
    assert float(np.abs(Bin[:, scope]).max()) > 0
    # contribution confined to the scope at birth
    H = np.asarray(np.random.default_rng(3).normal(size=(6, 8)))
    contrib = _np_of(net, blk.predict(net._bk.ingest(H)))
    assert np.array_equal(contrib[:, outside],
                          np.zeros((6, len(outside))))
    assert float(np.abs(contrib[:, scope]).max()) > 0
    sp = net.gain_ledger[-1]["specs"]
    assert sp["wiring"]["reads"][0]["span"] == sorted(scope)
    assert sp["wiring"]["write"]["span"] == sorted(scope)


def test_b1_scope_out_of_range_refused():
    net, X, y = _net_with_blocks(1, seed=60)
    with pytest.raises(ValueError, match="out of range"):
        net.deepen(scope=[0, 9])


# ---------------- B-1-3 provenance ----------------

def test_b1_specs_json_roundtrip():
    net, X, y = _net_with_blocks(2, seed=61)
    net.deepen(position=1, recipe="copy_layer", scope=[0, 1])
    sp = net.gain_ledger[-1]["specs"]
    assert json.loads(json.dumps(sp)) == sp
    assert sp["structure"]["kind"] == "block"
