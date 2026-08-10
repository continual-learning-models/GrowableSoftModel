"""Stage-5 unit tests: Phi triggers (fire on fixtures, silent on
healthy), sizing from accumulated store, gate-only takeover, incumbent
untouched (rollback trivially = keep serving the old object)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import reference_net                     # noqa: E402
from reference_net.net import Network                       # noqa: E402
from core.plasticity.refound import (                  # noqa: E402
    should_refound, size_from_store, refound, gated_takeover)


def test_trigger_fires_on_sustained_inversion_only():
    assert should_refound([0.6, 0.7, 0.8])["refound"]
    assert not should_refound([0.6, 0.7])["refound"]        # too short
    assert not should_refound([0.8, 0.4, 0.8])["refound"]   # not sustained
    assert not should_refound([0.1, 0.2, 0.1])["refound"]   # healthy


def test_trigger_disposed_and_lp_paths_need_high_saturation():
    assert should_refound([], disposed=2, max_saturation=0.6)["refound"]
    assert not should_refound([], disposed=2,
                              max_saturation=0.2)["refound"]
    assert should_refound([], lp_tail=[0.0, 0.0005, -0.0002],
                          max_saturation=0.6)["refound"]
    assert not should_refound([], lp_tail=[0.05, 0.02, 0.04],
                              max_saturation=0.6)["refound"]


def test_sizing_reads_accumulated_schema_not_batch_one():
    assert size_from_store(2, 5000) == max(16, 4 * 3)
    assert size_from_store(10, 5000) == 4 * 11   # schema grew -> bigger


def _world(seed, n, wrong=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = X[:, :1] * X[:, 1:2] + X[:, :1]
    return X, (-y if wrong else y)


def test_refound_fresh_beats_wrong_foundation_and_gate_promotes():
    Xw, yw = _world(7, 256, wrong=True)
    old = Network(d_in=2, hidden=8, seed=7)
    for _ in range(1500):
        old.train_step(Xw, yw)                 # confidently WRONG law
    Xs, ys = _world(8, 512)                    # store = true world
    Xh, yh = _world(9, 128)                    # untouched holdout
    old_pred_before = old.predict(Xh).copy()
    cand = refound(old, Xs, ys, steps=2500, seed=7)
    out = gated_takeover(old, cand, Xh, yh)
    assert out["promoted"] and out["candidate_mse"] < out["incumbent_mse"]
    assert out["event"]["event"] == "metamorphosis"
    # incumbent untouched throughout (kept serving; rollback = itself)
    assert np.array_equal(old.predict(Xh), old_pred_before)


def test_gate_refuses_undertrained_candidate():
    Xs, ys = _world(8, 512)
    Xh, yh = _world(9, 128)
    old = Network(d_in=2, hidden=8, seed=7)
    for _ in range(2000):
        old.train_step(Xs, ys)                 # healthy incumbent
    cand = refound(old, Xs, ys, steps=5, seed=7)   # nearly untrained
    out = gated_takeover(old, cand, Xh, yh)
    assert not out["promoted"]


def test_shrink_perturb_mode_runs_and_never_touches_old():
    Xs, ys = _world(8, 256)
    old = Network(d_in=2, hidden=8, seed=7)
    for _ in range(300):
        old.train_step(Xs, ys)
    W1_old = old.W1.copy()
    cand = refound(old, Xs, ys, steps=200, seed=7,
                   mode="shrink_perturb")
    assert cand.H == size_from_store(2, 256) and cand is not old
    assert np.array_equal(old.W1, W1_old)
