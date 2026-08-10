"""WP1 acceptance (A-WP1) + unit tests: curriculum containment and
disjointness, scaler-refresh function preservation, plateau detector,
rollback exactness, full scripted course (growth fires, pays, retains)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reference_net.curriculum import (containment_ok, make_splits, law,
                                   accuracy, BOUNDS)
from reference_net.net import Network
from reference_net.trainer import Course, collect_instability


def test_containment_and_disjointness():
    assert containment_ok()
    stages = make_splits()
    assert len(stages) == len(BOUNDS)
    for st in stages:
        rows = [set(map(tuple, st["X"][k])) for k in ("study", "practice", "eval")]
        assert not (rows[0] & rows[1]) and not (rows[0] & rows[2]) \
            and not (rows[1] & rows[2])
        # all points inside the stage bound; labels exactly the law
        assert np.all(st["X"]["study"] <= np.array(st["bound"]) + 1e-9)
        assert np.allclose(st["y"]["eval"][:, 0], law(st["X"]["eval"]))


def test_scaler_refresh_preserves_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2, (64, 3))
    y = law(X).reshape(-1, 1)
    net = Network(3, 8, seed=1)
    for _ in range(50):
        net.train_step(X, y)
    before = net.predict(X).copy()
    sd_old = net._y_sd
    # simulate the compensated refresh directly (same code path values)
    mu_n, sd_n = float(y.mean()) + 5.0, sd_old * 4.0
    net.W2 = net.W2 * (net._y_sd / sd_n)
    net.c = (net.c * net._y_sd + net._y_mu - mu_n) / sd_n
    net._y_mu, net._y_sd = mu_n, sd_n
    assert np.allclose(net.predict(X), before, atol=1e-8)


def test_plateau_detector_unit():
    net = Network(3, 4, seed=2)
    c = Course(net, make_splits()[:1], plateau_patience=3, min_gain=0.01)
    for acc in [0.50, 0.60, 0.70, 0.705, 0.703, 0.706]:   # stalled < target
        c.score_matrix.append({"t": 0, "stage_accs": [acc]})
    assert c.plateaued(0, target=0.9)
    c.score_matrix.append({"t": 0, "stage_accs": [0.75]})  # real gain
    assert not c.plateaued(0, target=0.9)


def test_rollback_restores_exactly():
    stages = make_splits()[:1]
    net = Network(3, 8, seed=3)
    c = Course(net, stages, eval_every=50, seed=3)
    c.study_block(0, 50)
    Xp = stages[0]["X"]["eval"]
    before = c.net.predict(Xp).copy()
    ckpt = c.grow_event(stage="s1")
    c.study_block(0, 50)                      # diverge after growth
    c.rollback(ckpt, stage="s1")
    assert np.allclose(c.net.predict(Xp), before)


def test_full_course_grows_pays_and_retains():
    """Deterministic end-to-end acceptance. Owner's principle: the student
    is FIXED (H=16); growth is forced by PROBLEM COMPLEXITY (stages 5-6).
    Expected: no growth on stages 1-4; growth (incl. RECURSIVE depth-3,
    zero code change) on 5-6; everything mastered; stage-1 retained."""
    stages = make_splits()
    net = Network(3, 16, seed=5)
    c = Course(net, stages, eval_every=200, plateau_patience=4, seed=5)
    m, e = c.run_scripted([0.9] * 6, max_blocks=40, grow_budget=10,
                          t_post_blocks=6)
    kinds = [x["event"] for x in e]
    final = m[-1]["stage_accs"]
    assert kinds.count("grow") - kinds.count("rollback") >= 1   # growth kept
    grow_stages = {x["stage"] for x in e if x["event"] == "grow"}
    assert grow_stages <= {"stage5_extended", "stage6_deep"}, grow_stages
    assert final[0] >= 0.95, final          # retention (stage 1 stays)
    assert min(final) >= 0.9, final         # every stage mastered
    assert c.net.depth() >= 2               # (observed: 3 — recursive)


def test_instability_collection_covers_depths():
    net = Network(3, 8, seed=6)
    net.grow(2)
    net.inner[2].grow(1)
    paths = {r[0] for r in collect_instability(net)}
    assert {"root", "2", "2/1"} <= paths


if __name__ == "__main__":
    test_containment_and_disjointness()
    test_scaler_refresh_preserves_function()
    test_plateau_detector_unit()
    test_rollback_restores_exactly()
    test_full_course_grows_pays_and_retains()
    test_instability_collection_covers_depths()
    print("wp1 tests passed")
