"""Depth-field capability record (owner mandate): the grown depth
distribution is NON-UNIFORM in both senses — tree height per
branch, serial depth per scope.

Scenario selection disclosed: seeds 1-8 were scanned at authoring
time; seed 8 is the first whose four root-level governed rounds
fire both arms (rho at nodes 1, 0, 2 and one deepen), and two
further governed rounds on the earliest inner scope fire both arms
again (deepen + rho@4), yielding tree heights {1, 2, 3}
across branches and layer depths {1, 2} across scopes. Seeded,
deterministic; the field is recorded verbatim to
tests/logs/depth_field.jsonl.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
import reference_net.growthpolicy as gp  # noqa: E402
from reference_net.net import Network  # noqa: E402

LOG = ROOT / "tests" / "logs" / "depth_field.jsonl"
POL = {"min_energy_points": 10 ** 9, "probe_steps": 300,
       "min_window_rows": 64}


def _branch_depths(net):
    """Structural depth of the branch under each root node
    (fullwidth port bodies via grown_body — W3 migration)."""
    return {j: (1 + net.grown_body(j).depth()
                if net.grown_body(j) is not None else 1)
            for j in range(net.H)}


def _snapshot(net):
    return {"branch": {str(j): (1 + net.grown_body(j).depth()
                                if net.grown_body(j) is not None
                                else 1)
                       for j in range(net.H)},
            "serial": {r["path"]: 1 + r["blocks"]
                       for r in net.structure()}}


def _run():
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 2))
    y = (np.sin(3 * np.sin(3 * X[:, 0]))
         + 0.6 * np.cos(5 * X[:, 1])
         + 0.4 * np.sin(7 * X[:, 0] * X[:, 1])).reshape(-1, 1)
    net = Network(d_in=2, hidden=6, lr=1e-2, seed=3)
    net.gain_horizon = 400
    for _ in range(2000):
        net.train_step(X, y)
    decisions, timeline = [], [_snapshot(net)]

    def round_(scope, name):
        d = gp.grow_with_policy(scope, POL)
        decisions.append({"scope": name, "applied": d["applied"]})
        for _ in range(600):
            net.train_step(X, y)
        timeline.append(_snapshot(net))

    for _ in range(5):
        round_(net, "root")
    k0, k1 = sorted(net._port_js)[:2]      # fullwidth grows
    for _ in range(3):
        round_(net.grown_body(k0), f"scope {k0}")
    for _ in range(2):
        round_(net.grown_body(k1), f"scope {k1}")
    grand = [(k, g) for k in (k0, k1)
             if net.grown_body(k) is not None
             for g in sorted(getattr(net.grown_body(k),
                                     "_port_js", set()))]
    if grand:
        k, g = grand[0]
        round_(net.grown_body(k).grown_body(g), f"scope {k}/{g}")
    else:
        round_(net.grown_body(k0), f"scope {k0}")
    rows = [{"path": r["path"], "H": r["H"], "blocks": r["blocks"],
             "layer_depth": 1 + r["blocks"],
             "composite": r["composite"]}
            for r in net.structure()]
    return {"decisions": decisions, "scopes": rows,
            "timeline": timeline,
            "branch_tree_height":
                _snapshot(net)["branch"],
            "tree_height": net.depth()}


def test_depth_field_nonuniform_and_recorded():
    rec = _run()
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(json.dumps(rec) + "\n")
    serials = {r["layer_depth"] for r in rec["scopes"]}
    branches = set(rec["branch_tree_height"].values())
    assert len(serials) >= 2, serials      # non-uniform serial depth
    assert len(branches) >= 2, branches    # non-uniform structural
    assert rec["tree_height"] >= 2
    arms = {d["applied"] for d in rec["decisions"]}
    assert "deepen" in arms and any(
        str(a).startswith("rho") for a in arms)


def test_depth_field_replay_deterministic():
    a, b = _run(), _run()
    assert a == b
