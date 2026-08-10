"""Capture the TI-02 pre-branch baseline (plan 84 D-1/D-2).

Runs the scripted life of tests/integration/
test_preference_integration.py::_life on the UNMODIFIED tree
(before any D-3 implementation edit) and stores the decision-trace
sha, served-function sha and final mse as the byte-identity
baseline. MUST be executed only on a tree with zero
implementation edits; the commit that adds the fixture is the
proof of when it ran.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))

import reference_net.growthpolicy as gp            # noqa: E402
from reference_net.net import Network              # noqa: E402

ADDI = lambda x: np.sin(2 * x) + 0.6 * np.cos(5 * x)  # noqa: E731
BASE_POL = {"min_energy_points": 10 ** 9, "min_window_rows": 64,
            "probe_steps": 80, "seed": 0}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    rng = np.random.default_rng(100)
    X = rng.uniform(-2, 2, size=(64, 1))
    y = ADDI(X[:, 0]).reshape(-1, 1)
    net = Network(d_in=1, hidden=3, lr=1e-2, seed=5)
    for _ in range(600):
        net.train_step(X, y)
    decisions = []
    for _k in range(2):
        d = gp.grow_with_policy(net, dict(BASE_POL))
        decisions.append(d)
        for _ in range(300):
            net.train_step(X, y)
    grid = np.linspace(-2, 2, 257).reshape(-1, 1)
    pred = net.predict(grid)
    fn_sha = hashlib.sha256(
        np.ascontiguousarray(pred).tobytes()).hexdigest()
    trace = [{k: v for k, v in d.items()
              if k != "policy_snapshot"} for d in decisions]
    dec_sha = hashlib.sha256(json.dumps(
        _jsonable(trace), sort_keys=True).encode()).hexdigest()
    mse = float(np.mean((net.predict(X) - y) ** 2))
    out = {"fn_sha": fn_sha, "dec_sha": dec_sha, "mse": mse,
           "captured_on": "pre-implementation tree (84 D-2)",
           "arms": [d["arm"] for d in decisions],
           "applied": [d.get("applied") for d in decisions]}
    path = ROOT / "tests" / "fixtures" / \
        "preference_ti02_baseline.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
