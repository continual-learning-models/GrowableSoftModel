"""Depth-cap semantics (owner-confirmed): the cap applies PER BRANCH —
a branch at max_depth stops; shallower branches continue; refusal only
when no eligible branch remains. Total-size cap unchanged (global)."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from core.wiring import SysFactory
from core.lifecycle import Lifecycle, _site_level

RNG = np.random.default_rng(0)


def rows(n):
    X = RNG.uniform(0, 2, (n, 3))
    return [{"input": {"a": float(x[0]), "b": float(x[1]),
                       "c": float(x[2])},
             "target": str(round(float(x[0] + 2 * x[1]), 6))} for x in X]


def test_site_level_both_hosts():
    assert _site_level("root[3]") == 1          # mlp outer
    assert _site_level("2[1]") == 2             # mlp inner
    assert _site_level("2/1[0]") == 3           # mlp deep
    assert _site_level("layer0/ffn[3]") == 1    # transformer outer
    assert _site_level("layer1/ffn[10]::root[2]") == 2
    assert _site_level("layer1/ffn[10]::2[1]") == 3


def test_branch_at_cap_stops_others_continue():
    tmp = tempfile.mkdtemp()
    try:
        lc = Lifecycle(SysFactory(Config.from_env(backend="mlp",
                                                  models_root=Path(tmp))))
        lc.create("m", holdout=rows(30))
        lc.study("m", rows(150), steps=100)
        lc.set_policy("m", max_depth=2, max_params_mult=1000)
        g1 = lc.grow("m", k_nodes=1)             # depth 1 -> 2 (at cap)
        assert g1["grown"] and g1["depth"] == 2
        # branch at cap: its inner sites (level 2 -> depth 3) must be
        # SKIPPED; other shallow root sites (level 1 -> depth 2) allowed
        g2 = lc.grow("m", k_nodes=1)
        assert g2.get("grown"), g2               # growth continued
        assert g2["depth"] == 2                  # cap never exceeded
        assert g2["grown"][0] != g1["grown"][0]  # a DIFFERENT branch grew
        # exhaust all shallow sites -> refusal only then
        organ, _ = lc._load_working("m")
        for _ in range(organ.H + 2):
            r = lc.grow("m", k_nodes=1)
            if r.get("refusal"):
                break
        assert r.get("refusal") == "depth budget", r
        organ, _ = lc._load_working("m")
        assert organ.depth() == 2                # cap held throughout
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_site_level_both_hosts()
    test_branch_at_cap_stops_others_continue()
    print("depth-policy tests passed")
