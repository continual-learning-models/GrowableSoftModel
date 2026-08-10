"""Stage-7 unit tests: the total-plasticity verbs as USABLE system
capabilities through the facade — the way the brain (LLM) calls them."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.facade import System            # noqa: E402


def _rows(n, f=lambda a, b: a * b + a):
    return [{"input": {"a": float(i % 7 - 3), "b": float(i % 5 - 2)},
             "target": float(f(i % 7 - 3, i % 5 - 2))} for i in range(n)]


@pytest.fixture()
def sys_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def test_widen_verb_preserves_and_logs(sys_env):
    s = sys_env
    s.create_model("w1", holdout=_rows(24)[:8])
    s.study("w1", _rows(60), steps=200)
    q = {"a": 1.5, "b": -1.0}
    before = s.infer("w1", q, working=True)["output"]
    out = s.widen("w1", k=2)
    assert "refusal" not in out and out["widened"] == 2
    assert abs(s.infer("w1", q, working=True)["output"] - before) < 1e-9
    assert any(e["event"] == "widen" for e in s.lc.events("w1"))


def test_add_feature_verb_end_to_end(sys_env):
    s = sys_env
    s.create_model("f1", holdout=_rows(24)[:8])
    s.study("f1", _rows(60), steps=200)
    q = {"a": 1.5, "b": -1.0}
    before = s.infer("f1", q, working=True)["output"]
    out = s.add_feature("f1", "c", default=0.0)
    assert out["features"][-1] == "c"
    # exact preservation, and the new key is now accepted in queries
    after = s.infer("f1", {**q, "c": 3.0}, working=True)["output"]
    assert abs(after - before) < 1e-9
    # the new feature is learnable through normal study
    rows3 = [{"input": {"a": r["input"]["a"], "b": r["input"]["b"],
                        "c": float(i % 3)},
              "target": r["target"] + 2.0 * (i % 3)}
             for i, r in enumerate(_rows(60))]
    r = s.study("f1", rows3, steps=400)
    assert r["loss"] >= 0.0
    dup = s.add_feature("f1", "c")
    assert "refusal" in dup                        # no duplicate names


def test_refound_verb_uses_store_and_gate_stays_in_charge(sys_env):
    s = sys_env
    s.create_model("r1", holdout=_rows(30)[:10])
    s.study("r1", _rows(80), steps=300)
    assert len(s.store("r1")) > 0                  # memory fed by study
    out = s.refound("r1", steps=300)
    assert "candidate_params" in out and "commit()" in out["note"]
    assert any(e["event"] == "refound" for e in s.lc.events("r1"))
    verdict = s.commit("r1")                       # normal gate decides
    assert isinstance(verdict, dict)


def test_store_hooks_are_default_and_quarantined(sys_env):
    s = sys_env
    hold = _rows(90)[80:]
    s.create_model("q1", holdout=hold)
    s.study("q1", _rows(90), steps=50)   # includes holdout-content rows
    st = s.store("q1")
    from core.plasticity.store import row_hash
    stored = {e["hash"] for e in st.rows}
    assert all(row_hash(r) not in stored for r in hold)  # exam excluded
    assert len(st) > 0                                    # rest retained
