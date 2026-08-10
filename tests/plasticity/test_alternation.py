"""S9-3 unit tests: practice alternation is opt-in; attempt-0
protocol semantics (model answers first; only failures corrected)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.facade import System                 # noqa: E402


def _rows(n):
    return [{"input": {"a": float(i % 7 - 3), "b": float(i % 5 - 2)},
             "target": float((i % 7 - 3) * (i % 5 - 2))}
            for i in range(n)]


@pytest.fixture()
def s(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def _curriculum(rows):
    return [{"name": "L1", "examples": rows,
             "suite": {"X": [r["input"] for r in rows[:30]],
                       "y": [r["target"] for r in rows[:30]]},
             "target": 0.5}]


def test_default_loop_has_no_practice(s):
    s.create_model("c0", holdout=_rows(20)[:6])
    rep = s.run_course("c0", _curriculum(_rows(60)),
                       policy={"max_blocks_per_stage": 4,
                               "steps_per_block": 100})
    assert "practice_blocks" not in rep["stages"][0]
    ev = [e["event"] for e in s.lc.events("c0")]
    assert "practice" not in " ".join(ev)


def test_alternation_runs_and_counts_blocks(s):
    s.create_model("c1", holdout=_rows(20)[:6])
    rep = s.run_course("c1", _curriculum(_rows(60)),
                       policy={"max_blocks_per_stage": 6,
                               "steps_per_block": 100,
                               "practice_alternate": True})
    st = rep["stages"][0]
    assert st.get("practice_blocks", 0) >= 1
    assert st["blocks"] <= 6            # practice consumes budget too
