"""S9-1 unit tests: self_review readouts — hand-computed retention
verdicts, saturation presence, uncertainty/question-list behavior,
and the consent guarantee (report changes nothing)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.facade import System                 # noqa: E402
from core.plasticity.self_review import (              # noqa: E402
    self_review, RETENTION_SAG)


def _rows(n):
    return [{"input": {"a": float(i % 7 - 3), "b": float(i % 5 - 2)},
             "target": float((i % 7 - 3) * (i % 5 - 2))}
            for i in range(n)]


@pytest.fixture()
def s(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def _fake_evals(s, mid, acc_seq):
    for accs in acc_seq:
        s.lc._log(mid, "evaluate", stage_accs=accs)


def test_sagging_detection_hand_computed(s):
    s.create_model("sr", holdout=_rows(20)[:6])
    s.study("sr", _rows(40), steps=30)
    # suite 0 healthy drift; suite 1 sagging (-0.05/-0.06/-0.05 tail)
    _fake_evals(s, "sr", [[0.9, 0.99], [0.9, 0.95], [0.91, 0.90],
                          [0.90, 0.84]])
    r = self_review(s, "sr")
    assert r["suites"][0]["sagging"] is False
    assert r["suites"][1]["sagging"] is True
    assert r["sagging_suites"] == [1]
    assert abs(r["suites"][1]["retention_trend"]
               - ((0.95 - 0.99) + (0.90 - 0.95) + (0.84 - 0.90)) / 3) < 1e-9
    assert RETENTION_SAG == -0.03              # frozen constant


def test_review_reports_structure_saturation_and_store(s):
    s.create_model("sr2", holdout=_rows(20)[:6])
    s.study("sr2", _rows(40), steps=30)
    s.widen("sr2", k=1)
    r = self_review(s, "sr2")
    assert r["structure_events"] == 1
    assert "root" in r["saturation"]
    assert r["store"]["rows"] > 0
    assert "granted" in r["consent_note"]


def test_uncertainty_question_list_never_answers_itself(s):
    s.create_model("sr3", holdout=_rows(20)[:6])
    s.study("sr3", _rows(60), steps=100)
    s.commit("sr3")
    probes = [{"a": float(x), "b": float(y)}
              for x in (-3, 0, 3) for y in (-2, 0, 2)]
    r = self_review(s, "sr3", probe_inputs=probes)
    qs = r["uncertainty"]["questions"]
    assert 0 < len(qs) <= 10
    assert all(set(q) == {"input", "reason", "score"} for q in qs)
    assert all("answer" not in q for q in qs)  # asks; never answers
    scores = [q["score"] for q in qs]
    assert scores == sorted(scores, reverse=True)


def test_report_only_consent_guarantee(s):
    """self_review must change NOTHING: organ, events, store."""
    s.create_model("sr4", holdout=_rows(20)[:6])
    s.study("sr4", _rows(40), steps=30)
    organ_before, _ = s.lc._load_working("sr4")
    w1 = copy.deepcopy(organ_before.W1)
    n_events = len(s.lc.events("sr4"))
    n_store = len(s.store("sr4"))
    self_review(s, "sr4", probe_inputs=[{"a": 1.0, "b": 1.0}])
    organ_after, _ = s.lc._load_working("sr4")
    assert np.array_equal(organ_after.W1, w1)
    assert len(s.lc.events("sr4")) == n_events
    assert len(s.store("sr4")) == n_store
