"""S9-2/5 unit tests: run_self — consent/budget/audit semantics,
review-before-failure action, store-only material, growth only behind
the explicit flag, gate untouched."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.facade import System                 # noqa: E402
from core.plasticity.run_self import run_self          # noqa: E402


def _rows(n, off=0):
    return [{"input": {"a": float((i + off) % 7 - 3),
                       "b": float((i + off) % 5 - 2)},
             "target": float(((i + off) % 7 - 3) * ((i + off) % 5 - 2))}
            for i in range(n)]


@pytest.fixture()
def s(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def _boot(s, mid="m"):
    s.create_model(mid, holdout=_rows(24, off=100)[:8])
    s.study(mid, _rows(60), steps=100)
    return mid


def test_refusals_budget_store_working(s):
    mid = _boot(s)
    assert "refusal" in run_self(s, mid, 0)                 # C3
    s.create_model("empty", holdout=_rows(8, off=50))
    assert "refusal" in run_self(s, "empty", 2)             # no organ/
    # (create-only model has neither store rows nor working organ)


def test_session_is_logged_and_budget_bounded(s):
    mid = _boot(s)
    out = run_self(s, mid, 3)
    assert out["blocks"] == 3                               # C3/C6
    ev = [e["event"] for e in s.lc.events(mid)]
    assert "self_study_start" in ev and "self_study_end" in ev  # C4
    start = next(e for e in s.lc.events(mid)
                 if e["event"] == "self_study_start")
    assert start["budget"] == 3 and start["allow_growth"] is False


def test_blocks_consolidate_and_report_quiz_diagnostics(s):
    """Post-SZ-1 semantics: self-quiz is DIAGNOSTIC ONLY — each block
    reports quiz_acc/weak_count while digestion stays uniform mixed
    replay; sagging stays a report (post-SR-1)."""
    mid = _boot(s)
    out = run_self(s, mid, 2)
    for a in out["actions"]:
        assert a["action"] == "consolidate"
        assert "quiz_acc" in a and "weak_count" in a


def test_self_quiz_grades_against_real_store_labels(s):
    from core.plasticity.run_self import self_quiz
    mid = _boot(s)
    q = self_quiz(s, mid)
    assert 0.0 <= q["quiz_acc"] <= 1.0
    st_rows = {str(sorted(r["input"].items()))
               for r in s.store(mid).all_rows()}
    for r in q["weak_rows"]:            # questions come from OWN store
        assert str(sorted(r["input"].items())) in st_rows


def test_variation_check_is_label_free_and_flags_ranked(s):
    from core.plasticity.run_self import variation_check
    mid = _boot(s)
    v = variation_check(s, mid)
    assert v["typical_response"] is not None
    for f in v["flags"]:
        assert f["reason"] == "variation_inconsistent"
        assert "target" not in f        # never fabricates answers
    ev_before = len(s.lc.events(mid))
    variation_check(s, mid)             # read-only: no events, no train
    assert len(s.lc.events(mid)) == ev_before


def test_no_growth_without_explicit_flag(s):
    mid = _boot(s)
    out = run_self(s, mid, 4)                               # default
    assert all(not a["action"].startswith("grow")
               for a in out["actions"])
    n_struct = sum(1 for e in s.lc.events(mid)
                   if e["event"] in ("grow", "widen"))
    assert n_struct == 0                                    # C1


def test_growth_possible_only_with_flag_and_gate_still_rules(s):
    mid = _boot(s)
    suites = [{"name": "t", "X": [r["input"] for r in _rows(20)],
               "y": [r["target"] for r in _rows(20)]}]
    s.evaluate(mid, suites)
    s.evaluate(mid, suites)                # flat LP on record
    out = run_self(s, mid, 4, suites=suites, allow_growth=True)
    assert out["blocks"] == 4
    # commit still the only promotion path: working != committed until
    verdict = s.commit(mid)
    assert isinstance(verdict, dict)


def test_material_comes_only_from_store(s):
    """C2: the session consumes exactly the store's rows (quarantined
    world); holdout content can never be replayed."""
    mid = _boot(s)
    st = s.store(mid)
    from core.plasticity.store import row_hash
    hold_hashes = st._quarantine
    assert all(e["hash"] not in hold_hashes for e in st.rows)
    run_self(s, mid, 2)
    assert all(e["hash"] not in hold_hashes for e in st.rows)
