"""Stage-2 unit tests: quarantine (both directions, loud), reservoir
statistics, persistence round-trip, provenance, PlasticSystem wiring."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.plasticity.store import (   # noqa: E402
    ExperienceStore, QuarantineViolation, row_hash)


def _tmp():
    return Path(tempfile.mkdtemp()) / "store"


def _row(i):
    return {"input": [float(i), float(i + 1)], "label": float(2 * i)}


def test_quarantine_refuses_holdout_source_loudly():
    st = ExperienceStore(_tmp())
    with pytest.raises(QuarantineViolation):
        st.add([_row(1)], source="holdout")


def test_quarantine_refuses_registered_content_loudly():
    st = ExperienceStore(_tmp())
    st.register_holdout([_row(5)])
    st.add([_row(4)], source="study")           # fine
    with pytest.raises(QuarantineViolation):
        st.add([_row(5)], source="study")       # same CONTENT as holdout
    assert st.n_refused == 1


def test_quarantine_evicts_already_stored_rows():
    st = ExperienceStore(_tmp())
    st.add([_row(1), _row(2)], source="study")
    out = st.register_holdout([_row(2)])
    assert out["evicted"] == 1
    assert st.all_rows() == [_row(1)]
    assert _row(2) not in st.all_rows()


def test_reservoir_cap_and_spread():
    st = ExperienceStore(_tmp(), cap=100, seed=7)
    st.add([_row(i) for i in range(1000)], source="study")
    assert len(st) == 100 and st.n_seen == 1000
    kept_idx = [int(r["input"][0]) for r in st.all_rows()]
    # uniform reservoir: mean kept index near 500 (tolerance 15%)
    assert abs(np.mean(kept_idx) - 500) < 150
    # early items must not dominate (algorithm-R property)
    assert sum(1 for i in kept_idx if i < 100) < 30


def test_persistence_round_trip():
    d = _tmp()
    st = ExperienceStore(d, cap=10, seed=7)
    st.add([_row(i) for i in range(5)], source="study")
    st.register_holdout([_row(99)])
    st.save()
    st2 = ExperienceStore(d, cap=10, seed=7)
    assert len(st2) == 5 and st2.n_seen == 5
    with pytest.raises(QuarantineViolation):
        st2.add([_row(99)], source="study")     # quarantine survived


def test_provenance_and_hash_stability():
    st = ExperienceStore(_tmp())
    st.add([_row(1)], source="stream-A")
    assert st.rows[0]["source"] == "stream-A"
    assert row_hash(_row(1)) == row_hash({"label": 2.0,
                                          "input": [1.0, 2.0]})


def test_plastic_system_feeds_store_and_quarantines(tmp_path,
                                                    monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    from core.plasticity.session import PlasticSystem
    s = PlasticSystem()
    rows = [{"input": {"a": float(i), "b": float(i + 1)},
             "target": float(3 * i)} for i in range(30)]
    hold = [{"input": {"a": 100.0, "b": 101.0}, "target": 300.0}]
    s.create_model("st2", holdout=hold)
    s.study("st2", rows, steps=5)
    st = s.store("st2")
    assert len(st) == 30                        # study rows retained
    with pytest.raises(QuarantineViolation):
        st.add(hold, source="study")            # holdout content refused
    # add_holdout registers BEFORE the stream: same content refused too
    hold2 = [{"input": {"a": 200.0, "b": 201.0}, "target": 600.0}]
    s.add_holdout("st2", hold2)
    with pytest.raises(QuarantineViolation):
        st.add(hold2, source="study")
