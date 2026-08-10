"""M2 drift-awareness tests: reality changes -> drift detected -> re-teach
wins on the recent slice (which M1 gating would have blocked)."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.spec import ModelSpec

FEATURES = ["amount", "night", "foreign"]

# OLD reality (rule A): HIGH iff amount >= 500 or (night and foreign)
TRAIN_OLD = [
    {"input": {"amount": 100, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 50, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 200, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 300, "night": 1, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 150, "night": 0, "foreign": 1}, "target": "LOW"},
    {"input": {"amount": 250, "night": 0, "foreign": 1}, "target": "LOW"},
    {"input": {"amount": 900, "night": 0, "foreign": 0}, "target": "HIGH"},
    {"input": {"amount": 700, "night": 1, "foreign": 0}, "target": "HIGH"},
    {"input": {"amount": 1200, "night": 0, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 600, "night": 1, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 100, "night": 1, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 50, "night": 1, "foreign": 1}, "target": "HIGH"},
]
HOLDOUT_OLD = [
    {"input": {"amount": 120, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 80, "night": 1, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 220, "night": 0, "foreign": 1}, "target": "LOW"},
    {"input": {"amount": 60, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 800, "night": 0, "foreign": 0}, "target": "HIGH"},
    {"input": {"amount": 650, "night": 1, "foreign": 0}, "target": "HIGH"},
    {"input": {"amount": 70, "night": 1, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 1500, "night": 0, "foreign": 0}, "target": "HIGH"},
]

# NEW reality (rule B): fraud moved to daytime-foreign —
# HIGH iff amount >= 500 or (foreign and NOT night)
TRAIN_NEW = [
    {"input": {"amount": 100, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 200, "night": 1, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 300, "night": 1, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 150, "night": 1, "foreign": 1}, "target": "LOW"},
    {"input": {"amount": 250, "night": 1, "foreign": 1}, "target": "LOW"},
    {"input": {"amount": 60, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 900, "night": 1, "foreign": 0}, "target": "HIGH"},
    {"input": {"amount": 700, "night": 0, "foreign": 0}, "target": "HIGH"},
    {"input": {"amount": 120, "night": 0, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 80, "night": 0, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 600, "night": 0, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 50, "night": 0, "foreign": 1}, "target": "HIGH"},
]
HOLDOUT_NEW = [
    {"input": {"amount": 110, "night": 0, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 90, "night": 1, "foreign": 1}, "target": "LOW"},
    {"input": {"amount": 240, "night": 1, "foreign": 0}, "target": "LOW"},
    {"input": {"amount": 70, "night": 1, "foreign": 1}, "target": "LOW"},
    {"input": {"amount": 800, "night": 0, "foreign": 0}, "target": "HIGH"},
    {"input": {"amount": 75, "night": 0, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 130, "night": 0, "foreign": 1}, "target": "HIGH"},
    {"input": {"amount": 1500, "night": 1, "foreign": 0}, "target": "HIGH"},
]


def test_drift_detect_and_reteach():
    tmp = tempfile.mkdtemp()
    try:
        f = SoftModelFactory(Config.from_env(
            backend="mlp", models_root=Path(tmp),
            gate_recent_n=8, drift_tolerance=0.15))
        m = "risk"
        f.create(ModelSpec(m, holdout=HOLDOUT_OLD))

        # 1) learn OLD reality
        r1 = f.teach(m, TRAIN_OLD)
        assert r1["promoted"] and r1["candidate_metric"] >= 0.9, r1

        # no drift while reality is unchanged
        d0 = f.check_drift(m)
        assert not d0["drifted"], d0

        # 2) reality changes: fresh labeled data appended to held-out
        f.add_holdout(m, HOLDOUT_NEW)

        # 3) drift detected: recent slice (the 8 new examples) collapses
        d1 = f.check_drift(m)
        assert d1["drifted"] and d1["needs_reteach"], d1
        assert d1["recent_metric"] <= d1["baseline_metric"] - 0.15, d1

        # 4) re-teach with new-reality examples, windowed to shed contradicted
        #    old labels; gate re-evaluates the LIVE version fresh on the recent
        #    slice, so the adapted candidate can win (M1 gating would block it)
        r2 = f.teach(m, TRAIN_NEW, window=len(TRAIN_NEW))
        assert r2["promoted"], r2
        assert r2["candidate_metric"] >= 0.85, r2
        assert r2["live_metric_before"] <= 0.6, r2   # stale live re-scored low

        # 5) drift cleared
        d2 = f.check_drift(m)
        assert not d2["drifted"], d2
    finally:
        shutil.rmtree(tmp)


def test_window_only_affects_training_not_lineage():
    tmp = tempfile.mkdtemp()
    try:
        f = SoftModelFactory(Config.from_env(
            backend="mlp", models_root=Path(tmp), gate_recent_n=8))
        m = "risk"
        f.create(ModelSpec(m, holdout=HOLDOUT_OLD))
        f.teach(m, TRAIN_OLD)
        r = f.teach(m, TRAIN_NEW, window=len(TRAIN_NEW))
        # the version's saved store keeps FULL lineage (old + new examples)
        from generator.data import read_jsonl
        store = read_jsonl(
            f.registry.weights_dir(m, r["candidate_version"]) / "train_store.jsonl")
        assert len(store) == len(TRAIN_OLD) + len(TRAIN_NEW)
    finally:
        shutil.rmtree(tmp)


def test_teach_recent_n_unblocks_adaptation():
    """After drift, a mixed-era holdout can score the adapted candidate and
    the stale live version as a TIE (each aces its own era) — the strict gate
    then blocks adaptation. teach(recent_n=N) judges the gate on the recent
    slice and unblocks it. (Reproduces the acceptance-run scenario.)"""
    import random
    tmp = tempfile.mkdtemp()
    try:
        f = SoftModelFactory(Config.from_env(backend="mlp", models_root=Path(tmp)))
        rng = random.Random(20260702)
        space = [(a, b, c) for a in range(4) for b in range(4) for c in range(4)]
        rng.shuffle(space)
        fa = lambda a, b, c: a + 2 * b - c        # old reality
        fb = lambda a, b, c: a + b + c            # new reality
        rows = lambda tr, fn: [{"input": {"a": a, "b": b, "c": c},
                                "target": str(fn(a, b, c))} for a, b, c in tr]
        m = "calc"
        f.create(ModelSpec(m, holdout=rows(space[44:54], fa)))
        f.teach(m, rows(space[:34], fa))
        f.teach(m, rows(space[34:44], fa))
        f.add_holdout(m, rows(space[44:54], fb))   # reality changed
        assert f.check_drift(m, recent_n=10)["needs_reteach"]

        # without recent_n: mixed-era tie -> gate blocks the adapted candidate
        r_blocked = f.teach(m, rows(space[:34], fb), window=34)
        assert not r_blocked["promoted"], r_blocked

        # with recent_n: judged against reality as it is now -> promoted
        r = f.teach(m, rows(space[:34], fb), window=34, recent_n=10)
        assert r["promoted"], r
        assert not f.check_drift(m, recent_n=10)["drifted"]
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_drift_detect_and_reteach()
    test_window_only_affects_training_not_lineage()
    test_teach_recent_n_unblocks_adaptation()
    print("drift tests passed")
