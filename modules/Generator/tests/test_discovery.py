"""M3 Type-D tests: genuine regularity discovery from raw observations,
gated against held-out reality, composing with drift awareness (M2)."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.spec import ModelSpec
from generator.rules import induce_rules

from test_drift import (FEATURES, TRAIN_OLD, HOLDOUT_OLD, TRAIN_NEW,
                        HOLDOUT_NEW)  # noqa: E402  (shared scenario data)

CLASSES = ["LOW", "HIGH"]


def test_induction_recovers_generating_regularities():
    """EmpiricalBench-style acceptance: recover the generating law from raw
    observations — and generalize to held-out rows it never saw."""
    rl = induce_rules(TRAIN_OLD, FEATURES, CLASSES)
    correct = sum(1 for r in HOLDOUT_OLD
                  if rl.predict(r["input"])["output"] == r["target"])
    assert correct >= 7, (correct, rl.describe())
    text = " | ".join(rl.describe())
    assert "amount >=" in text          # the threshold regularity was surfaced
    assert any(r.confidence >= 0.8 for r in rl.rules)


def test_induction_surfaces_interactions():
    """The new reality's generating law is an interaction (foreign AND day);
    a 2-condition rule must surface it."""
    rl = induce_rules(TRAIN_NEW, FEATURES, CLASSES)
    lines = rl.describe()
    assert any(("foreign == 1" in ln and "night == 0" in ln and "HIGH" in ln)
               for ln in lines), lines
    correct = sum(1 for r in HOLDOUT_NEW
                  if rl.predict(r["input"])["output"] == r["target"])
    assert correct >= 7, (correct, lines)


def test_discovery_model_end_to_end_with_drift():
    tmp = tempfile.mkdtemp()
    try:
        f = SoftModelFactory(Config.from_env(
            backend="mlp", models_root=Path(tmp),
            gate_recent_n=8, drift_tolerance=0.15))
        m = "risk_rules"
        f.create(ModelSpec(m, holdout=HOLDOUT_OLD))

        # untrained discoveries
        d = f.discoveries(m)
        assert d["n_rules"] == 0 and d["note"] == "untrained"

        # 1) teach raw observations -> regularities discovered and gated
        r1 = f.teach(m, TRAIN_OLD)
        assert r1["promoted"] and r1["candidate_metric"] >= 0.9, r1
        d1 = f.discoveries(m)
        assert d1["n_rules"] >= 1 and d1["metric"] >= 0.9
        assert "amount >=" in " | ".join(d1["regularities"])

        # explainable inference: the fired rule is reported
        out = f.infer(m, {"amount": 800, "night": 0, "foreign": 0})
        assert out["output"] == "HIGH" and out["rule"] and "amount" in out["rule"]

        # 2) contradictory batch -> gate rejects, live regularities unchanged
        garbage = [{"input": ex["input"],
                    "target": ("HIGH" if ex["target"] == "LOW" else "LOW")}
                   for ex in TRAIN_OLD[:6]]
        r2 = f.teach(m, garbage)
        assert not r2["promoted"], r2
        assert f.versions(m)["active"] == r1["candidate_version"]

        # 3) reality changes (M2 composes): drift -> windowed re-teach ->
        #    the DISCOVERED REGULARITIES THEMSELVES evolve
        probe = {"amount": 75, "night": 0, "foreign": 1}
        assert f.infer(m, probe)["output"] == "LOW"       # old regularity
        f.add_holdout(m, HOLDOUT_NEW)
        assert f.check_drift(m)["needs_reteach"]
        r3 = f.teach(m, TRAIN_NEW, window=len(TRAIN_NEW))
        assert r3["promoted"] and r3["candidate_metric"] >= 0.85, r3
        assert f.infer(m, probe)["output"] == "HIGH"      # new regularity
        d3 = f.discoveries(m)
        assert any(("foreign == 1" in ln and "night == 0" in ln)
                   for ln in d3["regularities"]), d3["regularities"]
        assert not f.check_drift(m)["drifted"]

        # discoveries is available for ANY model (no human-declared types)
        assert f.discoveries(m)["n_rules"] >= 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_induction_recovers_generating_regularities()
    test_induction_surfaces_interactions()
    test_discovery_model_end_to_end_with_drift()
    print("discovery tests passed")
