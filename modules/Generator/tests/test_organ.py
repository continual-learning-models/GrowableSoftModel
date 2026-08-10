"""Organ (mlp backend) tests: real learning, generalization, gate, lineage."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.factory import SoftModelFactory
from generator.spec import ModelSpec
from generator.data import featurize, read_jsonl

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "risk"


def _factory(tmp):
    return SoftModelFactory(Config.from_env(backend="mlp", models_root=Path(tmp)))


def test_featurize():
    assert featurize({"a": 1, "b": 2.5}, ["a", "b", "c"]) == [1.0, 2.5, 0.0]
    assert featurize([1, 2, 3], ["a", "b", "c"]) == [1.0, 2.0, 3.0]
    try:
        featurize("free text", ["a"])
        assert False, "should reject raw text"
    except ValueError:
        pass


def test_organ_learns_and_generalizes():
    tmp = tempfile.mkdtemp()
    try:
        f = _factory(tmp)
        train = read_jsonl(EXAMPLES / "train.jsonl")
        holdout = read_jsonl(EXAMPLES / "holdout.jsonl")
        m = "risk"
        f.create(ModelSpec(m, holdout=holdout))

        # v0: untrained default
        assert f.evaluate(m)["metric"] <= 0.5 + 1e-9

        # teach: trains from scratch on the versioned store; holdout is disjoint
        r1 = f.teach(m, train)
        assert r1["promoted"], r1
        assert r1["candidate_metric"] >= 0.9, r1   # generalization, not memorization

        # confidence is a probability
        out = f.infer(m, {"amount": 70, "night": 1, "foreign": 1})
        assert out["output"] == "HIGH"
        assert 0.0 <= out["confidence"] <= 1.0

        # contradictory batch -> candidate degrades -> gate rejects
        garbage = [{"input": ex["input"],
                    "target": ("HIGH" if ex["target"] == "LOW" else "LOW")}
                   for ex in train[:6]]
        r2 = f.teach(m, garbage)
        assert not r2["promoted"], r2
        # live version unchanged; rejected examples must not pollute the live store
        assert f.versions(m)["active"] == r1["candidate_version"]

        # teaching again from the clean live lineage still works
        r3 = f.teach(m, train[:2])   # no metric gain -> rejected, harmless
        assert f.versions(m)["active"] == r1["candidate_version"] or r3["promoted"]

        # rollback
        f.rollback(m, "v0")
        assert f.versions(m)["active"] == "v0"
    finally:
        shutil.rmtree(tmp)


def test_numeric_self_shaping():
    """No human-declared types: numeric targets make the model shape itself
    with a numeric head and GENERALIZE (the exact failure the old
    classification-only design had)."""
    import random
    tmp = tempfile.mkdtemp()
    try:
        f = _factory(tmp)
        rng = random.Random(7)
        space = [(a, b, c) for a in range(4) for b in range(4) for c in range(4)]
        rng.shuffle(space)
        formula = lambda a, b, c: a + 2 * b - c          # hidden numeric law
        rows = [{"input": {"a": a, "b": b, "c": c}, "target": str(formula(a, b, c))}
                for a, b, c in space]
        train, holdout, gen = rows[:44], rows[44:54], rows[54:64]
        f.create(ModelSpec("calc", holdout=holdout))
        r = f.teach("calc", train)
        assert r["promoted"] and r["candidate_metric"] >= 0.9, r
        # self-shaped numeric
        shape = f.card("calc")["learned_shape"]
        assert shape["mode"] == "numeric" and shape["integer"] is True
        # generalization on never-seen triples, exact integers (+-0.5 match)
        ok = sum(1 for g in gen
                 if f.infer("calc", g["input"])["output"] == int(g["target"]))
        assert ok >= 9, ok
        # discoveries honestly reports numeric shape
        assert "numeric" in f.discoveries("calc")["note"]
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_featurize()
    test_organ_learns_and_generalizes()
    test_numeric_self_shaping()
    print("organ tests passed")
