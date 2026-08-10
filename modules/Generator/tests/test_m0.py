"""Control-loop tests (mock backend, no deps beyond stdlib)."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.config import Config
from generator.registry import ModelRegistry
from generator.model_manager import ModelManager
from generator.trainer import Trainer
from generator.evaluator import Evaluator
from generator.evolve import Evolve
from generator.data import write_jsonl, normalize


def _stack(tmp):
    cfg = Config.from_env(backend="mock", models_root=Path(tmp))
    reg = ModelRegistry(cfg)
    mm = ModelManager(cfg, reg)
    tr = Trainer(cfg, reg, mm)
    ev = Evaluator(cfg, mm, reg)
    ec = Evolve(cfg, reg, mm, tr, ev)
    return reg, mm, tr, ev, ec


def test_registry_lifecycle():
    tmp = tempfile.mkdtemp()
    try:
        reg, *_ = _stack(tmp)
        reg.create_model("m1")
        assert reg.active("m1") == "v0"
        assert reg.next_version("m1") == "v1"
        reg.add_version("m1", "v1", parent="v0", score=0.5)
        reg.set_active("m1", "v1")
        assert reg.active("m1") == "v1"
        assert reg.get_score("m1", "v1") == 0.5
    finally:
        shutil.rmtree(tmp)


def test_normalize():
    assert normalize("Delivered!!!") == "delivered"
    assert normalize("  On   Its Way ") == "on its way"


def test_teach_improves_and_gate_rejects():
    tmp = tempfile.mkdtemp()
    try:
        reg, mm, tr, ev, ec = _stack(tmp)
        reg.create_model("m")
        holdout = [{"input": "a b", "target": "A"}, {"input": "c d", "target": "C"}]
        write_jsonl(reg.holdout_path("m"), holdout)

        assert ev.eval_version("m", "v0")["metric"] == 0.0

        r1 = ec.teach("m", [holdout[0]])
        assert r1["promoted"] and r1["candidate_metric"] == 0.5

        r2 = ec.teach("m", [holdout[1]])
        assert r2["promoted"] and r2["candidate_metric"] == 1.0

        # non-improving update must be rejected by the gate
        r3 = ec.teach("m", [{"input": "zz", "target": "Z"}])
        assert not r3["promoted"]
        assert reg.active("m") == r2["candidate_version"]
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_registry_lifecycle()
    test_normalize()
    test_teach_improves_and_gate_rejects()
    print("all tests passed ✅")
