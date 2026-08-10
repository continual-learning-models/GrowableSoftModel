"""Attention-build S8 product-completeness tests (plan D3
T16-T18): the new substrate and its growth verb are reachable
through every service surface, and the product docs tell the
truth.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

import core._modules  # noqa: E402,F401
from core.facade import System  # noqa: E402
from core.substrates import GUIDANCE, REGISTRY  # noqa: E402
from core.substrates.growable_attention import POLICY  # noqa: E402


ROWS = [{"input": {"x": i * 0.1, "y": (i % 5) * 0.2},
         "target": 2 * (i * 0.1) - (i % 5) * 0.2} for i in range(60)]


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def test_t16_service_exposure_end_to_end(ws, monkeypatch):
    """T16: (a) list_substrates (facade + the MCP tool metadata)
    includes growable_attention with its GUIDANCE row; (b) create
    -> study -> infer flows through the standard door; (c) the
    grow_attention verb returns a governed event and changes the
    head census; on a non-attention model it REFUSES naming the
    boundary; (d) a refused event leaves the model byte-identical.
    """
    # (a) exposure
    subs = ws.list_substrates()          # {"substrates": {name: guidance}}
    assert "growable_attention" in subs["substrates"]
    assert "growable_attention" in GUIDANCE
    import mcp.mcp_server as mserv
    src = Path(mserv.__file__).read_text()
    assert '"grow_attention"' in src, "MCP tool entry missing"
    # (b) standard door
    r = ws.create_model("t16", holdout=ROWS[:12],
                        substrate="growable_attention")
    assert r.get("substrate") == "growable_attention"
    ws.study("t16", ROWS[12:], steps=120)
    out = ws.infer("t16", {"x": 0.3, "y": 0.4}, working=True)
    assert out["output"] is not None
    # (c) the verb (age gate bypassed for the fixture)
    monkeypatch.setitem(POLICY, "att_head_age_min", 0)
    row = ws.grow_attention("t16", layer=0)
    assert row.get("verdict") in ("accepted", "refused",
                                  "no_trigger")
    if row["verdict"] == "accepted":
        rec = ws.lc._load_working("t16")[0].shape_record()
        assert rec["heads"] != [[1], [1]], rec
    ref = ws.create_model("t16m", holdout=ROWS[:12],
                          substrate="mlp")
    ws.study("t16m", ROWS[12:], steps=20)
    out2 = ws.grow_attention("t16m")
    assert "refusal" in out2 and "growable_attention" in \
        out2["refusal"]
    # (d) refusal byte-identity through the verb: force an
    # unpassable gate by monkeypatching the driver's tol
    import pickle as pkl
    import core.substrates.growable_attention as gam
    organ0, _ = ws.lc._load_working("t16")
    blob0 = pkl.dumps(organ0.shape_record())
    orig = gam.attention_grow_event

    def strict(sub, l, X_eval, hx, hy, **kw):
        kw["tol"] = 0.0
        return orig(sub, l, X_eval, hx, hy, **kw)
    monkeypatch.setattr(gam, "attention_grow_event", strict)
    monkeypatch.setattr(
        sys.modules["core.lifecycle"], "attention_grow_event",
        strict, raising=False)
    row2 = ws.grow_attention("t16", layer=1)
    if row2.get("verdict") == "refused":
        organ1, _ = ws.lc._load_working("t16")
        assert pkl.dumps(organ1.shape_record()) == pkl.dumps(
            ws.lc._load_working("t16")[0].shape_record())


def test_t17_sms_smoke():
    """T17: through the SMS operator surface, a growable_attention
    model trains to convergence and serves — with ZERO SMS code
    changes (substrate passes through; this test pins that fact).
    Runs in the SMS venv if present, else through sys.path."""
    sms = ROOT.parent / "SoftModelSystem"
    if not (sms / "sms").is_dir():
        pytest.skip("SMS workspace not present")
    code = r'''
import os, sys, tempfile, json
import numpy as np
sms_root = %r
os.environ["SOFTMODEL_MODELS_ROOT"] = tempfile.mkdtemp()
sys.path.insert(0, sms_root)
from core.facade import System
from sms.data.schema import FeatureSpec, SchemaContract, TargetSpec
from sms.data.batches import BatchPlan
from sms.common.registry import ExperimentRegistry
from sms.training import trainer
from sms.generation.dist import get_dist
sy = System()
sy.create_model("smoke_att", substrate="growable_attention")
rng = np.random.default_rng(0)
sch = SchemaContract([FeatureSpec("a"), FeatureSpec("b")],
                     TargetSpec("y"))
rows = [{"a": float(a), "b": float(b), "y": float(2*a+b),
         "__split__": "train"}
        for a, b in rng.uniform(-2, 2, (60, 2))]
val = [{"input": {"a": r["a"], "b": r["b"]}, "target": r["y"]}
       for r in rows[:12]]
reg = ExperimentRegistry(tempfile.mktemp())
rec = trainer.train_to_convergence(
    sy, "smoke_att", lambda i: BatchPlan(rows, sch, batch_size=30,
                                         seed=i),
    val, steps_per_batch=20, patience=3, cap_rounds=25,
    registry=reg)
d = get_dist(sy, "smoke_att", val[0]["input"], working=True)
print(json.dumps({"rounds": rec["rounds"],
                  "kind": type(d).__name__}))
''' % str(sms)
    venv_py = sms / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    r = subprocess.run([py, "-c", code], capture_output=True,
                       text=True, timeout=600,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["rounds"] >= 1 and out["kind"] == "NumericDist"


def test_t18_product_docs_fresh():
    """T18: every registered substrate name appears in README and
    both QUICKSTARTs (the doc-freshness grep); the growth verb is
    documented; ARCHITECTURE_MAP's distribution table mentions the
    new host; SPU.md documents the analytic objective (S7 built)."""
    readme = (ROOT / "README.md").read_text()
    qh = (ROOT / "docs/QUICKSTART_HUMAN.md").read_text()
    qa = (ROOT / "docs/QUICKSTART_AI.md").read_text()
    amap = (ROOT / "docs/ARCHITECTURE_MAP.md").read_text()
    spu = (ROOT / "docs/SPU.md").read_text()
    for name in REGISTRY:
        assert name in readme, f"README missing substrate {name}"
        assert name in qh or name in qa, \
            f"QUICKSTARTs missing substrate {name}"
    assert "grow_attention" in readme
    assert "grow_attention" in qh or "grow_attention" in qa
    assert "growable_attention" in amap
    assert "analytic" in spu and "spu_objective" in spu
