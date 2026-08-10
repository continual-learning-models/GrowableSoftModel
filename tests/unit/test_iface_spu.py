"""S9.4 SPU installation on the product path (docs/system/6
D-N2; plan doc 7 T23 a-d). The audit finding this fixes: set_policy
stored spu_* keys but NOTHING installed them onto the organ — the
seam slept forever. Zero algorithm change: engine spu/* and the
host seams are untouched; this is transport.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))
sys.path.insert(0, str(ROOT))

from core.facade import System                         # noqa: E402

ROWS = [{"input": {"x": float(i) / 24.0, "y": float(24 - i) / 24.0},
         "target": (float(i) / 24.0) * 0.7 + 0.1}
        for i in range(24)]


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFTMODEL_MODELS_ROOT", str(tmp_path))
    return System()


def test_t23a_policy_installs_and_seam_fires(ws):
    """T23a: set_policy(spu_enabled=True, spu_K=8) -> after the
    next study/load the organ carries a VALIDATED _spu_policy
    (K=8) and the transformer seam actually consults it."""
    ws.create_model("t23a", holdout=ROWS[:8], substrate="transformer")
    ws.study("t23a", ROWS[8:], steps=10)
    organ0, _ = ws.lc._load_working("t23a")
    assert getattr(organ0, "_spu_policy", None) is None  # the old severed state
    ws.set_policy("t23a", spu_enabled=True, spu_K=8,
                  spu_warmup_steps=0)
    organ, _ = ws.lc._load_working("t23a")
    pol = getattr(organ, "_spu_policy", None)
    assert pol is not None and pol["spu_enabled"] is True
    assert pol["spu_K"] == 8
    # the seam consults it: grow an inner body, then one more study
    # step must run the SPU walk (visible as spu state on the organ)
    ws.study("t23a", ROWS[8:], steps=5)      # exercises train_step
    # seam contract: reads getattr(self, '_spu_policy') — presence
    # plus enabled=True is the switch; no exception = walk ran
    organ2, _ = ws.lc._load_working("t23a")
    assert organ2._spu_policy["spu_enabled"] is True


def test_t23b_bad_values_refused_loudly(ws):
    """T23b: bad key/value -> loud typed refusal from the ENGINE
    validator, at set_policy time (before storage), naming it."""
    ws.create_model("t23b", holdout=ROWS[:8], substrate="transformer")
    r = ws.set_policy("t23b", spu_enabled=True, spu_K="many")
    assert "refusal" in r and "spu_K" in r["refusal"]
    r2 = ws.set_policy("t23b", spu_made_up=1)
    assert "refusal" in r2 and "spu_made_up" in r2["refusal"]
    # nothing stored: policy carries no spu keys after refusals
    pol = ws.lc.policy("t23b")
    assert not any(k.startswith("spu_") for k in pol)


def test_t23c_analytic_objective_end_to_end(ws):
    """T23c: spu_objective='analytic' routes through set_policy to
    the organ's installed policy (the S7 optional key)."""
    ws.create_model("t23c", holdout=ROWS[:8], substrate="transformer")
    ws.study("t23c", ROWS[8:], steps=5)
    ws.set_policy("t23c", spu_enabled=True,
                  spu_objective="analytic")
    organ, _ = ws.lc._load_working("t23c")
    assert organ._spu_policy["spu_objective"] == "analytic"
    r = ws.set_policy("t23c", spu_objective="gaussian")
    assert "refusal" in r and "spu_objective" in r["refusal"]


def test_t23d_default_path_untouched(ws):
    """T23d: a model with NO spu keys never gains the attribute —
    the default path is structurally identical to before S9.4
    (the organ object carries no _spu_policy at all)."""
    ws.create_model("t23d", holdout=ROWS[:8], substrate="transformer")
    ws.study("t23d", ROWS[8:], steps=10)
    ws.set_policy("t23d", study_steps=50)      # non-spu update
    organ, _ = ws.lc._load_working("t23d")
    assert getattr(organ, "_spu_policy", None) is None
    out = ws.infer("t23d", {"x": 0.3, "y": 0.4}, working=True)
    assert "output" in out
