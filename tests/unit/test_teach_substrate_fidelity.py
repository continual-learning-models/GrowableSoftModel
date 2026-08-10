"""60AA P2: teach-lane substrate FIDELITY. The Phase-1 defect:
the lane built an MLP regardless of the model's chosen
substrate. Boxes written FIRST; RED today IS the defect
(taught organ comes back MLPSubstrate)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.facade import System                       # noqa: E402
from core.wiring import Config                       # noqa: E402

TR_SP = {"d_model": 8, "n_layers": 1, "n_heads": 1, "seed": 3}
ATT_SP = {"d_model": 8, "n_layers": 1, "heads_spec": [[1]],
          "seed": 3}

RNG = np.random.default_rng(4)
ROWS = [{"input": {"a": float(x[0]), "b": float(x[1]),
                   "c": float(x[2])},
         "target": float(2 * x[0] - x[1])}
        for x in RNG.normal(size=(32, 3))]
SEQ_ROWS = [{"input": [[float(i + j)] for j in range(4)],
             "target": float(i + 4)} for i in range(24)]


def _sys(tmp_path):
    return System(Config.from_env(backend="mlp",
                                  models_root=tmp_path / "ws"))


def _taught_organ(s, mid):
    """The organ the teach lane actually built (candidate v1
    when the gate did not promote)."""
    ver = s.lc.reg.active(mid)
    loaded = s.f.model_manager._load(
        mid, ver if ver != "v0" else "v1")
    assert loaded is not None, (mid, ver)
    return loaded[0]


def test_t2a_transformer_model_teaches_transformer(tmp_path):
    s = _sys(tmp_path)
    out = s.create_model("t", substrate="transformer",
                         policy={"substrate_params": TR_SP})
    assert "refusal" not in out, out
    r = s.teach("t", ROWS)
    assert isinstance(r, dict), r
    organ = _taught_organ(s, "t")
    assert type(organ).__name__ == "TransformerSubstrate", \
        type(organ).__name__                      # RED: MLP
    assert organ.d == 8                           # params honored
    X = np.array([[r_["input"]["a"], r_["input"]["b"],
                   r_["input"]["c"]] for r_ in ROWS[:4]])
    p = np.asarray(organ._bk.to_numpy(organ.predict(X)))
    assert np.isfinite(p).all()


def test_t2b_ga_model_teaches_ga_with_spec(tmp_path):
    s = _sys(tmp_path)
    out = s.create_model("g", substrate="growable_attention",
                         policy={"substrate_params": ATT_SP})
    assert "refusal" not in out, out
    s.teach("g", ROWS)
    organ = _taught_organ(s, "g")
    assert type(organ).__name__ == \
        "GrowableAttentionSubstrate", type(organ).__name__
    assert len(organ.heads[0]) == 1               # spec honored


def test_t2c_dist_composes_with_fidelity(tmp_path):
    s = _sys(tmp_path)
    out = s.create_model("td", substrate="transformer",
                         policy={"numeric_head": "dist",
                                 "substrate_params": TR_SP})
    assert "refusal" not in out, out
    s.teach("td", ROWS)
    organ = _taught_organ(s, "td")
    assert type(organ).__name__ == "TransformerSubstrate"
    assert organ.mode == "numeric_dist"
    X = np.array([[r_["input"]["a"], r_["input"]["b"],
                   r_["input"]["c"]] for r_ in ROWS[:4]])
    v, sd = organ.predict_dist(X)
    assert np.isfinite(np.asarray(v)).all()
    assert (np.asarray(sd) > 0).all()


def test_t2d_sequence_model_refuses_on_teach_lane(tmp_path):
    s = _sys(tmp_path)
    out = s.create_model("q", holdout=SEQ_ROWS[:4])
    assert out.get("substrate") == "sequence", out
    with pytest.raises(ValueError,
                       match="teach lane serves vector"):
        s.teach("q", SEQ_ROWS)


def test_t2e_override_knob_precedence(tmp_path):
    s = _sys(tmp_path)
    s.create_model("t", substrate="transformer",
                   policy={"substrate_params": TR_SP})
    s.f.trainer.substrate_name = "mlp"    # legacy knob wins
    s.teach("t", ROWS)
    organ = _taught_organ(s, "t")
    assert type(organ).__name__ == "MLPSubstrate"
