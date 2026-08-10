"""SIM-E2E — real-user product simulation (plan 96 spec,
registered 2026-07-30 BEFORE this file): the FULL chain from
pretraining-under-growth to RL, on the documented user doors
only (System facade = product door; RL_TRAINER_RUNBOOK user
pattern for the P-loop). REAL components, no mocks; lines
L-E2E-1..10 asserted in place."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _m in ("Engine", "ReferenceNet", "RLTrainer"):
    p = str(ROOT / "modules" / _m)
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(ROOT))

from core.facade import System                    # noqa: E402
from core.wiring import Config                    # noqa: E402
from rl_trainer.eval_provider import (            # noqa: E402
    EvalEpisodeProvider, gate_adjudicate)
from rl_trainer.regime import RegimeDispatcher    # noqa: E402
from rl_trainer.runner import OrganPPORunner      # noqa: E402
from rl_trainer.worlds import StagedExpansionWorld  # noqa: E402

ROWS = [{"input": {"a": float(i) / 24.0,
                   "b": float(24 - i) / 24.0,
                   "c": float(i % 5) / 5.0},
         "target": (float(i) / 24.0) * 0.7
         + (float(i % 5) / 5.0) * 0.2 + 0.1}
        for i in range(24)]
VAL = [{"input": r["input"], "target": r["target"]}
       for r in ROWS[::3]]


def _mse(s, mid):
    errs = []
    for v in VAL:
        out = s.infer(mid, v["input"], working=True)
        assert "refusal" not in out, out       # L-E2E-5
        errs.append((float(out["output"]) - v["target"]) ** 2)
    return float(np.mean(errs))


def test_sim_e2e_pretrain_growth_to_rl(tmp_path):
    s = System(Config.from_env(backend="mlp",
                               models_root=tmp_path / "ws"))

    # ---------------- LEG 1: product pretraining under growth
    out = s.create_model("m", description="e2e",
                         holdout=VAL,
                         policy={"max_params_mult": 200})
    assert "refusal" not in out, out
    r = s.set_policy("m", growth_params={
        "preference.rule": "thompson",
        "preference.bucket_spec": "b0",
        "preference.min_count": 1,
        "rl.trainer": "ppo",              # both namespaces in
        "rl.kl_ref_coef": 0.1})           # one door call
    assert "refusal" not in r, r
    r = s.study("m", ROWS, steps=200)
    assert "refusal" not in r, r
    pre_mse = _mse(s, "m")
    params0 = s.describe("m")["params_total"]

    # user-controlled growth mid-pretraining (deepen verb)
    g = s.deepen("m", m=4, force=True)
    assert "refusal" not in g, g
    assert s.describe("m")["params_total"] > params0  # L-E2E-1
    _mse(s, "m")                       # serves right after growth
    r = s.study("m", ROWS, steps=300)  # quasi-static continue
    assert "refusal" not in r, r
    post_mse = _mse(s, "m")
    assert post_mse < pre_mse                    # L-E2E-2

    # S-loop lesson + product read
    blob = s._preference_open("m")
    assert "refusal" not in blob, blob
    from reference_net.growthpolicy import preference as prf
    part = prf.GrowthPreference(blob["policy"])
    if blob["blob"]:
        part.restore(blob["blob"])
    part.credit({"event_id": "e2e", "bucket": "deepen",
                 "move": "deepen", "batch": 1,
                 "quoted_gain": 0.0,
                 "window_gains": [pre_mse - post_mse],
                 "credited_gain": pre_mse - post_mse,
                 "advantage": pre_mse - post_mse})
    s._preference_write("m", part.snapshot(), part.audit_events)
    ins = s.preference_inspect("m")
    assert ins["stats"], ins                     # L-E2E-3

    # promote, then rollback with default keep: lessons survive
    c = s.commit("m", note="e2e-v1")
    assert "refusal" not in c, c
    vs = s.get_versions("m")
    active = vs["active"]
    rb = s.rollback("m", active)
    assert "refusal" not in rb, rb
    ins2 = s.preference_inspect("m")
    assert "deepen" in ins2["stats"], ins2       # L-E2E-4

    # ---------------- LEG 2: RL continuation (runbook pattern)
    d = RegimeDispatcher(policy={})
    labeled = [{"input": [0.1], "target": 1.0}]
    rewards = [{"source": "env_return", "value": 1.0}]
    assert d.dispatch(labeled_rows=labeled) == "teach"
    world = StagedExpansionWorld(seed=11, boundary=256,
                                 ep_len=32)
    run = OrganPPORunner(world, seed=12, hidden=10,
                         policy={"rl.trainer": "ppo",
                                 "rl.ent_coef": 0.01})
    # teach-pretrain the policy organ on world-oracle labels
    Xp = np.stack([world.sample_state(i) for i in range(32)])
    yp = np.eye(world.n_actions)[[world.oracle(x) for x in Xp]]
    for _ in range(30):
        run.policy_adapter.organ.train_step(Xp, yp)
    # evidence turns reward-only -> rl phase
    assert d.dispatch(reward_records=rewards) == "rl"
    rules = {e["rule"] for e in d.audit}
    assert rules == {"labeled-first"} and \
        [e["to"] for e in d.audit] == ["teach", "rl"]  # L-E2E-6
    run.audit.extend(d.audit)

    floor = run.mean_recent_return()             # 0 pre-episodes
    for _ in range(6):
        run.train_rounds(1, horizon=256)         # crosses 256
    # user-controlled growth mid-RL + aligned gate
    cand = copy.deepcopy(run)
    cand.policy_adapter.organ.grow(0, hidden=6, force=True)
    p_probe = cand.policy_adapter.probs(Xp)
    assert np.isfinite(p_probe).all()            # L-E2E-10
    assert np.allclose(p_probe.sum(axis=1), 1.0)
    cand.train_rounds(2, horizon=256)
    prov = EvalEpisodeProvider(StagedExpansionWorld,
                               world_seed=11, boundary=256,
                               ep_len=32, eval_seed_base=3)
    prov.align_to(run.world)                     # E-6 law

    class _A:
        def __init__(self, ad):
            self.ad = ad

        def act_probs(self, st):
            return self.ad.probs(
                np.asarray(st, dtype=float)[None])[0]
    v = gate_adjudicate(prov, _A(run.policy_adapter),
                        _A(cand.policy_adapter),
                        {"rl.eval_episode_budget": 4,
                         "rl.eval_window": 4})
    assert v["audit"]["episodes"] == 8           # L-E2E-8
    if v["adopt"]:
        run = cand
    run.audit.append({"kind": "gate_verdict", **v})
    # preference credit of the gate outcome (J-1 pattern)
    part.credit({"event_id": "e2e-rl", "bucket": "grow",
                 "move": "grow", "batch": 6, "quoted_gain": 0.0,
                 "window_gains":
                     [v["score_cand"] - v["score_inc"]],
                 "credited_gain":
                     v["score_cand"] - v["score_inc"],
                 "advantage":
                     v["score_cand"] - v["score_inc"]})
    s._preference_write("m", part.snapshot(), part.audit_events)

    for _ in range(4):
        run.train_rounds(1, horizon=256)
    assert run.mean_recent_return() > floor      # L-E2E-7

    # grpo smoke on the R5 frame
    g2 = OrganPPORunner(StagedExpansionWorld(seed=13, ep_len=32),
                        seed=14, hidden=8,
                        policy={"rl.trainer": "grpo",
                                "rl.n_epochs": 2})
    st = g2.train_rounds(1, horizon=128)
    assert st["epochs_run"] > 0
    run.audit.extend(g2.drain_audit())

    # product observability: persist + replay-read (TR-G4 path)
    w = s._rl_audit_write("m", run.drain_audit())
    assert "refusal" not in w, w
    tail = (s.lc._wdir("m") / "rl_audit.jsonl").read_text()
    events = [json.loads(x) for x in tail.splitlines() if x]
    kinds = {e["kind"] for e in events}
    assert "rl_round" in kinds and "gate_verdict" in kinds \
        and "phase_switch" in kinds              # L-E2E-9
    _mse(s, "m")                                 # still serving
