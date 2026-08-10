"""Track-B D-P4 sub-step 1 boxes: N4 EvalEpisodeProvider +
gate adjudication in reward worlds (plan 84 v2.18; acceptance
basis = doc 89 FR-3.4 [gate adjudicates adoptions in reward
worlds over a QUARANTINED evaluation-episode stream, paired
return comparison, cost recorded (L17)] and FR-4.1 [gate over
a pluggable evaluation-stream provider]; shapes per doc 86
§3.35). Tests-first: RED by missing rl_trainer.eval_provider.

T-AX3 method table:
  TE-01 provider contract: evaluate_pair returns the §3.35
        shape; QUARANTINE proven by seed-set disjointness
        (eval seeds never appear in rollout streams) and by
        determinism (same pair scored twice = identical);
        cost recorded (episodes spent, L17)
  TE-02 paired verdict law: a strictly-better candidate is
        adopted, a strictly-worse one refused, tie follows
        the registered tolerance (hand-constructed organs:
        the world's own oracle vs an untrained newborn)
  TX-04 governed silence: on the stationary world a
        do-nothing candidate (bitwise clone) NEVER wins
        adoption (no free adoptions from noise) across
        multiple paired evaluations
"""
import copy
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "modules" / "RLTrainer"),
           str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _OracleActor:
    """Serves the world's own oracle (upper anchor)."""

    def __init__(self, world):
        self.world = world

    def act_probs(self, s):
        p = np.zeros(self.world.n_actions)
        p[self.world.oracle(s)] = 1.0
        return p


class _UniformActor:
    """Uninformed actor (lower anchor)."""

    def __init__(self, world):
        self.world = world

    def act_probs(self, s):
        return np.full(self.world.n_actions,
                       1.0 / self.world.n_actions)


# ------------------------------------------------------ TE-01
def test_te01_provider_contract_quarantine_cost():
    """TE-01 (FR-3.4/FR-4.1): shape, quarantine, determinism,
    cost. Quarantine route: the provider's episode seeds are
    drawn from a DEDICATED namespace disjoint from the
    worlds' rollout stream namespace (2000+seed) — asserted
    on the provenance echo; determinism route: identical
    calls => identical scores (bitwise floats)."""
    from rl_trainer.eval_provider import EvalEpisodeProvider
    from rl_trainer.worlds import StationaryWorld
    w = StationaryWorld(seed=6)
    prov = EvalEpisodeProvider(StationaryWorld, world_seed=6,
                               eval_seed_base=17)
    a, b = _OracleActor(w), _UniformActor(w)
    out = prov.evaluate_pair(a, b, {"rl.eval_episode_budget": 6,
                                    "rl.eval_window": 6})
    for k in ("score_inc", "score_cand", "n", "cost",
              "provenance"):
        assert k in out, k
    assert out["n"] == 6
    assert out["cost"]["episodes"] == 12          # both sides
    seeds = out["provenance"]["episode_seeds"]
    assert len(set(seeds)) == 6                   # dedicated set
    assert all(s >= 9_000_000 for s in seeds)     # namespace
    out2 = prov.evaluate_pair(a, b,
                              {"rl.eval_episode_budget": 6,
                               "rl.eval_window": 6})
    assert out2["score_inc"] == out["score_inc"]  # bitwise
    assert out2["score_cand"] == out["score_cand"]


# ------------------------------------------------------ TE-02
def test_te02_paired_verdict_law():
    """TE-02 (FR-3.4 adjudication): oracle vs uniform on the
    stationary world — oracle-as-candidate is ADOPTED,
    uniform-as-candidate is REFUSED (strictly-better law with
    the registered tolerance rl.eval_tol, default 1.0 =
    strictly greater)."""
    from rl_trainer.eval_provider import (EvalEpisodeProvider,
                                          gate_adjudicate)
    from rl_trainer.worlds import StationaryWorld
    w = StationaryWorld(seed=6)
    prov = EvalEpisodeProvider(StationaryWorld, world_seed=6,
                               eval_seed_base=3)
    pol = {"rl.eval_episode_budget": 8, "rl.eval_window": 8}
    up = gate_adjudicate(prov, _UniformActor(w),
                         _OracleActor(w), pol)
    assert up["adopt"] is True
    assert up["score_cand"] > up["score_inc"]
    down = gate_adjudicate(prov, _OracleActor(w),
                           _UniformActor(w), pol)
    assert down["adopt"] is False
    assert "audit" in up and up["audit"]["episodes"] == 16


# ------------------------------------------------------ TX-04
def test_tx04_governed_silence_no_free_adoptions():
    """TX-04 (plan 84 gate; FR-5 refusals): a candidate that
    is a BITWISE CLONE of the incumbent must never win
    adoption (paired scores tie exactly on the shared seed
    set => strictly-better law refuses), across 5 repeated
    adjudications on the stationary world."""
    from rl_trainer.eval_provider import (EvalEpisodeProvider,
                                          gate_adjudicate)
    from rl_trainer.worlds import StationaryWorld
    from rl_trainer.runner import OrganPPORunner
    run = OrganPPORunner(StationaryWorld(seed=2), seed=5,
                         hidden=8)
    run.train_rounds(1)

    class _OrganActor:
        def __init__(self, adapter):
            self.adapter = adapter

        def act_probs(self, s):
            return self.adapter.probs(
                np.asarray(s, dtype=float)[None])[0]
    inc = _OrganActor(run.policy_adapter)
    cand = _OrganActor(copy.deepcopy(run.policy_adapter))
    prov = EvalEpisodeProvider(StationaryWorld, world_seed=2,
                               eval_seed_base=41)
    pol = {"rl.eval_episode_budget": 4, "rl.eval_window": 4}
    for k in range(5):
        out = gate_adjudicate(prov, inc, cand, pol)
        assert out["adopt"] is False, k
        assert out["score_cand"] == out["score_inc"]


# ------------------------------------------------------ TB-P11
def test_tbp11_facade_rl_key_doors():
    """TB-P11 (D-P4 s2; FR-6 rl.* configuration through the
    EXISTING set_policy surface; §4.5-style loud refusals at
    BOTH policy doors, the preference-door precedent):
    - unknown rl.* key -> refusal naming it;
    - invalid value (rl.lr <= 0, rl.gamma > 1,
      gate.eval_stream not in {labeled_slice, eval_episodes})
      -> refusal naming key and valid range;
    - valid rl.*/gate.eval_stream keys -> accepted (no
      refusal) and echoed by set_policy."""
    import tempfile
    from core.facade import System
    from generator.config import Config
    with tempfile.TemporaryDirectory() as tmp:
        s = System(Config.from_env(backend="mlp",
                                   models_root=Path(tmp)))
        s.create_model("m1")
        r1 = s.set_policy("m1",
                          growth_params={"rl.lr": -1.0})
        assert "refusal" in r1 and "rl.lr" in r1["refusal"]
        r2 = s.set_policy("m1",
                          growth_params={"rl.bogus": 1})
        assert "refusal" in r2 and "rl.bogus" in r2["refusal"]
        r3 = s.set_policy("m1", growth_params={
            "gate.eval_stream": "bogus"})
        assert "refusal" in r3 and "eval_episodes" in \
            r3["refusal"]
        r4 = s.set_policy("m1", growth_params={
            "rl.clip": 0.3, "rl.trainer": "grpo",
            "gate.eval_stream": "eval_episodes"})
        assert "refusal" not in r4
        r5 = s.create_model("m2", policy={"growth_params":
                                          {"rl.gamma": 2.0}})
        assert "refusal" in r5 and "rl.gamma" in r5["refusal"]
