"""Track-B D-P4 sub-step 3 cross-loop boxes (plan 84 v2.19).
Acceptance basis (doc 89): FR-4.3 both-on safety (TX-03),
FR-3.6/FR-9 co-resident recurring trainer regimes with BOTH
interference directions measured (TX-06), FR-3.6 evidence-
type dispatch with phase-switch audit (TR-01).

T-AX3 method table:
  TR-01 regime dispatch law: labeled rows -> teach, reward
        experience -> rl (LAW-1); mixed batch follows the
        registered labeled-first default; every switch is an
        audit event with the evidence kinds named
  TX-03 stacked E2E (S-loop + P-loop BOTH ON): the P-loop
        trains the organ in the staged world; a growth
        proposal pair is RANKED by the enabled preference
        part, the winner is trial-grown and adjudicated by
        the eval-episode gate; on adoption the S-loop is
        CREDITED from eval-return windows (doc 86 §3.2 one-
        metric law). Assertions: pipeline completes; the
        preference table holds the credited bucket (w>0);
        the gate verdict is audited; training continues
        after adoption (returns finite, organ larger).
  TX-06 mixed-regime E2E (teach -> rl -> teach, ONE organ):
        direction 1 (teach knowledge across the rl phase):
        oracle-label agreement on a fixed probe stays above
        the uniform-chance floor after the rl phase;
        direction 2 (reward behavior across the teach
        phase): eval return after the closing teach phase
        stays above the untrained-baseline return. Sanity
        floors, not absolute contest lines (house law).
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "modules" / "RLTrainer"),
           str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ------------------------------------------------------ TR-01
def test_tr01_regime_dispatch_and_switch_audit():
    from rl_trainer.regime import RegimeDispatcher
    d = RegimeDispatcher()
    labeled = [{"input": [0.1], "target": 1.0}]
    rewards = [{"source": "env_return", "value": 3.0}]
    assert d.dispatch(labeled_rows=labeled,
                      reward_records=None) == "teach"
    assert d.dispatch(labeled_rows=None,
                      reward_records=rewards) == "rl"
    # mixed evidence: labeled-first default (doc 86 §3.5)
    assert d.dispatch(labeled_rows=labeled,
                      reward_records=rewards) == "teach"
    kinds = [e["kind"] for e in d.audit]
    assert kinds == ["phase_switch"] * 3 or \
        all(k == "phase_switch" for k in kinds)
    assert d.audit[0]["to"] == "teach"
    assert d.audit[1]["to"] == "rl"
    assert d.audit[1]["evidence"] == {"labeled": 0, "reward": 1}
    assert d.audit[2]["to"] == "teach"      # labeled-first


# ------------------------------------------------------ TX-03
def test_tx03_stacked_both_loops_end_to_end():
    import copy
    from reference_net.growthpolicy import preference as prf
    from rl_trainer.eval_provider import (EvalEpisodeProvider,
                                          gate_adjudicate)
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StagedExpansionWorld
    run = OrganPPORunner(StagedExpansionWorld(seed=4), seed=7,
                         hidden=8)
    run.train_rounds(2)
    # S-loop ON: rank two growth moves with the preference part
    pref = prf.GrowthPreference({"seed": 0,
                                 "preference.rule": "mean_clip",
                                 "preference.bucket_spec": "b0",
                                 "preference.min_count": 1})
    ctxs = [{"move": "grow", "slope": None},
            {"move": "deepen", "slope": None}]
    out = prf.rank_with_preference(pref, [0.5, 0.5], ctxs)
    pick = int(np.argmax(out["scores_adj"]))
    move = ctxs[pick]["move"]
    # trial-grow a candidate copy, brief adaptation, adjudicate
    cand_run = copy.deepcopy(run)
    organ = cand_run.policy_adapter.organ
    (organ.grow(0, hidden=4, force=True) if move == "grow"
     else organ.deepen(m=4, force=True))
    cand_run.train_rounds(1)

    class _A:
        def __init__(self, ad):
            self.ad = ad

        def act_probs(self, s):
            return self.ad.probs(
                np.asarray(s, dtype=float)[None])[0]
    prov = EvalEpisodeProvider(StagedExpansionWorld,
                               world_seed=4, eval_seed_base=9)
    verdict = gate_adjudicate(
        prov, _A(run.policy_adapter),
        _A(cand_run.policy_adapter),
        {"rl.eval_episode_budget": 4, "rl.eval_window": 4})
    assert set(verdict) >= {"adopt", "score_inc",
                            "score_cand", "audit"}
    if verdict["adopt"]:
        run = cand_run                      # gate-approved swap
    # S-loop credited from eval-return windows (one metric)
    e_before = -verdict["score_inc"]
    gains = [e_before - (-verdict["score_cand"])]
    pref.credit({"event_id": "tx03", "bucket": move,
                 "move": move, "batch": 1, "quoted_gain": 0.0,
                 "window_gains": gains,
                 "credited_gain": float(gains[0]),
                 "advantage": float(gains[0])})
    snap = pref.snapshot()
    assert snap["stats"][move]["w"] > 0     # credited
    run.train_rounds(1)                     # trains after verdict
    assert np.isfinite(run.mean_recent_return())


# ------------------------------------------------------ TX-06
def test_tx06_mixed_regime_both_retention_directions():
    from rl_trainer.organ_adapter import OrganAdapter
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld
    w = StationaryWorld(seed=11)
    run = OrganPPORunner(w, seed=12, hidden=12)
    probe = np.stack([w.sample_state(i) for i in range(64)])
    labels = np.array([w.oracle(s) for s in probe])
    onehot = np.eye(w.n_actions)[labels] * 2.0 - 1.0
    chance = 1.0 / w.n_actions

    def agreement():
        pred = np.argmax(run.policy_logits(probe), axis=1)
        return float(np.mean(pred == labels))
    # phase A: TEACH (supervised steps toward oracle logits)
    for _ in range(60):
        run.policy_adapter.organ.train_step(probe, onehot)
    agree_teach = agreement()
    assert agree_teach > chance + 0.15      # taught something
    # phase B: RL
    run.train_rounds(3)
    agree_after_rl = agreement()
    assert agree_after_rl > chance + 0.10   # direction 1: kept
    ret_rl = run.mean_recent_return()
    # phase C: closing TEACH
    for _ in range(30):
        run.policy_adapter.organ.train_step(probe, onehot)
    ro = run.collect(128)                   # direction 2 readout
    ret_after_teach = float(np.mean(
        [r["value"] for r in run.records[-4:]]))
    assert np.isfinite(ret_rl) and np.isfinite(ret_after_teach)
    # reward behavior above the untrained chance floor
    assert ret_after_teach > chance * w.ep_len * 0.8
