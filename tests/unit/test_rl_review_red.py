"""Owner-ordered three-pass review findings — RED boxes ONLY
(2026-07-29; tests-first law: these boxes pin every finding
BEFORE any core-code change; each cites its requirement
clause — acceptance basis doc 89, owner ruling).

FINDINGS PINNED HERE:
  GK-01 STANDING DEAD-KEY MACHINE GATE: every registered
        rl.*/gate.* key must have >=1 consumer in
        rl_trainer beyond defaults/validation (the class
        that let F-1/F-2/F-3 through; institutionalized)
  TR-F1 entropy term EFFECT (FR-3.1 "entropy terms"):
        ent_coef>0 must change the update and raise policy
        entropy vs ent_coef=0 on identical data/seed
  TR-F2 rl.horizon EFFECT: the runner's rollout length must
        follow the key (episode count per round changes)
  TR-F3 gate.eval_stream DISPATCH (FR-4.1): the key must
        select the stream — adjudication echoes the stream
        it used; eval_episodes without a provider refuses
  TR-F4 organ-path canonical epochs/KL (FR-3.1 "KL early
        stop"): train_rounds honors rl.n_epochs and stops
        early on rl.target_kl, reporting stats
  TR-F6 KL-to-incumbent anchor (FR-3.6 / FR-9.3
        regularization family): rl.kl_ref_coef registered;
        large coef keeps the policy measurably closer to
        the incumbent reference than coef=0 on identical
        data/seed
  TR-F7 regime policy override + interleave (FR-3.6):
        rl.regime override forces the phase regardless of
        evidence; interleave ratio yields the registered
        teach:rl alternation
"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "modules" / "RLTrainer"),
           str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _roll(n=128, seed=1, obs=3, acts=2):
    r = np.random.default_rng(seed)
    return {"obs": r.normal(size=(n, obs)),
            "actions": r.integers(0, acts, n),
            "rewards": r.normal(size=n),
            "dones": np.zeros(n, bool),
            "logp": r.normal(-0.7, 0.1, n),
            "values": r.normal(size=n), "last_value": 0.0}


# ------------------------------------------------------ GK-01
def test_gk01_standing_dead_key_gate():
    """[NFR-5(iii) doc 89 key-effectiveness] GK-01: THE MACHINE GATE — a registered key without a
    consumer is a defect by construction. Consumer = any
    read of the key string in rl_trainer outside defaults.py
    (validation) — facade validation does not count."""
    import rl_trainer
    from rl_trainer.defaults import RL_DEFAULTS, GATE_DEFAULTS
    pkg = Path(rl_trainer.__file__).parent
    src = "".join(p.read_text() for p in pkg.glob("*.py")
                  if p.name != "defaults.py")
    dead = [k for k in list(RL_DEFAULTS) + list(GATE_DEFAULTS)
            if f'"{k}"' not in src]
    assert not dead, f"dead keys (registered, no consumer): {dead}"


# ------------------------------------------------------ TR-F1
def test_trf1_entropy_term_has_effect():
    """[FR-4.2 ent term; plan 95 Formula A]"""
    from rl_trainer.trainers import PPOTrainer
    a = PPOTrainer(3, 2, seed=7, policy={"rl.ent_coef": 0.0})
    b = PPOTrainer(3, 2, seed=7, policy={"rl.ent_coef": 0.5})
    a.step(_roll())
    b.step(_roll())
    diff = any(not np.array_equal(a.pi[i][0], b.pi[i][0])
               for i in range(3))
    assert diff, "ent_coef has NO effect (FR-3.1 entropy term)"
    X = _roll()["obs"]
    from rl_trainer.math import entropy
    assert entropy(b.policy(X)) > entropy(a.policy(X))  # strict
    # default 0.0 keeps the blueprint path untouched
    c = PPOTrainer(3, 2, seed=7)
    c.step(_roll())
    assert all(np.array_equal(a.pi[i][0], c.pi[i][0])
               for i in range(3))
    # FD PARITY REFEREE (G-A): the implemented analytic
    # entropy gradient must match central finite differences
    # of H(softmax(z)) at 1e-9 on a random case.
    from rl_trainer.math import entropy_grad_logits
    rng = np.random.default_rng(33)
    z = rng.normal(size=(1, 4))
    ana = entropy_grad_logits(
        np.exp(z - z.max()) / np.exp(z - z.max()).sum())
    h = 1e-6
    fd = np.zeros(4)
    for k in range(4):
        zp, zm = z.copy(), z.copy()
        zp[0, k] += h
        zm[0, k] -= h
        def _H(zz):
            e = np.exp(zz - zz.max())
            p = e / e.sum()
            return float(-(p * np.log(p)).sum())
        fd[k] = (_H(zp) - _H(zm)) / (2 * h)
    assert np.allclose(ana[0], fd, atol=1e-9)


# ------------------------------------------------------ TR-F2
def test_trf2_horizon_key_effect():
    """[FR-4.4 rl.horizon; doc 89 NFR-5(iii)]"""
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld
    r64 = OrganPPORunner(StationaryWorld(seed=2), seed=3,
                         hidden=8, policy={"rl.horizon": 64})
    r64.train_rounds(1)
    n64 = len(r64.records)
    r256 = OrganPPORunner(StationaryWorld(seed=2), seed=3,
                          hidden=8, policy={"rl.horizon": 256})
    r256.train_rounds(1)
    n256 = len(r256.records)
    assert n64 == 2 and n256 == 8, (
        f"rl.horizon has no effect: {n64} vs {n256} episodes "
        "(64/32=2, 256/32=8 expected)")


# ------------------------------------------------------ TR-F3
def test_trf3_eval_stream_key_dispatch():
    """[FR-3.3 gate.eval_stream; doc 86 SS3.3]"""
    from rl_trainer.eval_provider import (EvalEpisodeProvider,
                                          adjudicate_with_policy)
    from rl_trainer.worlds import StationaryWorld

    class _U:
        def act_probs(self, s):
            return np.full(3, 1.0 / 3.0)
    prov = EvalEpisodeProvider(StationaryWorld, world_seed=1,
                               eval_seed_base=5)
    pol_ep = {"gate.eval_stream": "eval_episodes",
              "rl.eval_episode_budget": 3, "rl.eval_window": 3}
    out = adjudicate_with_policy(pol_ep, _U(), _U(),
                                 provider=prov)
    assert out["stream"] == "eval_episodes"
    assert out["adopt"] is False               # exact tie
    # eval_episodes selected but NO provider -> loud refusal
    try:
        adjudicate_with_policy(pol_ep, _U(), _U(), provider=None)
        assert False, "no-provider accepted"
    except ValueError as e:
        assert "provider" in str(e)
    # labeled_slice selected -> this dispatcher declines and
    # NAMES the existing holdout gate as the owner of that
    # stream (single-gate law; no second comparison logic)
    out2 = adjudicate_with_policy({"gate.eval_stream":
                                   "labeled_slice"},
                                  _U(), _U(), provider=prov)
    assert out2["stream"] == "labeled_slice"
    assert "holdout" in out2["refusal"]


# ------------------------------------------------------ TR-F4
def test_trf4_runner_honors_epochs_and_kl():
    """[FR-4.3 canonical epochs+KL; doc 86 SS3.35]"""
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld
    stats = OrganPPORunner(
        StationaryWorld(seed=4), seed=5, hidden=8,
        policy={"rl.n_epochs": 3}).train_rounds(1)
    assert isinstance(stats, dict) and stats["epochs_run"] == 3
    stats2 = OrganPPORunner(
        StationaryWorld(seed=4), seed=5, hidden=8,
        policy={"rl.n_epochs": 5,
                "rl.target_kl": 1e-12}).train_rounds(1)
    assert stats2["epochs_run"] < 5
    assert stats2["kl_stopped"] is True


# ------------------------------------------------------ TR-F6
def test_trf6_kl_to_incumbent_anchor():
    """[FR-4.5 KL anchor; design 86 LAW-3(ii); plan 95 Formula B]"""
    from rl_trainer.defaults import RL_DEFAULTS
    assert "rl.kl_ref_coef" in RL_DEFAULTS, \
        "FR-3.6/FR-9.3 anchor key not registered"
    assert RL_DEFAULTS["rl.kl_ref_coef"] == 0.0   # off default
    from rl_trainer.trainers import PPOTrainer

    def drift(coef):
        t = PPOTrainer(3, 2, seed=9,
                       policy={"rl.kl_ref_coef": coef})
        ref = [W.copy() for W, _ in t.pi]
        for k in range(4):
            t.step(_roll(seed=20 + k))
        return sum(float(np.abs(W - r).sum())
                   for (W, _), r in zip(t.pi, ref))
    d_free = drift(0.0)
    d_anch = drift(50.0)
    assert d_anch > 0.0, "anchor must not freeze learning"
    assert d_anch < d_free * 0.8, (
        f"anchor has no effect: drift {d_anch} vs {d_free}")
    # FD PARITY REFEREE (G-A): implemented analytic KL grad
    # vs central finite differences of KL(softmax(z)||q).
    from rl_trainer.math import kl_grad_logits
    rng = np.random.default_rng(44)
    z = rng.normal(size=(1, 4))
    zr = rng.normal(size=(1, 4))

    def _sm(a):
        e = np.exp(a - a.max())
        return e / e.sum()
    q = _sm(zr[0])[None]
    ana = kl_grad_logits(_sm(z[0])[None], q)

    def _KL(zz):
        p = _sm(zz[0])
        return float((p * (np.log(p) - np.log(q[0]))).sum())
    h = 1e-6
    fd = np.zeros(4)
    for k in range(4):
        zp, zm = z.copy(), z.copy()
        zp[0, k] += h
        zm[0, k] -= h
        fd[k] = (_KL(zp) - _KL(zm)) / (2 * h)
    assert np.allclose(ana[0], fd, atol=1e-9)


# ------------------------------------------------------ TR-F7
def test_trf7_regime_override_and_interleave():
    """[FR-5.1/5.2 regime keys; doc 86 SS4]"""
    from rl_trainer.regime import RegimeDispatcher
    labeled = [{"input": [0.1], "target": 1.0}]
    rewards = [{"source": "env_return", "value": 3.0}]
    # policy override (FR-3.6): rl.regime forces the phase
    d = RegimeDispatcher(policy={"rl.regime": "rl"})
    assert d.dispatch(labeled_rows=labeled,
                      reward_records=rewards) == "rl"
    d2 = RegimeDispatcher(policy={"rl.regime": "teach"})
    assert d2.dispatch(labeled_rows=None,
                       reward_records=rewards) == "teach"
    # interleave ratio (FR-3.6 refresher interleave): with
    # rl.interleave = [2, 1], mixed evidence yields the
    # registered teach,teach,rl repeating schedule
    d3 = RegimeDispatcher(policy={"rl.interleave": [2, 1]})
    seq = [d3.dispatch(labeled_rows=labeled,
                       reward_records=rewards)
           for _ in range(6)]
    assert seq == ["teach", "teach", "rl",
                   "teach", "teach", "rl"]
    # audit still names evidence + the deciding rule
    assert all(e["kind"] == "phase_switch" for e in d3.audit)


# ------------------------------------------------------ TS-F1
def test_tsf1_system_level_door_to_organ_effect():
    """[FR-6 both-doors law; doc 89 acceptance basis] TS-F1 (SYSTEM-LEVEL pin, four-axis law T-AX2): the
    FULL product path — the facade policy door ACCEPTS
    rl.ent_coef (validated, echoed), and the same key fed to
    the end-to-end organ training pipeline MUST change
    behavior. Today the door says yes and the pipeline
    ignores it — the system-level shape of F-1."""
    import tempfile
    from core.facade import System
    from generator.config import Config
    with tempfile.TemporaryDirectory() as tmp:
        s = System(Config.from_env(backend="mlp",
                                   models_root=Path(tmp)))
        s.create_model("m1")
        r = s.set_policy("m1", growth_params={
            "rl.ent_coef": 0.5})
        assert "refusal" not in r          # door accepts
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld
    a = OrganPPORunner(StationaryWorld(seed=5), seed=6,
                       hidden=8, policy={"rl.ent_coef": 0.0})
    b = OrganPPORunner(StationaryWorld(seed=5), seed=6,
                       hidden=8, policy={"rl.ent_coef": 0.5})
    a.train_rounds(2)
    b.train_rounds(2)
    assert not np.array_equal(
        np.asarray(a.policy_adapter.organ.W1),
        np.asarray(b.policy_adapter.organ.W1)), (
        "SYSTEM-LEVEL: door-accepted key has zero effect on "
        "the end-to-end organ pipeline (F-1)")
