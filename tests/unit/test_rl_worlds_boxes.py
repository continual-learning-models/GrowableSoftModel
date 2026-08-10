"""Track-B D-P3 boxes: N2 reward worlds + RewardRecord schema
+ TX gates (plan 84 v2.18; doc 86 N2/§3.2/§3.4; doc 88 B-1
exactness contract). Tests-first: RED by missing modules.

T-AX3 method table:
  TW-01 world determinism (same seed => bit-identical
        episode streams, all three worlds) + spec echo
  TW-02 staged-expansion semantics: stage-B reward depends
        on the dims that are SILENT in stage A (hand-
        constructed check via the world's own oracle)
  TW-03 sensor-arrival semantics: pre-arrival sensor dims
        are EXACTLY zero, post-arrival active
  TX-01 exactness under the P-loop: organ policy serving at
        a growth instant between adapter updates — deepen
        bitwise-identical logits; widen (grow port) within
        machine epsilon (doc 88 B-1 contract: pure paths
        bitwise, width riders <= 2.5e-16-class footprint)
  TX-02 channel separation (FR-4.2): the L2 runner consumes
        env_return records ONLY — feeding it a ledger_gain
        record is a LOUD refusal
  TX-05 RewardRecord uniformity (doc 86 §3.2): ONE schema
        validates both sources; malformed records refused
"""
import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "modules" / "RLTrainer"),
           str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _episode_stream(world, n_steps, seed):
    obs = world.reset(seed=seed)
    out = []
    r = np.random.default_rng(seed + 1)
    for _ in range(n_steps):
        a = int(r.integers(world.n_actions))
        obs2, rew, done = world.step(a)
        out.append((obs.copy(), a, rew, done))
        obs = world.reset(seed=None) if done else obs2
    return out


# ------------------------------------------------------ TW-01
def test_tw01_world_determinism_and_spec():
    """[FR-2.1/NFR-1, plan 84 D-P3] TW-01: all three worlds are seeded and replay
    bit-identically; spec() echoes type/dims/actions."""
    from rl_trainer.worlds import (StagedExpansionWorld,
                                   StationaryWorld,
                                   SensorArrivalWorld)
    for W in (StagedExpansionWorld, StationaryWorld,
              SensorArrivalWorld):
        a = _episode_stream(W(seed=3), 200, seed=7)
        b = _episode_stream(W(seed=3), 200, seed=7)
        for (o1, a1, r1, d1), (o2, a2, r2, d2) in zip(a, b):
            assert np.array_equal(o1, o2)
            assert a1 == a2 and r1 == r2 and d1 == d2
        sp = W(seed=3).spec()
        assert sp["obs_dim"] > 0 and sp["n_actions"] >= 2
        assert sp["type"] in ("staged_expansion",
                              "stationary", "sensor_arrival")


# ------------------------------------------------------ TW-02
def test_tw02_staged_expansion_semantics():
    """[FR-2.2, plan 84 D-P3 world W2] TW-02: before the boundary, the world's optimal action
    (its own oracle) NEVER depends on the expansion dims;
    after the boundary it DOES (checked by flipping the
    expansion dims of the same state and asking the oracle)."""
    from rl_trainer.worlds import StagedExpansionWorld
    w = StagedExpansionWorld(seed=5)
    r = np.random.default_rng(0)
    for _ in range(50):
        s = r.normal(size=w.spec()["obs_dim"])
        s2 = s.copy()
        s2[w.expansion_dims] = -s2[w.expansion_dims] + 0.7
        assert w.oracle(s, stage=0) == w.oracle(s2, stage=0)
    diff = sum(w.oracle(s, stage=1) != w.oracle(
        (lambda t: (t.__setitem__(w.expansion_dims,
                                  -t[w.expansion_dims] + 0.7),
                    t)[1])(s.copy()), stage=1)
        for s in r.normal(size=(50, w.spec()["obs_dim"])))
    assert diff > 0


# ------------------------------------------------------ TW-03
def test_tw03_sensor_arrival_semantics():
    """[FR-2.3, plan 84 D-P3 world W3] TW-03: sensor dims are EXACTLY zero pre-arrival and
    nonzero (informative) post-arrival."""
    from rl_trainer.worlds import SensorArrivalWorld
    w = SensorArrivalWorld(seed=4)
    obs = w.reset(seed=2)
    seen_pre = []
    for _ in range(w.arrival_step - 1):
        obs, _, done = w.step(0)
        seen_pre.append(obs[w.sensor_dims].copy())
        if done:
            obs = w.reset(seed=None)
    assert all(np.all(x == 0.0) for x in seen_pre)
    seen_post = []
    for _ in range(50):
        obs, _, done = w.step(0)
        seen_post.append(obs[w.sensor_dims].copy())
        if done:
            obs = w.reset(seed=None)
    assert any(np.any(x != 0.0) for x in seen_post)


# ------------------------------------------------------ TX-01
def test_tx01_exactness_under_ploop_growth():
    """TX-01 (doc 88 B-1 contract): with the L2 runner mid-
    training in a world, apply growth between updates:
    deepen => probe logits BITWISE identical (pure path);
    grow (port widen) => within the machine-eps width-rider
    footprint (<= 5e-15 across the probe batch)."""
    from rl_trainer.worlds import StationaryWorld
    from rl_trainer.runner import OrganPPORunner
    w = StationaryWorld(seed=8)
    run = OrganPPORunner(w, seed=9, hidden=10)
    run.train_rounds(2)                     # some real updates
    probe = np.stack([w.sample_state(i) for i in range(16)])
    pre = run.policy_logits(probe).copy()
    run.policy_adapter.organ.deepen(m=4, force=True)
    post = run.policy_logits(probe)
    assert np.array_equal(pre, post)        # pure path bitwise
    pre2 = run.policy_logits(probe).copy()
    run.policy_adapter.organ.grow(0, hidden=6, force=True)
    post2 = run.policy_logits(probe)
    assert np.max(np.abs(post2 - pre2)) <= 5e-15
    run.train_rounds(1)                     # trains after growth


# ------------------------------------------------------ TX-02
def test_tx02_channel_separation_refusal():
    """TX-02 (FR-4.2): the P-loop runner consumes env_return
    records only — a ledger_gain record is refused LOUDLY."""
    from rl_trainer.records import validate_reward_record
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld
    rec = {"source": "ledger_gain", "object": "structure",
           "scope": "ev-1", "value": 0.2, "baseline": None,
           "life_id": "L1", "episode": 0, "provenance": {}}
    assert validate_reward_record(rec) is None   # valid schema
    run = OrganPPORunner(StationaryWorld(seed=1), seed=2,
                         hidden=8)
    try:
        run.ingest_record(rec)
        assert False, "ledger_gain accepted by P-loop"
    except ValueError as e:
        assert "env_return" in str(e)


# ------------------------------------------------------ TX-05
def test_tx05_reward_record_uniformity():
    """TX-05 (doc 86 §3.2): ONE schema, two sources — both
    validate; wrong source/missing field refused with the
    offending key named."""
    from rl_trainer.records import validate_reward_record
    s_rec = {"source": "ledger_gain", "object": "structure",
             "scope": "event-7", "value": 0.31,
             "baseline": 0.25, "life_id": "L1", "batch": 12,
             "provenance": {"ledger": "sha"}}
    p_rec = {"source": "env_return", "object": "weights",
             "scope": "ep-99", "value": 187.0,
             "baseline": None, "life_id": "L1",
             "episode": 99, "provenance": {"seed": 5}}
    assert validate_reward_record(s_rec) is None
    assert validate_reward_record(p_rec) is None
    bad = dict(p_rec, source="mystery")
    msg = validate_reward_record(bad)
    assert msg and "source" in msg
    bad2 = {k: v for k, v in p_rec.items() if k != "value"}
    msg2 = validate_reward_record(bad2)
    assert msg2 and "value" in msg2


# ------------------------------------------------------ TB-P09
def test_tbp09_width_fair_health_instruments():
    """TB-P09 (plan 84 D-P3 'width-fair health metric per
    doc 88 B-2'): the registered erank/width ratio was
    WIDTH-CONFOUNDED (trivially favors narrow nets); the
    width-fair form is the ABSOLUTE effective dimension +
    dead-unit fraction.
    WORKSHEET (mpmath-30): H = [[2,0,0],[0,1,0],[0,0,0],
    [0,0,0]] -> singular values (2,1,0); p = (0.8, 0.2);
    entropy = 0.50040242353818788;
    eRank = exp(entropy) = 1.6493848884661178 (absolute —
    NOT divided by width). Dead units (col std < 0.01,
    population std): col stds = (0.866025403784,
    0.433012701892, 0.0) -> dead fraction = 1/3.
    ORGAN ROUTE (second route, T-AX3): health(organ, probe)
    on a live organ returns finite absolute eRank in
    (0, hidden], dead_frac in [0,1], width echoed — and
    after a widen the ABSOLUTE eRank does not shrink just
    because width grew (the B-2 confound is gone by
    construction: no /width anywhere)."""
    from rl_trainer.instruments import (effective_dim,
                                        dead_unit_fraction,
                                        organ_health)
    H = np.array([[2.0, 0, 0], [0, 1.0, 0],
                  [0, 0, 0], [0, 0, 0]])
    assert abs(effective_dim(H) - 1.6493848884661178) < 1e-12
    assert abs(dead_unit_fraction(H) - 1.0 / 3.0) < 1e-15
    from reference_net.net import Network
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 3))
    organ = Network(3, 8, seed=5, out_width=2)
    organ.train_step(X, rng.normal(size=(32, 2)))
    h1 = organ_health(organ, X)
    assert 0.0 < h1["effective_dim"] <= h1["width"] == 8
    assert 0.0 <= h1["dead_frac"] <= 1.0
    organ.grow(0, hidden=6, force=True)      # widen
    h2 = organ_health(organ, X)
    assert h2["width"] >= h1["width"]
    assert h2["effective_dim"] >= h1["effective_dim"] - 1e-9


# ------------------------------------------------------ TB-P10
def test_tbp10_335_schema_conformance():
    """TB-P10 (D-P3 completeness re-check, owner order;
    ACCEPTANCE BASIS = REQUIREMENTS doc 89, owner ruling —
    the design doc is shape reference only):
    (i) FR-3.0 grpo group baseline needs episode identity;
    (ii) NFR-1 determinism needs rollout provenance;
    (iii) FR-7 observability needs update stats a third
    party can check. Concretely (shapes per doc 86 §3.35):
    (a) RolloutBatch from the runner carries episode_ids,
        group_ids (None for ppo) and seed_tag alongside
        obs/actions/rewards/dones/logp/values;
    (b) TrainerPlug UpdateStats carries kl, epochs_run,
        clip_frac, loss_terms{policy,value,entropy},
        early_stopped (superset keys approx_kl/n_updates/
        kl_stopped retained);
    (c) trainers expose spec() -> {trainer, keys echo}.
    Route: schema assertions + value sanity (clip_frac in
    [0,1]; entropy >= 0; episode_ids nondecreasing,
    incremented exactly at done edges)."""
    from rl_trainer.worlds import StationaryWorld
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.trainers import PPOTrainer, GRPOTrainer
    run = OrganPPORunner(StationaryWorld(seed=3), seed=4,
                         hidden=8)
    ro = run.collect(96)
    for k in ("obs", "actions", "rewards", "dones", "logp",
              "values", "episode_ids", "group_ids",
              "seed_tag", "last_value"):
        assert k in ro, k
    eid = ro["episode_ids"]
    assert np.all(np.diff(eid) >= 0)
    edges = np.flatnonzero(ro["dones"][:-1])
    assert np.array_equal(np.flatnonzero(np.diff(eid) == 1),
                          edges)
    assert ro["group_ids"] is None            # ppo rollout
    assert isinstance(ro["seed_tag"], str) and ro["seed_tag"]
    r = np.random.default_rng(1)
    N = 64
    roll = {"obs": r.normal(size=(N, 3)),
            "actions": r.integers(0, 2, N),
            "rewards": r.normal(size=N),
            "dones": np.zeros(N, bool),
            "logp": r.normal(-0.7, 0.1, N),
            "values": r.normal(size=N), "last_value": 0.0}
    t = PPOTrainer(3, 2, seed=2, policy={"rl.n_epochs": 2})
    st = t.step(roll)
    for k in ("kl", "epochs_run", "clip_frac", "loss_terms",
              "early_stopped"):
        assert k in st, k
    assert 0.0 <= st["clip_frac"] <= 1.0
    lt = st["loss_terms"]
    assert set(lt) == {"policy", "value", "entropy"}
    assert lt["entropy"] >= 0.0
    sp = t.spec()
    assert sp["trainer"] == "ppo" and "rl.clip" in sp["keys"]
    g = GRPOTrainer(3, 2, seed=2)
    assert g.spec()["trainer"] == "grpo"


# ------------------------------------------------------ TB-P12
def test_tbp12_runner_grpo_mode():
    """TB-P12 (D-P5 precursor; FR-3.0 both trainers on the
    ORGAN): rl.trainer='grpo' switches the runner to the
    critic-free path — complete-episode group advantages, NO
    value-organ updates (its params stay bit-identical across
    rounds), policy organ still trains; unknown trainer value
    refused at the runner door too."""
    import copy
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld
    run = OrganPPORunner(StationaryWorld(seed=3), seed=6,
                         hidden=8,
                         policy={"rl.trainer": "grpo"})
    v_before = copy.deepcopy(np.asarray(
        run.value_adapter.organ.W1))
    p_before = copy.deepcopy(np.asarray(
        run.policy_adapter.organ.W1))
    run.train_rounds(2)
    assert np.array_equal(
        v_before, np.asarray(run.value_adapter.organ.W1))
    assert not np.array_equal(
        p_before, np.asarray(run.policy_adapter.organ.W1))
    try:
        OrganPPORunner(StationaryWorld(seed=3), seed=6,
                       hidden=8, policy={"rl.trainer": "sarsa"})
        assert False, "unknown trainer accepted"
    except ValueError as e:
        assert "trainer" in str(e)
