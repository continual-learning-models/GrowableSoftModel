"""R3 E-0 — the three permanent machine gates (plan 96 v1.6;
owner order: gates FIRST, and they must go RED on the known
G-1/G-2/G-6 before any fix lands — a gate that doesn't bite
is not a gate).

 GK-01x DEAD-KEY GATE, ALL NAMESPACES: every registered key
   (preference.* / rl.* / gate.*) must be READ somewhere
   outside its registry line and outside the validators.
 GK-02 OPTION-COVERAGE GATE: every validator-accepted enum
   value maps to verified behavior or a loud refusal —
   table-driven, per-option probes.
 GK-03 ALL-KEY REFUSAL GATE: every registered key refuses a
   type-garbage value at the validator (loud, §4.5).
"""
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "modules" / "RLTrainer"),
           str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _strip_validator(src, fn_name):
    return re.sub(rf"def {fn_name}\(.*?(?=\ndef |\Z)", "", src,
                  flags=re.S)


# ----------------------------------------------------- GK-01x
def test_gk01x_dead_key_gate_all_namespaces():
    import rl_trainer
    from rl_trainer.defaults import RL_DEFAULTS, GATE_DEFAULTS
    from reference_net.growthpolicy import preference as prf
    rl_src = "".join(p.read_text() for p in
                     Path(rl_trainer.__file__).parent.glob("*.py")
                     if p.name != "defaults.py")
    gp_dir = Path(prf.__file__).parent
    pref_src = "".join(p.read_text() for p in gp_dir.glob("*.py"))
    pref_src = _strip_validator(pref_src,
                                "validate_preference_policy")
    # facade counts for wiring keys consumed at verb level
    pref_src += Path(_ROOT / "core" / "facade.py").read_text()
    dead = []
    for k in prf.PREFERENCE_DEFAULTS:
        short = k.split(".", 1)[1]
        body = pref_src.replace(f'"{k}":', "", 1)  # registry line
        if f'"{k}"' not in body and f'_p("{short}")' not in body \
                and f'"{short}"' not in body:
            dead.append(k)
    for k in list(RL_DEFAULTS) + list(GATE_DEFAULTS):
        if f'"{k}"' not in rl_src:
            dead.append(k)
    assert not dead, f"dead keys: {dead}"


# ----------------------------------------------------- GK-02
def test_gk02_option_coverage_gate():
    """Every accepted enum option → live behavior probe or
    loud refusal. Table = the S4 closure surface."""
    from reference_net.growthpolicy import preference as prf
    from rl_trainer.defaults import validate_rl_policy
    failures = []
    # (a) reserved values refuse loudly
    if prf.validate_preference_policy(
            {"preference.bucket_spec": "b2"}) is None:
        failures.append("b2 accepted")
    # (b) batches:N must be EFFECTIVE (quota refreshes)
    p = prf.GrowthPreference({"seed": 0,
                              "preference.rule": "thompson",
                              "preference.min_count": 1,
                              "preference.explore_quota": 1,
                              "preference.quota_window":
                              "batches:2"})
    p._force_event_stats(5, 4.0, 0.0, 0.01)
    p._force_stats("grow|s1", w=5.0, m=0.5, v=0.0, n_raw=5)
    g0 = p.explore_offer({"move": "grow", "slope": 0.0,
                          "batch": 0}, raw_eff=-0.2)
    g9 = p.explore_offer({"move": "grow", "slope": 0.0,
                          "batch": 9}, raw_eff=-0.2)
    if not (g0 is True and g9 is True):
        failures.append(f"batches:N ineffective (g0={g0}, "
                        f"g9={g9})")
    # (c) rollback_mode must be WIRED at the rollback verb
    fac = Path(_ROOT / "core" / "facade.py").read_text()
    rb = re.search(r"def rollback\(.*?(?=\n    def )", fac,
                   re.S).group(0)
    if "apply_rollback" not in rb and "rollback_mode" not in rb:
        failures.append("rollback_mode unwired at the verb")
    # (d) rl enums live probes (existing behavior — must stay)
    from rl_trainer.regime import RegimeDispatcher
    lab = [{"input": [0.1], "target": 1.0}]
    rew = [{"source": "env_return", "value": 1.0}]
    if RegimeDispatcher(policy={"rl.regime": "rl"}).dispatch(
            labeled_rows=lab, reward_records=rew) != "rl":
        failures.append("rl.regime override dead")
    if validate_rl_policy({"rl.trainer": "sarsa"}) is None:
        failures.append("rl.trainer enum leak")
    assert not failures, failures


# ----------------------------------------------------- GK-03
def test_gk03_all_key_refusal_gate():
    """Every registered key refuses a type-garbage value —
    iterates the registries (auto-covers future keys)."""
    from reference_net.growthpolicy import preference as prf
    from rl_trainer.defaults import (GATE_DEFAULTS, RL_DEFAULTS,
                                     validate_rl_policy)

    class _Garbage:
        pass
    leaks = []

    def _check(fn, k):
        # a CRASH is also not a loud refusal (§4.5): the
        # validator must return a refusal dict, never raise
        try:
            r = fn({k: _Garbage()})
        except Exception as e:
            leaks.append(f"{k} (validator CRASHED: "
                         f"{type(e).__name__})")
            return
        if r is None:
            leaks.append(f"{k} (garbage accepted)")
    for k in prf.PREFERENCE_DEFAULTS:
        _check(prf.validate_preference_policy, k)
    for k in list(RL_DEFAULTS) + list(GATE_DEFAULTS):
        _check(validate_rl_policy, k)
    assert not leaks, f"non-loud keys: {leaks}"
    # None = merge-deletion sentinel (preference_reset kills
    # keys via {k: None}): every key must ACCEPT it — the E-1
    # type pre-pass regressed exactly here (TS-03 catch)
    blocked = [k for k in prf.PREFERENCE_DEFAULTS
               if prf.validate_preference_policy({k: None})
               is not None]
    assert not blocked, f"None sentinel refused: {blocked}"


# ----------------------------------------------------- TR-G2
def test_trg2_quota_window_semantics():
    """E-3 companion (doc 83 M7): batches:N refreshes the
    explore quota at each N-batch window; life NEVER refreshes;
    the window cursor survives a snapshot/restore roundtrip
    (FR-1.5 replayability)."""
    from reference_net.growthpolicy import preference as prf

    def _mk(qw):
        p = prf.GrowthPreference({"seed": 0,
                                  "preference.rule": "thompson",
                                  "preference.min_count": 1,
                                  "preference.explore_quota": 1,
                                  "preference.quota_window": qw})
        p._force_event_stats(5, 4.0, 0.0, 0.01)
        p._force_stats("grow|s1", w=5.0, m=0.5, v=0.0, n_raw=5)
        return p

    def _offer(p, batch):
        return p.explore_offer({"move": "grow", "slope": 0.0,
                                "batch": batch}, raw_eff=-0.2)

    life = _mk("life")
    assert _offer(life, 0) is True
    assert _offer(life, 9) is False      # life: spent forever
    win = _mk("batches:2")
    assert _offer(win, 0) is True
    assert _offer(win, 1) is False       # same window [0,1]
    assert _offer(win, 2) is True        # new window [2,3]
    # roundtrip: cursor + usage survive; batch 3 stays spent
    q = _mk("batches:2")
    assert _offer(q, 2) is True
    r = prf.GrowthPreference({"seed": 0,
                              "preference.rule": "thompson",
                              "preference.min_count": 1,
                              "preference.explore_quota": 1,
                              "preference.quota_window":
                              "batches:2"})
    out = r.restore(q.snapshot())
    assert out is None or "refusal" not in (out or {})
    r._force_event_stats(5, 4.0, 0.0, 0.01)
    assert _offer(r, 3) is False         # window [2,3] spent
    assert _offer(r, 4) is True          # next window refreshes


def test_trg2b_quota_window_bad_n_refused():
    """E-3 guard: batches:N with non-positive/non-integer N is
    refused at the validator (else the window arithmetic would
    divide by zero — crash is not a loud refusal, §4.5)."""
    from reference_net.growthpolicy.preference import \
        validate_preference_policy as v
    for bad in ("batches:0", "batches:abc", "batches:-3",
                "batches:", "batches:1.5"):
        assert v({"preference.quota_window": bad}) is not None, bad
    assert v({"preference.quota_window": "batches:2"}) is None
    assert v({"preference.quota_window": "life"}) is None



def test_trg3_grad_fns_extreme_logits_no_nan():
    """E-4 (G-8): entropy/KL gradient functions must stay
    finite when softmax underflows to exact 0 (extreme logits
    happen mid-training; the in-file 1e-300 precedent applies).
    MATH LIMIT referee: lim p->0 of -p(ln p + H) = 0, so the
    guarded value at p_k=0 must be exactly 0 for entropy; the
    KL row limit p_k->0 is likewise p_k[(ln p_k - ln q_k)-KL]
    -> 0.
    """
    from rl_trainer.math import entropy_grad_logits, \
        kl_grad_logits

    def _softmax(z):
        z = np.asarray(z, float)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    p = _softmax([[0.0, 1000.0, 3.0]])       # p[0,0] == 0 exactly
    assert p[0, 0] == 0.0
    ge = entropy_grad_logits(p)
    assert np.isfinite(ge).all(), ge
    assert ge[0, 0] == 0.0                    # the math limit
    q = _softmax([[1.0, 0.5, -2.0]])
    gk = kl_grad_logits(p, q)
    assert np.isfinite(gk).all(), gk
    gk2 = kl_grad_logits(q, p)                # q_k -> 0 side too
    assert np.isfinite(gk2).all(), gk2



# ----------------------------------------------------- TR-G5
def test_trg5_kl_reference_fixed_incumbent_injection():
    """E-5 (N-1, design 86 LAW-3(ii)): trainers accept an
    injected FIXED reference (the committed incumbent) for
    the KL anchor. Laws boxed here:
      (1) the injected reference stays FIXED across step()
          calls (its arrays are bitwise unchanged after k
          steps);
      (2) anchored-to-incumbent end-state KL(pi_k || pi_0)
          <= snapshot-anchored end-state KL on the SAME data
          and seed (the moving snapshot dilutes the anchor
          into a trust-region — the fixed anchor pulls back
          to pi_0 by construction);
      (3) NO injection => bit-identical to today's per-step
          snapshot behavior (default path untouched).
    """
    from rl_trainer.trainers import PPOTrainer

    def _mk(seed=1):
        return PPOTrainer(obs_dim=3, n_actions=2, seed=seed,
                          policy={"rl.n_epochs": 4,
                                  "rl.kl_ref_coef": 0.5,
                                  "rl.target_kl": None})

    def _roll(r):
        N = 64
        return {"obs": r.normal(size=(N, 3)),
                "actions": r.integers(0, 2, N),
                "rewards": r.normal(size=N),
                "dones": np.zeros(N, bool),
                "logp": r.normal(-0.7, 0.1, N),
                "values": r.normal(size=N),
                "last_value": 0.0}

    def _probs(t, net, X):
        z, _ = t._fwd(net, X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _kl(p, q):
        return float(np.mean(np.sum(
            p * (np.log(p + 1e-300) - np.log(q + 1e-300)),
            axis=1)))

    rolls = [_roll(np.random.default_rng(k)) for k in range(3)]
    X_ref = rolls[0]["obs"]

    # anchored run: inject the incumbent BEFORE any step
    ta = _mk()
    inc = [(W.copy(), b.copy()) for W, b in ta.pi]   # pi_0
    ta.set_kl_reference(inc)
    ref_before = [(W.copy(), b.copy())
                  for W, b in ta._kl_ref]
    for ro in rolls:
        ta.step(ro)
    for (W0, b0), (W1, b1) in zip(ref_before, ta._kl_ref):
        assert np.array_equal(W0, W1)                 # law (1)
        assert np.array_equal(b0, b1)

    # snapshot run: same seed/data, no injection
    ts = _mk()
    p0 = _probs(ts, ts.pi, X_ref)                     # == pi_0
    for ro in rolls:
        ts.step(ro)
    kl_anchored = _kl(_probs(ta, ta.pi, X_ref), p0)
    kl_snapshot = _kl(_probs(ts, ts.pi, X_ref), p0)
    assert kl_anchored <= kl_snapshot, (              # law (2)
        kl_anchored, kl_snapshot)

    # default path untouched: two uninjected runs bit-match
    tu1, tu2 = _mk(), _mk()
    for ro in rolls:
        tu1.step(ro)
        tu2.step(ro)
    for (W1, b1), (W2, b2) in zip(tu1.pi, tu2.pi):    # law (3)
        assert np.array_equal(W1, W2)
        assert np.array_equal(b1, b2)


# ----------------------------------------------------- TR-G5b
def test_trg5b_runner_kl_reference_injection():
    """E-5 runner side: the injected incumbent adapter stays
    FIXED across train_rounds (its organ outputs on a fixed
    probe are bitwise unchanged), and the uninjected default
    path is bit-identical to a twin run (today's per-round
    snapshot untouched)."""
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld

    def _mk():
        return OrganPPORunner(
            StationaryWorld(seed=8), seed=9, hidden=10,
            policy={"rl.kl_ref_coef": 0.5,
                    "rl.target_kl": None})

    probe = np.stack([StationaryWorld(seed=8).sample_state(i)
                      for i in range(6)])
    ra = _mk()
    ra.set_kl_reference(ra.policy_adapter)     # incumbent pi_0
    before = ra._kl_ref.outputs(probe).copy()
    ra.train_rounds(2, horizon=64)
    assert np.array_equal(before, ra._kl_ref.outputs(probe))
    r1, r2 = _mk(), _mk()
    r1.train_rounds(2, horizon=64)
    r2.train_rounds(2, horizon=64)
    assert np.array_equal(r1.policy_adapter.outputs(probe),
                          r2.policy_adapter.outputs(probe))


# ----------------------------------------------------- TR-G6
def test_trg6_eval_stage_alignment():
    """E-6 (G-7, FR-3.4 intent): after align_to(live_world),
    the provider's fresh episode worlds start in the SAME
    regime as the live world (stage 1 / post-arrival), so the
    gate scores adoption evidence for the CURRENT service
    regime; before alignment (or pre-boundary) they keep
    stage 0 — today's behavior stays the default."""
    from rl_trainer.eval_provider import EvalEpisodeProvider
    from rl_trainer.worlds import (SensorArrivalWorld,
                                   StagedExpansionWorld)

    live = StagedExpansionWorld(seed=4, boundary=8)
    live.reset(seed=0)
    for _ in range(10):                       # cross boundary
        live.step(0)
    assert live._stage() == 1
    prov = EvalEpisodeProvider(StagedExpansionWorld,
                               world_seed=4, boundary=8)
    w0 = prov.world_cls(seed=prov.world_seed,
                        **prov.world_kwargs)
    w0.reset(seed=9_000_000)
    assert w0._stage() == 0                   # unaligned: today
    prov.align_to(live)
    w1 = prov.world_cls(seed=prov.world_seed,
                        **prov.world_kwargs)
    w1.reset(seed=9_000_000)
    assert w1._stage() == 1                   # aligned: current
    # pre-boundary live world must NOT flip the provider
    early = StagedExpansionWorld(seed=4, boundary=8)
    early.reset(seed=0)
    prov2 = EvalEpisodeProvider(StagedExpansionWorld,
                                world_seed=4, boundary=8)
    prov2.align_to(early)
    w2 = prov2.world_cls(seed=prov2.world_seed,
                         **prov2.world_kwargs)
    w2.reset(seed=9_000_001)
    assert w2._stage() == 0
    # sensor world: post-arrival observations must be live
    lv = SensorArrivalWorld(seed=5, arrival_step=4)
    lv.reset(seed=0)
    for _ in range(6):
        lv.step(0)
    pr = EvalEpisodeProvider(SensorArrivalWorld,
                             world_seed=5, arrival_step=4)
    pr.align_to(lv)
    w3 = pr.world_cls(seed=pr.world_seed, **pr.world_kwargs)
    obs = w3.reset(seed=9_000_002)
    assert np.any(obs[3:] != 0.0)             # sensors live


# ----------------------------------------------------- TR-G4
def test_trg4_rl_audit_tail_persisted_and_replayable(tmp_path):
    """E-7 (G-4, plan 96 R3-4): the P-loop leaves a persistent
    audit tail.
      (1) runner: self.audit collects events (dispatcher
          phase_switch + whatever gate verdicts it is handed)
          and drain_audit() returns-and-clears;
      (2) facade: _rl_audit_write(model_id, events) appends
          JSON LINES to rl_audit.jsonl in the model's working
          dir (mirror of _preference_write's audit tail);
      (3) replay-readable: every line json.loads cleanly even
          when events carry numpy scalars/arrays (default=str
          precedent) — JSON-serializability is the live-proven
          G-4 risk.
    """
    import json
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld
    from core.facade import System
    from core.wiring import Config

    run = OrganPPORunner(StationaryWorld(seed=8), seed=9,
                         hidden=10)
    run.train_rounds(1, horizon=64)
    run.audit.append({"kind": "gate_verdict",
                      "adopted": np.bool_(True),
                      "score": np.float64(0.25),
                      "seeds": np.arange(3)})
    ev = run.drain_audit()
    assert ev and run.audit == []            # returns+clears
    assert any(e["kind"] == "gate_verdict" for e in ev)

    s = System(Config.from_env(backend="mlp",
                               models_root=tmp_path / "ws"))
    out = s.create_model("m", description="t")
    assert "refusal" not in out, out
    rows = [{"input": {"a": float(i) / 24.0,
                       "b": float(24 - i) / 24.0},
             "target": float(i) / 24.0} for i in range(24)]
    r = s.study("m", rows, steps=5)
    assert "refusal" not in r, r
    w = s._rl_audit_write("m", ev)
    assert "refusal" not in w, w
    f = s.lc._wdir("m") / "rl_audit.jsonl"
    assert f.exists()
    lines = [json.loads(x) for x in
             f.read_text().splitlines() if x]      # (3) replay
    assert any(l["kind"] == "gate_verdict" for l in lines)
    # unknown model refuses loudly (preference precedent)
    assert "refusal" in s._rl_audit_write("ghost", ev)


# ----------------------------------------------------- TR-G7
def test_trg7_runner_same_seed_bit_identity_replay():
    """E-8/T-3 (NFR-1 replayability): two OrganPPORunner
    lives with identical (world seed, runner seed, policy)
    produce BIT-IDENTICAL policy/value functions after the
    same train_rounds schedule (np.array_equal on outputs
    over a fixed probe batch — no tolerance), and identical
    rollout streams (episode_ids + seed_tag + rewards)."""
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StagedExpansionWorld

    def _life():
        run = OrganPPORunner(StagedExpansionWorld(seed=6),
                             seed=7, hidden=10,
                             policy={"rl.ent_coef": 0.01,
                                     "rl.kl_ref_coef": 0.1,
                                     "rl.target_kl": None})
        ro = run.collect(64)
        run.train_rounds(2, horizon=64)
        return run, ro

    r1, ro1 = _life()
    r2, ro2 = _life()
    probe = np.stack([StagedExpansionWorld(seed=6)
                      .sample_state(i) for i in range(8)])
    assert np.array_equal(r1.policy_adapter.outputs(probe),
                          r2.policy_adapter.outputs(probe))
    assert np.array_equal(r1.value_adapter.outputs(probe),
                          r2.value_adapter.outputs(probe))
    assert np.array_equal(ro1["rewards"], ro2["rewards"])
    assert np.array_equal(ro1["episode_ids"],
                          ro2["episode_ids"])
    assert ro1["seed_tag"] == ro2["seed_tag"]


# ----------------------------------------------------- TR-G8
def test_trg8_cli_rl_key_refusal(tmp_path):
    """E-8/G-3 (FR-6 both-doors law, CLI door): set_policy
    through cli.py with an invalid rl.* value returns the
    loud refusal JSON through the CLI channel (exit 0,
    refusal in stdout — the facade's loud validation passes
    through the thin mirror unmodified)."""
    import json as _json
    import os
    import subprocess
    import sys as _sys
    env = dict(os.environ,
               SOFTMODEL_MODELS_ROOT=str(tmp_path / "ws"))
    mk = subprocess.run(
        [_sys.executable, "-m", "cli.cli", "create_model",
         _json.dumps({"model_id": "m"})],
        cwd=str(_ROOT), env=env, capture_output=True,
        text=True, timeout=180)
    assert _json.loads(mk.stdout).get("refusal") is None
    p = subprocess.run(
        [_sys.executable, "-m", "cli.cli", "set_policy",
         _json.dumps({"model_id": "m", "growth_params":
                      {"rl.trainer": "sarsa"}})],
        cwd=str(_ROOT), env=env, capture_output=True,
        text=True, timeout=180)
    out = _json.loads(p.stdout)
    assert "refusal" in out, out
    assert "sarsa" in out["refusal"] or "rl.trainer" in \
        out["refusal"]


# ----------------------------------------------------- TR-G9
def test_trg9_mcp_rl_key_refusal(tmp_path):
    """E-8/G-3 (FR-6 both-doors law, MCP door): the same
    invalid rl.* value refuses loudly through the MCP
    tools/call dispatch (generic door — the refusal dict
    rides the JSON-RPC result)."""
    import json as _json
    from core.facade import System
    from core.wiring import Config
    from mcp.mcp_server import MCPServer
    srv = MCPServer(System(Config.from_env(
        backend="mlp", models_root=tmp_path / "ws")))
    r = srv.handle({"jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": {"name": "create_model",
                               "arguments":
                               {"model_id": "m"}}})
    assert r["result"]["isError"] is False
    r = srv.handle({"jsonrpc": "2.0", "id": 2,
                    "method": "tools/call",
                    "params": {"name": "set_policy",
                               "arguments":
                               {"model_id": "m",
                                "updates": {"growth_params":
                                {"rl.trainer": "sarsa"}}}}})
    out = _json.loads(
        r["result"]["content"][0]["text"])
    assert "refusal" in out, out


# ----------------------------------------------------- TR-R4
def test_trr4_review_batch_validator_strictness():
    """R4 batch (post-close review, owner order 2026-07-30):
      R-E2: every rl.*/gate.* key ACCEPTS the None deletion
            sentinel (merge-None law, 72B D-1 [RV F6]);
      R-R3: int-semantic rl keys REFUSE non-integer numbers
            (never silent truncation, SS4.5 analog — the
            preference namespace already refuses);
      O-1:  non-finite numbers (inf) REFUSE on every numeric
            rl key (NaN already refused by range logic).
    """
    from rl_trainer.defaults import (GATE_DEFAULTS, RL_DEFAULTS,
                                     validate_rl_policy as v)
    # R-E2 — None sentinel passes every key
    blocked = [k for k in list(RL_DEFAULTS) + list(GATE_DEFAULTS)
               if v({k: None}) is not None]
    assert not blocked, f"None sentinel refused: {blocked}"
    # R-R3 — int keys refuse floats loudly
    for k in ("rl.n_epochs", "rl.batch_size", "rl.horizon",
              "rl.eval_episode_budget", "rl.eval_window"):
        assert v({k: 2.7}) is not None, f"{k} accepted 2.7"
        assert v({k: 4}) is None, f"{k} refused int 4"
    # O-1 — inf refused on every numeric key
    import math as _m
    leaks = [k for k in RL_DEFAULTS
             if isinstance(RL_DEFAULTS[k], (int, float))
             and not isinstance(RL_DEFAULTS[k], bool)
             and v({k: float("inf")}) is None]
    assert not leaks, f"inf accepted: {leaks}"


# ----------------------------------------------------- TR-R4b
def test_trr4b_entropy_guard_and_grpo_skip_audit():
    """R4 batch:
      R-R1: math.entropy stays finite on softmax-underflow
            rows (same G-8 boundary; math limit: the p=0
            term contributes exactly 0);
      R-R4: the runner's grpo branch leaves an audit event
            when it SKIPS a <2-episode round (loudness law —
            the trainer-level path already reports).
    """
    from rl_trainer.math import entropy
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld

    z = np.array([[0.0, 1000.0, 3.0]])
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    assert p[0, 0] == 0.0
    assert np.isfinite(entropy(p))                    # R-R1

    run = OrganPPORunner(StationaryWorld(seed=8, ep_len=32),
                         seed=9, hidden=10,
                         policy={"rl.trainer": "grpo"})
    run.train_rounds(1, horizon=16)   # 16 < ep_len -> 0 dones
    ev = run.drain_audit()
    kinds = [e["kind"] for e in ev]
    assert "rl_round_skipped" in kinds, kinds         # R-R4


# ----------------------------------------------------- TR-R5
def test_trr5_runner_grpo_path_key_effect():
    """R5 D-1 (plan 98; R2-1, doc 89 NFR-5(iii) per-PATH key
    effectiveness): on the runner's grpo path the four loop-
    level keys must be LIVE — changing each must change the
    organ end-state; target_kl must early-stop; the round
    audit must report epochs_run > 0."""
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld

    def _final(policy):
        run = OrganPPORunner(StationaryWorld(seed=3), seed=6,
                             hidden=8, policy=policy)
        run.train_rounds(2, horizon=128)
        return (np.asarray(run.policy_adapter.organ.W1).copy(),
                run.drain_audit())

    base, ev = _final({"rl.trainer": "grpo"})
    assert ev[-1]["epochs_run"] > 0, ev[-1]     # audit truth
    for delta in ({"rl.n_epochs": 1},
                  {"rl.ent_coef": 0.05},
                  {"rl.kl_ref_coef": 0.5}):
        W, _ = _final({"rl.trainer": "grpo", **delta})
        assert not np.array_equal(base, W), \
            f"key inert on grpo path: {delta}"
    _, ev = _final({"rl.trainer": "grpo",
                    "rl.target_kl": 1e-12})
    assert ev[-1]["epochs_run"] < 10, ev[-1]    # early stop


# ----------------------------------------------------- TR-R5b
def test_trr5b_runner_door_none_sentinel_clean():
    """R5 D-2 (plan 98; R3-1, the R4 self-regression): the L1
    runner door has no merge layer, so a None-valued key must
    behave as the sentinel MEANS — key absent, default applies
    — never a downstream int(None)/float(None) crash (SS4.5:
    a crash is not a loud refusal). Equivalence law: the
    None-carrying run is bit-identical to the default run."""
    from rl_trainer.runner import OrganPPORunner
    from rl_trainer.worlds import StationaryWorld

    def _final(policy):
        run = OrganPPORunner(StationaryWorld(seed=3), seed=6,
                             hidden=8, policy=policy)
        run.train_rounds(1, horizon=64)
        return np.asarray(run.policy_adapter.organ.W1).copy()

    base = _final({})
    for k in ("rl.horizon", "rl.n_epochs", "rl.lr",
              "rl.trainer", "rl.target_kl"):
        W = _final({k: None})               # must not crash
        assert np.array_equal(base, W), \
            f"{k}=None not equivalent to default"
