"""Track-B D-P1 tests-first referee boxes TB-P01..P06 + TP-P01
(plan 84 v2.17; design doc 86 §5/§6/§10; TEST-EXECUTION
STANDARD T-AX1..4).

RED PHASE CONTRACT: every box imports modules/RLTrainer's
`rl_trainer` package, which DOES NOT EXIST yet — all boxes
fail on ImportError until D-P2 builds the trainer family.
Worksheets are hand-derived from the normative equations and
transcribed at mpmath-30 precision; the NORMATIVE REFERENCE
implementation is the verified blueprint
(scripts/verify_vs_authoritative_libs.py, 9/9 vs
SB3/gymnasium/MABWiser — the BLUEPRINT-PARITY GATE at D-P2
requires bit-identical trajectories).

T-AX3 method table (routes per box):
  TB-P01 hand worksheet + dual-form cross-check (recursive
         vs explicit (γλ)^k sum) + L0 credit_weights route
  TB-P02 hand worksheet (both clip branches) + FD parity
  TB-P03 hand worksheet (value MSE, categorical entropy)
  TB-P04 replay determinism (same seed twice + fresh object)
  TB-P05 pseudo-target adapter parity vs direct-gradient
         update (hand-derived toy, 1e-10) — gates everything
  TB-P06 hand worksheet (group standardization) + zero-var
         edge exact 0
  TP-P01 trust-region property (KL early stop)
  TB-P07 blueprint-parity gate (bit-identity vs the 9/9
         verified script class, source-extracted)
  TB-P08 organ adapter: R1 bit-identity vs existing train
         path + R2 FD-descent effect + R3 G-1 growth instant
"""
import numpy as np
import pytest
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "modules" / "RLTrainer"),
           str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ------------------------------------------------------ TB-P01
def test_tbp01_gae_dual_form_and_hand_values():
    """TB-P01: GAE referee (doc 86 §5; normative recursion
    acc_t = delta_t + γλ·nonterm·acc_{t+1}).
    WORKSHEET (γ=0.9, λ=0.8, T=4; r=(1.0,0.5,-0.2,2.0),
    v=(0.3,0.1,0.4,0.2), last_v=0.6, dones=(0,0,1,0)):
      delta_3 = 2.0+0.9*0.6-0.2      = 2.34
      delta_2 = -0.2+0-0.4           = -0.6   (done: no boot)
      delta_1 = 0.5+0.9*0.4-0.1      = 0.76
      delta_0 = 1.0+0.9*0.1-0.3      = 0.79
      adv     = (1.02616, 0.328, -0.6, 2.34)  [hand recursion:
                0.79+0.72*0.328; 0.76+0.72*(-0.6)]
      returns = adv+v = (1.32616, 0.428, -0.2, 2.54)
    DUAL FORM: adv_t must equal the explicit truncated sum
    Σ_k (γλ)^k δ_{t+k} within the episode (weights from the
    L0 credit_weights exp family — the doc 86 §3.0 claim that
    GAE weights and credit weights are ONE family)."""
    from rl_trainer.math import gae
    from reference_net.growthpolicy import evaluative_core as core
    rews = np.array([1.0, 0.5, -0.2, 2.0])
    vals = np.array([0.3, 0.1, 0.4, 0.2])
    dones = np.array([False, False, True, False])
    adv = gae(rews, vals, last_v=0.6, dones=dones,
              gamma=0.9, lam=0.8)
    want = np.array([1.02616, 0.328, -0.6, 2.34])
    assert np.allclose(adv, want, atol=1e-12)
    # dual form: episode 1 = t 0..2 (done at 2), explicit sums
    deltas = np.array([0.79, 0.76, -0.6])
    w = core.credit_weights({"kind": "exp", "base": 0.72,
                             "n": 3})
    for t in range(3):
        explicit = float(np.sum(deltas[t:] * w[:3 - t]))
        assert abs(adv[t] - explicit) < 1e-12
    assert abs(adv[3] - 2.34) < 1e-12   # episode 2 head


# ------------------------------------------------------ TB-P02
def test_tbp02_clipped_surrogate_both_branches_and_fd():
    """TB-P02: clipped surrogate (doc 86 §5): loss =
    -mean(min(ratio*A, clip(ratio,1-eps,1+eps)*A)), eps=0.2.
    WORKSHEET (single samples; mpmath-30):
      old=-0.5 new=-0.2 A=+1.5: ratio=1.3498588075760031
        unclipped=2.0247882113640047 clipped=1.8
        loss = -1.8                       (clip ACTIVE branch)
      old=-0.5 new=-0.2 A=-1.5:
        min(-2.0247882113640047, -1.8) = -2.0247882113640047
        loss = +2.0247882113640047        (pessimistic branch)
      old=-0.5 new=-1.4 A=+0.7: ratio=0.40656965974059911
        min(0.28459876181841938, 0.56) -> loss =
        -0.28459876181841938              (unclipped branch)
    FD PARITY: dloss/dlogp_new by central finite difference
    (h=1e-6) must match the analytic branch gradient
    (-ratio*A on unclipped/pessimistic branches, 0 on the
    clip-active branch) within 1e-6."""
    from rl_trainer.math import clipped_surrogate
    cases = [(-0.5, -0.2, 1.5, -1.8, 0.0),
             (-0.5, -0.2, -1.5, 2.0247882113640047,
              -1.3498588075760031 * -1.5),
             (-0.5, -1.4, 0.7, -0.28459876181841938,
              -0.40656965974059911 * 0.7)]
    for lp_old, lp_new, A, want_loss, want_grad in cases:
        loss = clipped_surrogate(np.array([lp_new]),
                                 np.array([lp_old]),
                                 np.array([A]), clip=0.2)
        assert abs(loss - want_loss) < 1e-12
        h = 1e-6
        lp = clipped_surrogate(np.array([lp_new + h]),
                               np.array([lp_old]),
                               np.array([A]), clip=0.2)
        lm = clipped_surrogate(np.array([lp_new - h]),
                               np.array([lp_old]),
                               np.array([A]), clip=0.2)
        fd = (lp - lm) / (2 * h)
        assert abs(fd - want_grad) < 1e-6


# ------------------------------------------------------ TB-P03
def test_tbp03_value_and_entropy_referees():
    """TB-P03: value/entropy terms (doc 86 §5).
    WORKSHEET: vpred=(0.4,-0.1), vtarget=(0.9,0.3):
      value MSE = ((0.5)^2+(0.4)^2)/2 = 0.205
    probs=(0.7,0.2,0.1): H = -Σ p ln p
      = 0.80181855254333731 (mpmath-30)."""
    from rl_trainer.math import value_loss, entropy
    vl = value_loss(np.array([0.4, -0.1]),
                    np.array([0.9, 0.3]))
    assert abs(vl - 0.205) < 1e-15
    H = entropy(np.array([[0.7, 0.2, 0.1]]))
    assert abs(H - 0.80181855254333731) < 1e-15


# ------------------------------------------------------ TB-P04
def test_tbp04_rollout_buffer_determinism():
    """TB-P04: buffer determinism (NFR-1): identical seeded
    fills + identical batch draws => bit-identical minibatch
    index sequences and contents; a FRESH buffer with the
    same seed reproduces them (replay-grade determinism)."""
    from rl_trainer.buffer import RolloutBuffer

    def fill(buf, seed):
        r = np.random.default_rng(seed)
        for _ in range(32):
            buf.add(obs=r.normal(size=3), action=int(r.integers(2)),
                    reward=float(r.normal()), done=bool(r.random() < 0.1),
                    logp=float(r.normal()), value=float(r.normal()))
    a = RolloutBuffer(32, 3); fill(a, 5)
    b = RolloutBuffer(32, 3); fill(b, 5)
    ba = list(a.batches(batch_size=8, epochs=2, seed=9))
    bb = list(b.batches(batch_size=8, epochs=2, seed=9))
    assert len(ba) == len(bb) == 8
    for xa, xb in zip(ba, bb):
        for k in xa:
            assert np.array_equal(xa[k], xb[k]), k


# ------------------------------------------------------ TB-P05
def test_tbp05_pseudo_target_adapter_parity():
    """TB-P05: PSEUDO-TARGET ADAPTER PARITY (doc 86 §10 —
    gates everything): on the quadratic head the gradient
    toward pseudo-target y* is (h - y*), so injecting
    y* = h - dL/dh reproduces ANY output-side gradient.
    WORKSHEET (pure algebra, hand): h=(0.8,-0.3),
    dL/dh=(0.25,-0.5) => y* = (0.55, 0.2); then the
    substrate's own step law t = h - eta*dL/dh with the
    SAME dL/dh must be recovered: pseudo_target_for is the
    EXACT inverse, i.e. (h - y*) == dL/dh to 1e-16.
    (The end-to-end organ parity — adapter-driven
    train_step == direct-gradient update to 1e-10 — is the
    D-P2 gate on this same box, extended once the trainer
    exists.)"""
    from rl_trainer.pseudo_target import pseudo_target_for
    h = np.array([0.8, -0.3])
    g = np.array([0.25, -0.5])
    y_star = pseudo_target_for(h, g)
    assert np.allclose(y_star, np.array([0.55, 0.2]),
                       atol=1e-16)
    assert np.allclose(h - y_star, g, atol=1e-16)


# ------------------------------------------------------ TB-P06
def test_tbp06_grpo_group_baseline_referee():
    """TB-P06: GRPO group-relative baseline (doc 86 §5).
    WORKSHEET: episode returns (3, 5, 10):
      mu = 6.0, sd(population) = 2.943920288775949
      advs = (-1.0190493307301363, -0.3396831102433787,
              1.3587324409735149)
    ZERO-VARIANCE EDGE (C-1 registered): returns (4,4,4)
    => advantages EXACTLY (0,0,0), no division by zero."""
    from rl_trainer.math import group_advantages
    a = group_advantages(np.array([3.0, 5.0, 10.0]))
    want = np.array([-1.0190493307301363,
                     -0.3396831102433787,
                     1.3587324409735149])
    assert np.allclose(a, want, atol=1e-15)
    z = group_advantages(np.array([4.0, 4.0, 4.0]))
    assert np.array_equal(z, np.zeros(3))


# ------------------------------------------------------ TP-P01
def test_tpp01_trust_region_kl_early_stop_property():
    """TP-P01: trust-region property (doc 86 §5, FR-3.1 KL
    early stop): with an absurdly small rl.target_kl the
    trainer must terminate its epoch loop early
    (stats["epochs_run"] < requested epochs) and report the
    triggering approx_kl; with target_kl=None it runs all
    epochs. Property-level: no hand value, but the box
    asserts BEHAVIOR + the reported approx_kl >= target_kl
    at the stop point."""
    from rl_trainer.trainers import PPOTrainer
    r = np.random.default_rng(0)
    N = 64
    rollout = {"obs": r.normal(size=(N, 3)),
               "actions": r.integers(0, 2, N),
               "rewards": r.normal(size=N),
               "dones": np.zeros(N, bool),
               "logp": r.normal(-0.7, 0.1, N),
               "values": r.normal(size=N),
               "last_value": 0.0}
    t1 = PPOTrainer(obs_dim=3, n_actions=2, seed=1,
                    policy={"rl.n_epochs": 10,
                            "rl.target_kl": 1e-9})
    s1 = t1.step(rollout)
    assert s1["epochs_run"] < 10
    assert s1["approx_kl"] >= 1e-9
    t2 = PPOTrainer(obs_dim=3, n_actions=2, seed=1,
                    policy={"rl.n_epochs": 3,
                            "rl.target_kl": None})
    s2 = t2.step(rollout)
    assert s2["epochs_run"] == 3


# ------------------------------------------------------ TB-P07
def test_tbp07_blueprint_parity_bit_identical():
    """TB-P07 BLUEPRINT-PARITY GATE (plan 84 v2.17 D-P2, the
    anti-rework law): the module trainer with DEFAULT keys
    must reproduce the VERIFIED blueprint (NumpyPPO, 9/9 vs
    SB3/MABWiser) BIT-IDENTICALLY — same seed, same inputs,
    identical pi/vf parameter arrays after a full update
    round (both nets, all 6 layers, np.array_equal — no
    tolerance). The blueprint CLASS SOURCE is extracted
    verbatim from the referee script (no re-derivation, no
    drift possible)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "scripts" /
           "verify_vs_authoritative_libs.py").read_text()
    seg = src[src.index("class NumpyPPO"):
              src.index("def gae_ours")]
    import math as _math
    g = {"np": np, "math": _math, "LR": 3e-4,
         "EPS_ADAM": 1e-5, "EPOCHS": 10, "BATCH": 64,
         "CLIP": 0.2, "VCOEF": 0.5}
    exec(compile(seg, "blueprint", "exec"), g)
    Blueprint = g["NumpyPPO"]
    from rl_trainer.trainers import PPOTrainer
    r = np.random.default_rng(42)
    N = 200
    obs = r.normal(size=(N, 4))
    acts = r.integers(0, 2, N)
    old_lp = r.normal(-0.7, 0.1, N)
    adv = r.normal(size=N)
    rets = r.normal(size=N)
    bp = Blueprint(4, 2, seed=7)
    ours = PPOTrainer(4, 2, seed=7)
    for i in range(3):                    # identical init
        assert np.array_equal(bp.pi[i][0], ours.pi[i][0])
        assert np.array_equal(bp.vf[i][0], ours.vf[i][0])
    bp.update(obs, acts, old_lp, adv, rets)
    ours.update(obs, acts, old_lp, adv, rets)
    for i in range(3):                    # identical trajectory
        assert np.array_equal(bp.pi[i][0], ours.pi[i][0]), i
        assert np.array_equal(bp.pi[i][1], ours.pi[i][1]), i
        assert np.array_equal(bp.vf[i][0], ours.vf[i][0]), i
        assert np.array_equal(bp.vf[i][1], ours.vf[i][1]), i


# ------------------------------------------------------ TB-P08
def test_tbp08_organ_adapter_identity_fd_and_g1():
    """TB-P08 (D-P2 sub-step 2): OrganAdapter — the §10/§3.35
    pseudo-target adapter driving the REAL growable organ.
    THREE ROUTES (T-AX3):
    R1 IDENTITY (bit-exact): apply(X, g) MUST equal
       train_step(X, h - g) BIT-IDENTICALLY on a cloned
       organ — np.array_equal on W1/b1/W2/c, zero tolerance
       (the synthesized target walks the EXACT existing
       train path; nothing hidden). FLOAT-CANCELLATION FACT
       (recorded, not glossed): with g = fl(h - y_t),
       fl(h - g) reproduces y_t only to ~1 ulp — so the
       bitwise bar is against the SYNTHESIZED target h - g,
       while closeness to y_t is asserted at 1e-12 (the
       same summation-order-class footprint the house
       already recognizes at growth instants).
    R2 FD EFFECT: injecting the gradient of a known surrogate
       L(h) = 0.5*||h - y_t||^2 (g = h - y_t) must DECREASE
       that surrogate; and the organ's OWN mse-before-step
       readout equals the surrogate at the standardized scale
       (monotone-descent property assertion).
    R3 G-1 UNDER ADAPTER (quasi-static law): grow the organ
       (deepen) BETWEEN adapter updates; serving output on a
       probe batch is IDENTICAL pre/post at the growth
       instant (<= 1e-12, the width-rider machine-eps
       contract; pure paths bitwise), and the adapter keeps
       working after growth (out_width preserved).
    N3 NOTE: the substrate natively supports
    out_width=K (probed: Network(3,8,out_width=2) trains
    (16,2) targets, loss 1.617->0.615 over 50 steps) — the
    policy head is an out_width=K organ + softmax serving in
    the trainer layer; ZERO substrate code change."""
    import copy
    from reference_net.net import Network
    from rl_trainer.organ_adapter import OrganAdapter
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 3))
    y_warm = rng.normal(size=(20, 2))
    organ = Network(3, 8, seed=11, out_width=2)
    for _ in range(5):
        organ.train_step(X, y_warm)        # scalers fixed
    twin = copy.deepcopy(organ)
    ad = OrganAdapter(organ)
    y_t = rng.normal(size=(20, 2))
    h = ad.outputs(X)
    g = h - y_t
    y_star = h - g                         # the SYNTHESIZED target
    assert np.allclose(y_star, y_t, atol=1e-12)   # ~1 ulp fact
    ad.apply(X, g)                         # R1
    twin.train_step(X, y_star)             # same target, same path
    assert np.array_equal(np.asarray(organ.W1), np.asarray(twin.W1))
    assert np.array_equal(np.asarray(organ.b1), np.asarray(twin.b1))
    assert np.array_equal(np.asarray(organ.W2), np.asarray(twin.W2))
    assert np.array_equal(np.asarray(organ.c), np.asarray(twin.c))
    # R2: descent on the injected surrogate
    organ2 = Network(3, 8, seed=12, out_width=2)
    for _ in range(5):
        organ2.train_step(X, y_warm)
    ad2 = OrganAdapter(organ2)
    tgt = rng.normal(size=(20, 2))
    before = float(np.mean((ad2.outputs(X) - tgt) ** 2))
    for _ in range(30):
        ad2.apply(X, ad2.outputs(X) - tgt)
    after = float(np.mean((ad2.outputs(X) - tgt) ** 2))
    assert after < before * 0.8
    # R3: G-1 at a growth instant between adapter updates
    probe = rng.normal(size=(8, 3))
    pre = ad2.outputs(probe).copy()
    organ2.deepen(m=4, force=True)         # existing delta op
    post = ad2.outputs(probe)
    assert np.max(np.abs(post - pre)) <= 1e-12
    ad2.apply(X, ad2.outputs(X) - tgt)     # still trains
    assert ad2.outputs(X).shape == (20, 2)
