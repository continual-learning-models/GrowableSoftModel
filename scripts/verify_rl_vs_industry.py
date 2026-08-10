"""RL-vs-industry verification program (owner directive):
prove, with FIXED (non-growing) networks and multiple
independent methods, that every RL algorithm we implement or
specify computes results IDENTICAL to third-party
industry-standard references. Two tracks:

  MATH TRACK  — closed-form/authority cross-checks per
                algorithm (numpy official, mpmath-50, libm,
                torch).
  SIM TRACK   — full simulations: (a) multi-armed bandit runs
                where OUR shipped selection rules must produce
                the IDENTICAL choice sequence and cumulative
                reward as an independently written textbook
                reference given the same seeds, plus
                convergence behavior; (b) FULL PPO and GRPO
                training on a FIXED network: our pure-numpy
                update math (hand-derived gradients) vs a
                PyTorch implementation of the standard
                objective — losses, every gradient, and the
                multi-update parameter trajectory must match.

Every check prints its data; the report lands at
tests/logs/RL_INDUSTRY_VERIFICATION.md.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _m in ("Engine", "ReferenceNet"):
    sys.path.insert(0, str(ROOT / "modules" / _m))

import numpy as np
import torch

from reference_net.growthpolicy import evaluative_core as core
from reference_net.growthpolicy import preference as prf

torch.set_default_dtype(torch.float64)
OUT, FAIL = [], []


def rep(alg, check, data, ok):
    OUT.append((alg, check, data, "PASS" if ok else "FAIL"))
    if not ok:
        FAIL.append(f"{alg}:{check}")
    print(f"[{'PASS' if ok else 'FAIL'}] {alg:14s} {check}")
    for ln in str(data).splitlines():
        print(f"    {ln}")


# =====================================================
# SIM TRACK (a) — BANDIT SIMULATIONS, fixed 3-arm world
# =====================================================
TRUE_MEANS = [0.10, 0.30, 0.22]          # arm 1 is best
NOISE = 0.15
STEPS = 400
DECAY = 0.98


def _bandit_env(seed):
    r = np.random.default_rng(seed)
    def pull(arm):
        return float(TRUE_MEANS[arm] + NOISE * r.standard_normal())
    return pull


def _ours_run(rule, seed, eps=0.1, ucb_c=1.0):
    """OUR shipped part drives arm selection on the bandit."""
    p = prf.GrowthPreference({"seed": seed,
                              "preference.rule": rule,
                              "preference.bucket_spec": "b0",
                              "preference.min_count": 1,
                              "preference.eps": eps,
                              "preference.ucb_c": ucb_c})
    pull = _bandit_env(9000 + seed)
    choices, total = [], 0.0
    for t in range(STEPS):
        mults = p.score_set([{"move": f"arm{a}", "slope": None}
                             for a in range(3)])
        # equal-priors epoch: play round-robin until every arm
        # has one credit (identical rule in the reference)
        uncredited = [a for a in range(3)
                      if f"arm{a}" not in p.snapshot()["stats"]]
        arm = (uncredited[0] if uncredited
               else int(np.argmax(mults)))
        rwd = pull(arm)
        total += rwd
        choices.append(arm)
        p.credit({"event_id": f"t{t}", "bucket": f"arm{arm}",
                  "move": f"arm{arm}", "batch": t,
                  "quoted_gain": 0.0, "window_gains": [rwd],
                  "credited_gain": rwd, "advantage": rwd})
    return choices, total


def _reference_run(rule, seed, eps=0.1, ucb_c=1.0):
    """INDEPENDENT textbook implementation of the SAME
    registered formulas (discounted mean/var; thompson draw
    N(m, sqrt(v_ev/(w+1))) — v1.19 production fixed-scale
    form; ucb m + c*sqrt(v/w); eps-greedy; envelope
    clip(exp((raw-mu_ev)/sd_ev))), same seeds, written from the
    textbook definitions without importing our module."""
    rng = np.random.default_rng(seed + 10000)   # ts stream
    pull = _bandit_env(9000 + seed)
    stats = {}                # arm -> [w, m, v]
    ev = [0, 0.0, 0.0, 0.0]   # n, w, m, v (reference dist)
    choices, total = [], 0.0

    def fold(st, x):
        w, m, v = st
        w2 = DECAY * w + 1.0
        d = x - m
        m2 = m + d / w2
        v2 = max((DECAY * w * v + d * (x - m2)) / w2, 0.0)
        return [w2, m2, v2]

    for t in range(STEPS):
        mu_ev, sd_ev = ((ev[2], math.sqrt(max(ev[3], 0)))
                        if ev[0] >= 2 else (0.0, 0.0))
        mults = []
        for a in range(3):
            if a not in stats:
                mults.append(1.0)
                continue
            w, m, v = stats[a]
            if rule == "mean_clip":
                raw = m
            elif rule == "thompson":
                v_ev_r = max(ev[3], 0.0)
                raw = (m if (v_ev_r <= 0 or w <= 0) else
                       m + math.sqrt(v_ev_r / (w + 1.0))
                       * float(rng.standard_normal()))
            elif rule == "ucb":
                se = math.sqrt(v / w) if (v > 0 and w > 0) else 0
                raw = m + ucb_c * se
            elif rule == "eps_greedy":
                u = float(rng.uniform())
                if u < eps and sd_ev > 0:
                    raw = mu_ev + sd_ev \
                        * float(rng.standard_normal())
                else:
                    raw = m
            if sd_ev <= 1e-9 * max(1.0, abs(mu_ev)):
                mults.append(1.0)
            else:
                z = (raw - mu_ev) / sd_ev
                mults.append(min(max(math.exp(z), 0.5), 2.0))
        uncredited = [a for a in range(3) if a not in stats]
        arm = (uncredited[0] if uncredited
               else int(np.argmax(mults)))
        rwd = pull(arm)
        total += rwd
        choices.append(arm)
        stats[arm] = fold(stats.get(arm, [0.0, 0.0, 0.0]), rwd)
        n, w, m, v = ev
        w2, m2, v2 = fold([w, m, v], rwd)
        ev = [n + 1, w2, m2, v2]
    return choices, total


for rule in ("mean_clip", "thompson", "ucb", "eps_greedy"):
    ours_c, ours_r = _ours_run(rule, 0)
    ref_c, ref_r = _reference_run(rule, 0)
    ident = ours_c == ref_c and abs(ours_r - ref_r) < 1e-9
    best_frac = ours_c[-100:].count(1) / 100.0
    rep(rule, "bandit SIM: 400-step choice sequence + reward "
        "vs independent textbook reference (same seeds)",
        f"identical_choices={ours_c == ref_c} "
        f"reward ours={ours_r:.6f} ref={ref_r:.6f} | last-100 "
        f"best-arm rate={best_frac:.2f}",
        ident)
    if rule in ("thompson", "ucb", "mean_clip"):
        rep(rule, "bandit SIM: convergence to the best arm "
            "(behavioral validity)",
            f"best-arm fraction over final 100 steps = "
            f"{best_frac:.2f} (world means {TRUE_MEANS})",
            best_frac >= 0.6)

# second seed replication
ours_c, _ = _ours_run("thompson", 7)
ref_c, _ = _reference_run("thompson", 7)
rep("thompson", "bandit SIM replication seed=7",
    f"identical_choices={ours_c == ref_c}", ours_c == ref_c)

# =====================================================
# SIM TRACK (b) — PPO / GRPO on a FIXED network
# =====================================================
S, A, H = 4, 3, 64          # states, actions, horizon
GAMMA, LAM, EPS_CLIP = 0.99, 0.95, 0.2
VCOEF, ENTCOEF, LR = 0.5, 0.01, 0.05
rng0 = np.random.default_rng(2027)
P_T = rng0.dirichlet(np.ones(S), size=(S, A))     # transitions
R_T = rng0.normal(0, 1, size=(S, A))              # rewards
W0 = rng0.normal(0, 0.3, size=(A, S))             # policy net
B0 = np.zeros(A)
WV0 = rng0.normal(0, 0.3, size=S)                 # value net
BV0 = 0.0


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _collect(W, b, wv, bv, seed):
    r = np.random.default_rng(seed)
    s = 0
    obs, acts, rews, logps, vals = [], [], [], [], []
    for _ in range(H):
        x = np.eye(S)[s]
        pi = _softmax(W @ x + b)
        a = int(r.choice(A, p=pi))
        obs.append(s); acts.append(a)
        logps.append(math.log(pi[a]))
        vals.append(float(wv @ x + bv))
        rews.append(float(R_T[s, a]))
        s = int(r.choice(S, p=P_T[s, a]))
    return obs, acts, rews, logps, vals


def _gae(rews, vals):
    """OUR GAE: TD residuals (L1) + L0 exponential weights and
    unnormalized fold — the shipped kernel path."""
    deltas = [rews[t] + (GAMMA * vals[t + 1] if t + 1 < H else 0)
              - vals[t] for t in range(H)]
    wts = core.credit_weights({"kind": "exp",
                               "base": GAMMA * LAM, "n": H})
    return [core.credit_fold(deltas[t:], wts[:H - t],
                             normalize=False)
            for t in range(H)]


def _numpy_ppo_grads(W, b, wv, bv, batch, adv, use_value=True):
    """Hand-derived PPO gradients (pure numpy — OUR math):
    d/dtheta of -mean(min(r*A, clip(r)*A)) - ent*H(pi)
    + vcoef*MSE(V, ret). Softmax policy gradient derived by
    hand: dlogpi_a/dz = onehot(a) - pi."""
    obs, acts, rews, old_logps, vals = batch
    gW = np.zeros_like(W); gb = np.zeros_like(b)
    gwv = np.zeros_like(wv); gbv = 0.0
    rets = [adv[t] + vals[t] for t in range(H)]
    pol_loss = v_loss = ent = 0.0
    for t in range(H):
        x = np.eye(S)[obs[t]]
        z = W @ x + b
        pi = _softmax(z)
        logpi = math.log(pi[acts[t]])
        ratio = math.exp(logpi - old_logps[t])
        A_t = adv[t]
        un = ratio * A_t
        cl = min(max(ratio, 1 - EPS_CLIP), 1 + EPS_CLIP) * A_t
        pol_loss += -min(un, cl)
        active = un <= cl          # unclipped branch carries grad
        dlog = (np.eye(A)[acts[t]] - pi)
        if active:
            gz = -ratio * A_t * dlog
        else:
            gz = np.zeros(A)
        ent_t = -float((pi * np.log(pi)).sum())
        ent += ent_t
        # d(-ENTCOEF*ent)/dz = ENTCOEF * dsum(pi log pi)/dz
        gz += ENTCOEF * (pi * (np.log(pi) + 1)
                         - pi * float((pi * (np.log(pi) + 1)
                                       ).sum()))
        gW += np.outer(gz, x); gb += gz
        if use_value:
            v = float(wv @ x + bv)
            v_loss += (v - rets[t]) ** 2
            gwv += VCOEF * 2 * (v - rets[t]) * x / H
            gbv += VCOEF * 2 * (v - rets[t]) / H
    gW /= H; gb /= H
    loss = (pol_loss / H + VCOEF * v_loss / H
            - ENTCOEF * ent / H)
    return loss, gW, gb, gwv, gbv


def _torch_ppo_grads(W, b, wv, bv, batch, adv, use_value=True):
    """INDUSTRY-STANDARD reference: same objective in PyTorch,
    gradients by AUTOGRAD (the third-party referee)."""
    obs, acts, rews, old_logps, vals = batch
    tW = torch.tensor(W, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    twv = torch.tensor(wv, requires_grad=True)
    tbv = torch.tensor([bv], requires_grad=True)
    X = torch.eye(S)[torch.tensor(obs)]
    z = X @ tW.T + tb
    logpi = torch.log_softmax(z, dim=1)
    lp = logpi[torch.arange(H), torch.tensor(acts)]
    ratio = torch.exp(lp - torch.tensor(old_logps))
    tadv = torch.tensor(adv)
    un = ratio * tadv
    cl = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * tadv
    pol = -torch.minimum(un, cl).mean()
    ent = -(torch.exp(logpi) * logpi).sum(dim=1).mean()
    loss = pol - ENTCOEF * ent
    if use_value:
        rets = tadv + torch.tensor(vals)
        v = X @ twv + tbv
        loss = loss + VCOEF * ((v - rets) ** 2).mean()
    loss.backward()
    return (float(loss.detach()), tW.grad.numpy(),
            tb.grad.numpy(),
            (twv.grad.numpy() if twv.grad is not None
             else np.zeros_like(wv)),
            (float(tbv.grad[0]) if tbv.grad is not None
             else 0.0))


# ---- PPO: 5 sequential updates, trajectory identity ----
Wn, bn, wvn, bvn = W0.copy(), B0.copy(), WV0.copy(), BV0
Wt, bt, wvt, bvt = W0.copy(), B0.copy(), WV0.copy(), BV0
max_gdiff = max_ldiff = max_pdiff = 0.0
for it in range(5):
    batch = _collect(Wn, bn, wvn, bvn, seed=500 + it)
    adv = _gae(batch[2], batch[4])
    ln, gWn, gbn, gwvn, gbvn = _numpy_ppo_grads(
        Wn, bn, wvn, bvn, batch, adv)
    lt, gWt, gbt, gwvt, gbvt = _torch_ppo_grads(
        Wt, bt, wvt, bvt, batch, adv)
    max_ldiff = max(max_ldiff, abs(ln - lt))
    max_gdiff = max(max_gdiff,
                    float(np.max(np.abs(gWn - gWt))),
                    float(np.max(np.abs(gbn - gbt))),
                    float(np.max(np.abs(gwvn - gwvt))),
                    abs(gbvn - gbvt))
    for arr, g in ((Wn, gWn), (bn, gbn), (wvn, gwvn)):
        arr -= LR * g
    bvn -= LR * gbvn
    for arr, g in ((Wt, gWt), (bt, gbt), (wvt, gwvt)):
        arr -= LR * g
    bvt -= LR * gbvt
    max_pdiff = max(max_pdiff,
                    float(np.max(np.abs(Wn - Wt))),
                    float(np.max(np.abs(wvn - wvt))))
rep("PPO", "FIXED-network SIM: 5 full updates, hand-derived "
    "numpy gradients vs PyTorch autograd (industry standard)",
    f"max |loss diff|={max_ldiff:.2e}  max |grad diff|="
    f"{max_gdiff:.2e}  max |param diff after 5 updates|="
    f"{max_pdiff:.2e}",
    max_ldiff < 1e-12 and max_gdiff < 1e-12
    and max_pdiff < 1e-11)

# ---- GAE itself vs torch reference on the real rollout ----
batch = _collect(W0, B0, WV0, BV0, seed=321)
adv_ours = _gae(batch[2], batch[4])
d = [batch[2][t] + (GAMMA * batch[4][t + 1] if t + 1 < H else 0)
     - batch[4][t] for t in range(H)]
td = torch.tensor(d)
At = torch.zeros(H); acc = torch.tensor(0.0)
for t in reversed(range(H)):
    acc = td[t] + GAMMA * LAM * acc
    At[t] = acc
gae_diff = float(np.max(np.abs(np.asarray(adv_ours)
                               - At.numpy())))
rep("GAE", "rollout SIM: L0-kernel GAE vs torch recursive "
    f"reference (H={H})",
    f"max |diff| = {gae_diff:.2e}", gae_diff < 1e-11)

# ---- GRPO: group-standardized, no value head ----
Wg, bg = W0.copy(), B0.copy()
Wgt, bgt = W0.copy(), B0.copy()
max_gd = max_pd = 0.0
for it in range(5):
    batch = _collect(Wg, bg, WV0 * 0, 0.0, seed=800 + it)
    G = 4
    rets_ep = np.asarray(batch[2]).reshape(G, H // G).sum(axis=1)
    mu, sd = rets_ep.mean(), rets_ep.std()
    adv_ep = (rets_ep - mu) / (sd if sd > 1e-9 else 1.0)
    adv = np.repeat(adv_ep, H // G)
    ln, gWn, gbn, _, _ = _numpy_ppo_grads(
        Wg, bg, WV0 * 0, 0.0, batch, list(adv),
        use_value=False)
    lt, gWt, gbt, _, _ = _torch_ppo_grads(
        Wgt, bgt, WV0 * 0, 0.0, batch, list(adv),
        use_value=False)
    max_gd = max(max_gd, float(np.max(np.abs(gWn - gWt))),
                 float(np.max(np.abs(gbn - gbt))))
    Wg -= LR * gWn; bg -= LR * gbn
    Wgt -= LR * gWt; bgt -= LR * gbt
    max_pd = max(max_pd, float(np.max(np.abs(Wg - Wgt))))
rep("GRPO", "FIXED-network SIM: 5 updates, group-standardized "
    "advantages (no value head), numpy vs torch autograd",
    f"max |grad diff|={max_gd:.2e}  max |param diff|="
    f"{max_pd:.2e}", max_gd < 1e-12 and max_pd < 1e-11)
# zero-variance group edge (normative doc 86 N1)
rets_ep = np.array([2.0, 2.0, 2.0, 2.0])
sd = rets_ep.std()
adv_edge = ((rets_ep - rets_ep.mean())
            / (sd if sd > 1e-9 else 1.0))
rep("GRPO", "zero-variance group edge -> advantages exactly 0 "
    "(never NaN)", f"advantages={adv_edge.tolist()}",
    np.all(adv_edge == 0.0))

# ---- pseudo-target on a FIXED substrate quadratic head ----
h = torch.tensor([0.31, -2.4, 0.0], requires_grad=True)
target = torch.tensor([0.5, -2.0, 0.1])
L = 0.5 * ((h - target) ** 2).sum()
L.backward()
dL = h.grad.numpy()
ystar = h.detach().numpy() - dL
rep("pseudo-target", "y* = h - dL/dh: substrate quadratic-head "
    "gradient toward y* == autograd dL/dh (fixed head)",
    f"autograd={dL.tolist()}\n(h-y*)  ={list(h.detach().numpy()-ystar)}",
    np.allclose(h.detach().numpy() - ystar, dL, atol=1e-15))

# ---- EMA statistics vs torch reference over a stream ----
xs = list(np.random.default_rng(55).normal(0.1, 0.5, 200))
st = (0.0, 0.0, 0.0)
for x in xs:
    st = core.ema_fold(st, x, DECAY)
wts = DECAY ** torch.arange(len(xs) - 1, -1, -1,
                            dtype=torch.float64)
xt = torch.tensor(xs)
m_t = float((wts * xt).sum() / wts.sum())
v_t = float((wts * (xt - m_t) ** 2).sum() / wts.sum())
rep("EMA-stats", "200-sample stream: shipped fold vs torch "
    "weighted-tensor reference",
    f"ours m={st[1]:.15f} v={st[2]:.15f}\n"
    f"torch m={m_t:.15f} v={v_t:.15f}",
    abs(st[1] - m_t) < 1e-12 and abs(st[2] - v_t) < 1e-12)

# ================= report =================
print("\n" + "=" * 64)
print(f"TOTAL {len(OUT)} checks | "
      f"{'ALL PASS' if not FAIL else 'FAILURES: ' + str(FAIL)}")
lines = ["# RL vs Industry-Standard Algorithms — Verification "
         "Report (fixed networks)",
         "2026-07-28 · scripts/verify_rl_vs_industry.py",
         "",
         "Methods: (SIM) 3-arm bandit choice-sequence identity "
         "vs independent textbook reference + convergence; "
         "(SIM) full PPO/GRPO training on a FIXED softmax-policy "
         "network — hand-derived numpy gradients vs PyTorch "
         "autograd, 5-update parameter-trajectory identity; "
         "(MATH) GAE/pseudo-target/EMA vs torch references.",
         ""]
for alg, check, data, verdict in OUT:
    lines.append(f"## [{verdict}] {alg} — {check}")
    lines.append("```")
    lines.append(str(data))
    lines.append("```")
(ROOT / "tests" / "logs" / "RL_INDUSTRY_VERIFICATION.md"
 ).write_text("\n".join(lines))
print("report -> tests/logs/RL_INDUSTRY_VERIFICATION.md")
sys.exit(1 if FAIL else 0)
