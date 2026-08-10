"""TrainerPlug family (doc 86 N1/§3.3): ppo + grpo, PORTED
from the verified blueprint (scripts/
verify_vs_authoritative_libs.py NumpyPPO — 9/9 vs SB3/
gymnasium/MABWiser). BLUEPRINT-PARITY GATE (plan 84 v2.17
D-P2, TB-P07): with default keys (rl.target_kl=None) the
update trajectory is BIT-IDENTICAL to the blueprint — port,
don't re-derive. Canonical details carried verbatim:
orthogonal init (gain sqrt(2) hidden / 0.01 policy head /
1.0 value head), per-minibatch advantage normalization,
global grad-norm clip 0.5, Adam eps 1e-5.

KL EARLY STOP (FR-3.1; SB3 semantics): when rl.target_kl is
set, the epoch loop stops as soon as the minibatch
approx_kl = mean((ratio-1) - log(ratio)) exceeds
1.5*target_kl, WITHOUT applying that minibatch; with the
default None the path is untouched (parity preserved).
"""
import math

import numpy as np

from .defaults import RL_DEFAULTS
from .math import (clipped_surrogate, entropy,
                   entropy_grad_logits, gae,
                   group_advantages, kl_grad_logits,
                   value_loss)


def _p(policy, key):
    policy = policy or {}
    return policy.get(key, RL_DEFAULTS[key])


class PPOTrainer:
    """TrainerPlug: step(rollout) -> update stats."""

    def __init__(self, obs_dim, n_actions, seed, policy=None):
        self.policy_keys = dict(policy or {})
        for k in self.policy_keys:
            if k not in RL_DEFAULTS:
                raise ValueError(f"unknown rl key {k!r}")
        r = np.random.default_rng(seed)

        def ortho(i, o, gain):
            a = r.normal(0, 1, (o, i))
            if o >= i:
                q, _ = np.linalg.qr(a)
            else:
                q, _ = np.linalg.qr(a.T)
                q = q.T
            return gain * q, np.zeros(o)
        g = math.sqrt(2.0)
        self.pi = [ortho(obs_dim, 64, g), ortho(64, 64, g),
                   ortho(64, n_actions, 0.01)]
        self.vf = [ortho(obs_dim, 64, g), ortho(64, 64, g),
                   ortho(64, 1, 1.0)]
        self.adam = {}
        self.t_step = 0
        self.n_act = n_actions
        self.rng = np.random.default_rng(seed + 1)
        self._kl_ref = None       # 96 E-5 (LAW-3(ii)) anchor

    def set_kl_reference(self, pi_weights):
        """96 E-5 (N-1, design 86 LAW-3(ii)): install the
        COMMITTED incumbent as the FIXED KL-anchor reference.
        Absent injection, update() keeps today's per-step
        snapshot (the documented trust-region flavor)."""
        self._kl_ref = [(W.copy(), b.copy())
                        for W, b in pi_weights]

    # ---------- forward ----------
    def _fwd(self, net, X):
        h = X
        acts = [X]
        for i, (W, b) in enumerate(net):
            z = h @ W.T + b
            h = np.tanh(z) if i < 2 else z
            acts.append(h)
        return h, acts

    def policy(self, X):
        logits, _ = self._fwd(self.pi, X)
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def value(self, X):
        v, _ = self._fwd(self.vf, X)
        return v[:, 0]

    def act(self, x):
        p = self.policy(x[None])[0]
        a = int(self.rng.choice(self.n_act, p=p))
        return a, math.log(p[a])

    # ---------- adam + backprop (blueprint verbatim) ----------
    def _adam(self, key, p, g):
        lr = float(_p(self.policy_keys, "rl.lr"))
        eps = float(_p(self.policy_keys, "rl.adam_eps"))
        m, v = self.adam.get(key, (np.zeros_like(p),
                                   np.zeros_like(p)))
        t = self.t_step
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t)
        vh = v / (1 - 0.999 ** t)
        self.adam[key] = (m, v)
        return p - lr * mh / (np.sqrt(vh) + eps)

    def _backprop(self, net, acts, gout, key):
        gmax = float(_p(self.policy_keys, "rl.max_grad_norm"))
        g = gout
        grads = {}
        for i in reversed(range(3)):
            W, b = net[i]
            h_in = acts[i]
            grads[i] = (g.T @ h_in, g.sum(axis=0))
            g_in = g @ W
            if i > 0:
                g_in = g_in * (1 - acts[i] ** 2)
            g = g_in
        total = math.sqrt(sum(float((gw * gw).sum()
                                    + (gb * gb).sum())
                              for gw, gb in grads.values()))
        scale = min(1.0, gmax / (total + 1e-12))   # SB3
        for i in range(3):                         # max_grad_norm
            W, b = net[i]
            gw, gb = grads[i]
            net[i] = (self._adam(f"{key}W{i}", W, gw * scale),
                      self._adam(f"{key}b{i}", b, gb * scale))

    # ---------- update (blueprint + KL early stop) ----------
    def update(self, obs, acts_, old_lp, adv, rets,
               use_value=True):
        clip = float(_p(self.policy_keys, "rl.clip"))
        vcoef = float(_p(self.policy_keys, "rl.vcoef"))
        ec = float(_p(self.policy_keys, "rl.ent_coef"))
        beta = float(_p(self.policy_keys, "rl.kl_ref_coef"))
        epochs = int(_p(self.policy_keys, "rl.n_epochs"))
        # 96 E-5 (LAW-3(ii)): an injected incumbent is the
        # FIXED reference; absent injection, today's per-step
        # snapshot stands (documented trust-region flavor).
        # beta=0 keeps the path fully closed either way.
        ref_pi = None
        if beta > 0.0:
            ref_pi = (self._kl_ref if self._kl_ref is not None
                      else [(W.copy(), b.copy())
                            for W, b in self.pi])
        batch = int(_p(self.policy_keys, "rl.batch_size"))
        target_kl = _p(self.policy_keys, "rl.target_kl")
        N = len(obs)
        idx = np.arange(N)
        epochs_run = 0
        approx_kl = 0.0
        n_updates = 0
        stopped = False
        clip_hits = 0
        clip_seen = 0
        last_pi = None
        last_mb = None
        for _ in range(epochs):
            self.rng.shuffle(idx)
            for s in range(0, N, batch):
                mb = idx[s:s + batch]
                X = obs[mb]
                logits, pacts = self._fwd(self.pi, X)
                z = logits - logits.max(axis=1, keepdims=True)
                e = np.exp(z)
                pi = e / e.sum(axis=1, keepdims=True)
                lp = np.log(pi[np.arange(len(mb)),
                               acts_[mb]] + 1e-300)
                ratio = np.exp(lp - old_lp[mb])
                if target_kl is not None:
                    approx_kl = float(np.mean((ratio - 1.0)
                                              - np.log(ratio)))
                    if approx_kl > 1.5 * float(target_kl):
                        stopped = True
                        break          # SB3: skip this minibatch
                self.t_step += 1
                A_ = adv[mb]
                A_ = (A_ - A_.mean()) / (A_.std() + 1e-8)
                un = ratio * A_
                cl = np.clip(ratio, 1 - clip, 1 + clip) * A_
                act_mask = (un <= cl)
                clip_hits += int(np.sum(~act_mask))
                clip_seen += len(mb)
                last_pi, last_mb = pi, mb
                onehot = np.eye(self.n_act)[acts_[mb]]
                gz = np.where(act_mask[:, None],
                              (-ratio * A_)[:, None]
                              * (onehot - pi), 0.0) / len(mb)
                if ec > 0.0:
                    # loss += ec*(-H)  =>  gz += ec*(-dH/dz)/n
                    # (plan 95 Formula A; ec=0 default takes
                    # the untouched blueprint path — TB-P07)
                    gz = gz + ec * (
                        -entropy_grad_logits(pi)) / len(mb)
                if ref_pi is not None:
                    # loss += beta*KL(pi||pi_ref) (plan 95
                    # Formula B; incumbent = start-of-step
                    # snapshot; beta=0 default = untouched)
                    zr, _ = self._fwd(ref_pi, X)
                    zq = zr - zr.max(axis=1, keepdims=True)
                    eq = np.exp(zq)
                    qref = eq / eq.sum(axis=1, keepdims=True)
                    gz = gz + beta * kl_grad_logits(
                        pi, qref) / len(mb)
                self._backprop(self.pi, pacts, gz, "pi")
                n_updates += 1
                if use_value:
                    v, vacts = self._fwd(self.vf, X)
                    gv = (vcoef * 2 * (v[:, 0] - rets[mb])
                          / len(mb))[:, None]
                    self._backprop(self.vf, vacts, gv, "vf")
            if stopped:
                break
            epochs_run += 1
        # UpdateStats (doc 86 §3.35 binding shape; superset
        # keys approx_kl/n_updates/kl_stopped retained). The
        # loss terms are diagnostics computed on the LAST
        # minibatch state — pure reads, no rng, no update
        # (blueprint parity untouched, TB-P07).
        if last_pi is not None:
            lp_d = np.log(last_pi[np.arange(len(last_mb)),
                                  acts_[last_mb]] + 1e-300)
            pol_loss = clipped_surrogate(lp_d, old_lp[last_mb],
                                         adv[last_mb], clip)
            ent = entropy(last_pi)
            if use_value:
                vd, _ = self._fwd(self.vf, obs[last_mb])
                val_l = value_loss(vd[:, 0], rets[last_mb])
            else:
                val_l = None
        else:
            pol_loss, ent, val_l = 0.0, 0.0, None
        return {"kl": approx_kl,
                "epochs_run": epochs_run,
                "clip_frac": (clip_hits / clip_seen
                              if clip_seen else 0.0),
                "loss_terms": {"policy": pol_loss,
                               "value": val_l,
                               "entropy": ent},
                "early_stopped": stopped,
                "approx_kl": approx_kl,
                "n_updates": n_updates,
                "kl_stopped": stopped}

    # ---------- TrainerPlug ----------
    def spec(self):
        keys = dict(RL_DEFAULTS)
        keys.update(self.policy_keys)
        return {"trainer": self._trainer_name, "keys": keys}

    _trainer_name = "ppo"

    def step(self, rollout):
        gamma = float(_p(self.policy_keys, "rl.gamma"))
        lam = float(_p(self.policy_keys, "rl.lam"))
        adv = gae(rollout["rewards"], rollout["values"],
                  rollout.get("last_value", 0.0),
                  rollout["dones"], gamma, lam)
        rets = adv + np.asarray(rollout["values"], dtype=float)
        return self.update(np.asarray(rollout["obs"], dtype=float),
                           np.asarray(rollout["actions"]),
                           np.asarray(rollout["logp"], dtype=float),
                           adv, rets, use_value=True)


class GRPOTrainer(PPOTrainer):
    _trainer_name = "grpo"
    """Critic-free: episode-level group-relative advantages
    (COMPLETE episodes only — a batch-edge partial episode
    would poison the group baseline; C-4 registered), same
    clipped core, no value net updates."""

    def step(self, rollout):
        rew = np.asarray(rollout["rewards"], dtype=float)
        dones = np.asarray(rollout["dones"], dtype=bool)
        N = len(rew)
        bounds = []
        start = 0
        for t in range(N):
            if dones[t]:
                bounds.append((start, t + 1))
                start = t + 1
        if len(bounds) < 2:
            return {"kl": 0.0, "epochs_run": 0,
                    "clip_frac": 0.0,
                    "loss_terms": {"policy": 0.0,
                                   "value": None,
                                   "entropy": 0.0},
                    "early_stopped": False,
                    "approx_kl": 0.0, "n_updates": 0,
                    "kl_stopped": False,
                    "skipped": "fewer than 2 complete episodes"}
        rets_ep = np.array([rew[a:b].sum() for a, b in bounds])
        g_adv = group_advantages(rets_ep)
        adv = np.zeros(N)
        mask = np.zeros(N, bool)
        for (a, b), ga in zip(bounds, g_adv):
            adv[a:b] = ga
            mask[a:b] = True
        obs = np.asarray(rollout["obs"], dtype=float)[mask]
        acts = np.asarray(rollout["actions"])[mask]
        logp = np.asarray(rollout["logp"], dtype=float)[mask]
        am = adv[mask]
        return self.update(obs, acts, logp, am, am,
                           use_value=False)
