"""OrganPPORunner — the L2 loop that trains the REAL growable
organ with PPO in a reward world (doc 86 §3.4 cadence law;
plan 84 D-P3). Policy = out_width=K organ + softmax (N3, zero
substrate change); value = out_width=1 organ; both driven
through the OrganAdapter pseudo-target path (§10 scheme (c)).
Growth operators apply BETWEEN updates; TX-01 pins the
exactness contract at those instants.

Channel law (FR-4.2, TX-02): this runner consumes env_return
RewardRecords ONLY — any other source is a loud refusal."""
import numpy as np

from .defaults import RL_DEFAULTS, validate_rl_policy
from .math import (entropy_grad_logits, gae,
                   group_advantages, kl_grad_logits)
from .organ_adapter import OrganAdapter
from .records import make_env_return_record, \
    validate_reward_record


def _p(policy, key):
    policy = policy or {}
    return policy.get(key, RL_DEFAULTS[key])


class OrganPPORunner:
    def __init__(self, world, seed, hidden=10, policy=None):
        from reference_net.net import Network
        self.world = world
        # 98 D-2 (R3-1): the L1 door has no merge layer — a
        # None value IS the deletion sentinel, so strip it
        # here (key absent -> default), never int(None) later
        self.policy_keys = {k: v for k, v in
                            (policy or {}).items()
                            if v is not None}
        r = validate_rl_policy(self.policy_keys)
        if r:
            raise ValueError(r["refusal"])
        self.trainer = str(_p(self.policy_keys, "rl.trainer"))
        sp = world.spec()
        self.n_act = sp["n_actions"]
        self.policy_adapter = OrganAdapter(
            Network(sp["obs_dim"], hidden, seed=seed,
                    out_width=self.n_act))
        self.value_adapter = OrganAdapter(
            Network(sp["obs_dim"], hidden, seed=seed + 1,
                    out_width=1))
        self.rng = np.random.default_rng(seed + 2)
        self._kl_ref = None       # 96 E-5 (LAW-3(ii)) anchor
        self.audit = []           # 96 E-7 (G-4) P-loop tail
        # scaler-fit warmup (documented): one zero-target step
        # fixes x/y standardization so serving is defined from
        # step 0; the newborn function stays near zero.
        X0 = np.stack([world.sample_state(i)
                       for i in range(8)])
        self.policy_adapter.organ.train_step(
            X0, np.zeros((8, self.n_act)))
        self.value_adapter.organ.train_step(
            X0, np.zeros((8, 1)))
        self._obs = world.reset(seed=seed + 3)
        self._episode = 0
        self._life = f"rl-{seed}"
        self.records = []

    # ---------- serving ----------
    def policy_logits(self, X):
        return self.policy_adapter.outputs(np.asarray(X))

    def _probs(self, X):
        return self.policy_adapter.probs(np.asarray(X))

    # ---------- records channel (TX-02) ----------
    def ingest_record(self, rec):
        msg = validate_reward_record(rec)
        if msg:
            raise ValueError(f"invalid RewardRecord: {msg}")
        if rec["source"] != "env_return":
            raise ValueError(
                "P-loop consumes env_return records only; got "
                f"source {rec['source']!r} (FR-4.2 channel "
                "separation)")
        self.records.append(rec)

    # ---------- rollout ----------
    def collect(self, horizon=None):
        if horizon is None:                # F-2: key-driven
            horizon = int(_p(self.policy_keys, "rl.horizon"))
        obs_l, act_l, rew_l, done_l, logp_l, val_l = \
            [], [], [], [], [], []
        eid_l = []
        ep_ret = 0.0
        for _ in range(int(horizon)):
            x = np.asarray(self._obs, dtype=float)[None]
            p = self._probs(x)[0]
            a = int(self.rng.choice(self.n_act, p=p))
            v = float(self.value_adapter.outputs(x)[0, 0])
            obs2, r, done = self.world.step(a)
            obs_l.append(x[0]); act_l.append(a)
            rew_l.append(float(r)); done_l.append(bool(done))
            logp_l.append(float(np.log(p[a] + 1e-300)))
            val_l.append(v)
            eid_l.append(self._episode)
            ep_ret += float(r)
            if done:
                self.ingest_record(make_env_return_record(
                    f"ep-{self._episode}", ep_ret, self._life,
                    self._episode,
                    {"world": self.world.spec()["type"]}))
                self._episode += 1
                ep_ret = 0.0
                self._obs = self.world.reset(seed=None)
            else:
                self._obs = obs2
        x_last = np.asarray(self._obs, dtype=float)[None]
        return {"obs": np.asarray(obs_l),
                "actions": np.asarray(act_l),
                "rewards": np.asarray(rew_l),
                "dones": np.asarray(done_l),
                "logp": np.asarray(logp_l),
                "values": np.asarray(val_l),
                "episode_ids": np.asarray(eid_l),
                "group_ids": None,          # ppo rollout
                "seed_tag": self._life,
                "last_value": float(
                    self.value_adapter.outputs(x_last)[0, 0])}

    def drain_audit(self):
        """96 E-7 (G-4): return-and-clear the collected audit
        events (the runner's own round events + phase switches
        / gate verdicts handed to it by the caller). The caller
        persists them via the facade's rl-audit tail."""
        ev, self.audit = self.audit, []
        return ev

    def set_kl_reference(self, adapter):
        """96 E-5 (N-1, design 86 LAW-3(ii)): install the
        COMMITTED incumbent adapter as the FIXED KL-anchor
        reference. Absent injection, train_rounds keeps
        today's per-round snapshot."""
        import copy as _copy
        self._kl_ref = _copy.deepcopy(adapter)

    # ---------- one PPO update round on the organ ----------
    def train_rounds(self, n_rounds, horizon=None):
        if horizon is None:                # F-2: key-driven
            horizon = int(_p(self.policy_keys, "rl.horizon"))
        clip = float(_p(self.policy_keys, "rl.clip"))
        vcoef = float(_p(self.policy_keys, "rl.vcoef"))
        ec = float(_p(self.policy_keys, "rl.ent_coef"))
        beta = float(_p(self.policy_keys, "rl.kl_ref_coef"))
        batch = int(_p(self.policy_keys, "rl.batch_size"))
        gamma = float(_p(self.policy_keys, "rl.gamma"))
        lam = float(_p(self.policy_keys, "rl.lam"))
        stats = {"epochs_run": 0, "approx_kl": 0.0,
                 "kl_stopped": False}     # last round's stats
        for _ in range(int(n_rounds)):
            ro = self.collect(horizon)
            if self.trainer == "grpo":
                # critic-free (FR-3.0): complete-episode group
                # advantages broadcast per step; value organ
                # untouched (TB-P12 pins bit-identity)
                dn = ro["dones"]
                bounds, start = [], 0
                for t in range(len(dn)):
                    if dn[t]:
                        bounds.append((start, t + 1))
                        start = t + 1
                if len(bounds) < 2:
                    # R-R4 loudness law: a skipped round
                    # leaves a trace (trainer path parity)
                    self.audit.append(
                        {"kind": "rl_round_skipped",
                         "trainer": self.trainer,
                         "reason": "fewer than 2 complete "
                                   "episodes",
                         "episodes": len(bounds),
                         "horizon": int(horizon)})
                    continue
                rets_ep = np.array([ro["rewards"][a:b].sum()
                                    for a, b in bounds])
                g_adv = group_advantages(rets_ep)
                adv = np.zeros(len(dn))
                mask = np.zeros(len(dn), bool)
                for (a, b), ga in zip(bounds, g_adv):
                    adv[a:b] = ga
                    mask[a:b] = True
                idx_all = np.flatnonzero(mask)
                # 98 D-1 (R2-1): the SAME canonical epochs+KL
                # frame as the ppo path below — the four loop-
                # level keys are live on this path too; every
                # term copied verbatim (blueprint-parity law)
                epochs = int(_p(self.policy_keys,
                                "rl.n_epochs"))
                target_kl = _p(self.policy_keys,
                               "rl.target_kl")
                ref_ad = None
                if beta > 0.0:
                    if self._kl_ref is not None:
                        ref_ad = self._kl_ref
                    else:
                        import copy as _copy
                        ref_ad = _copy.deepcopy(
                            self.policy_adapter)
                stats = {"epochs_run": 0, "approx_kl": 0.0,
                         "kl_stopped": False}     # per round
                for _ep in range(epochs):
                    self.rng.shuffle(idx_all)
                    stop = False
                    for s0 in range(0, len(idx_all), batch):
                        mb = idx_all[s0:s0 + batch]
                        X = ro["obs"][mb]
                        pi = self._probs(X)
                        lp = np.log(pi[np.arange(len(mb)),
                                      ro["actions"][mb]]
                                    + 1e-300)
                        ratio = np.exp(lp - ro["logp"][mb])
                        if target_kl is not None:
                            akl = float(np.mean((ratio - 1.0)
                                        - np.log(ratio)))
                            stats["approx_kl"] = akl
                            if akl > 1.5 * float(target_kl):
                                stats["kl_stopped"] = True
                                stop = True
                                break
                        A_ = adv[mb]
                        A_ = (A_ - A_.mean()) / (A_.std()
                                                 + 1e-8)
                        un = ratio * A_
                        cl = np.clip(ratio, 1 - clip,
                                     1 + clip) * A_
                        msk = (un <= cl)
                        onehot = np.eye(self.n_act)[
                            ro["actions"][mb]]
                        gz = np.where(msk[:, None],
                                      (-ratio * A_)[:, None]
                                      * (onehot - pi),
                                      0.0) / len(mb)
                        if ec > 0.0:
                            gz = gz + ec * (
                                -entropy_grad_logits(pi)
                            ) / len(mb)
                        if ref_ad is not None:
                            qref = ref_ad.probs(X)
                            gz = gz + beta * kl_grad_logits(
                                pi, qref) / len(mb)
                        self.policy_adapter.apply(X, gz)
                    if stop:
                        break
                    stats["epochs_run"] += 1
                continue
            adv = gae(ro["rewards"], ro["values"],
                      ro["last_value"], ro["dones"],
                      gamma, lam)
            rets = adv + ro["values"]
            N = len(adv)
            idx = np.arange(N)
            epochs = int(_p(self.policy_keys, "rl.n_epochs"))
            target_kl = _p(self.policy_keys, "rl.target_kl")
            # 96 E-5 (LAW-3(ii)): injected incumbent =
            # FIXED reference; absent injection, today's
            # per-round snapshot (trust-region flavor)
            ref_ad = None
            if beta > 0.0:
                if self._kl_ref is not None:
                    ref_ad = self._kl_ref
                else:
                    import copy as _copy
                    ref_ad = _copy.deepcopy(self.policy_adapter)
            stats = {"epochs_run": 0, "approx_kl": 0.0,
                     "kl_stopped": False}     # per round
            for _ep in range(epochs):                # F-4:
                self.rng.shuffle(idx)                # canonical
                stop = False                         # epochs
                for s in range(0, N, batch):
                    mb = idx[s:s + batch]
                    X = ro["obs"][mb]
                    pi = self._probs(X)
                    lp = np.log(pi[np.arange(len(mb)),
                                  ro["actions"][mb]] + 1e-300)
                    ratio = np.exp(lp - ro["logp"][mb])
                    if target_kl is not None:        # F-4: KL
                        akl = float(np.mean((ratio - 1.0)
                                            - np.log(ratio)))
                        stats["approx_kl"] = akl
                        if akl > 1.5 * float(target_kl):
                            stats["kl_stopped"] = True
                            stop = True
                            break
                    A_ = adv[mb]
                    A_ = (A_ - A_.mean()) / (A_.std() + 1e-8)
                    un = ratio * A_
                    cl = np.clip(ratio, 1 - clip,
                                 1 + clip) * A_
                    mask = (un <= cl)
                    onehot = np.eye(self.n_act)[
                        ro["actions"][mb]]
                    gz = np.where(mask[:, None],
                                  (-ratio * A_)[:, None]
                                  * (onehot - pi), 0.0) / len(mb)
                    if ec > 0.0:
                        gz = gz + ec * (
                            -entropy_grad_logits(pi)) / len(mb)
                    if ref_ad is not None:
                        qref = ref_ad.probs(X)
                        gz = gz + beta * kl_grad_logits(
                            pi, qref) / len(mb)
                    self.policy_adapter.apply(X, gz)
                    v = self.value_adapter.outputs(X)[:, 0]
                    gv = (vcoef * 2 * (v - rets[mb])
                          / len(mb))[:, None]
                    self.value_adapter.apply(X, gv)
                if stop:
                    break
                stats["epochs_run"] += 1
        # 96 E-7 (G-4): the runner's own round event
        self.audit.append({"kind": "rl_round",
                           "trainer": self.trainer,
                           "n_rounds": int(n_rounds),
                           "horizon": int(horizon), **stats})
        return stats

    def mean_recent_return(self, k=20):
        vals = [r["value"] for r in self.records[-int(k):]]
        return float(np.mean(vals)) if vals else 0.0
