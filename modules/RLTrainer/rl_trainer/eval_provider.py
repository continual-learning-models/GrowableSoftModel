"""N4 EvalEpisodeProvider + reward-world gate adjudication
(doc 89 FR-3.4/FR-4.1; shapes per doc 86 §3.35).

QUARANTINE CONTRACT: evaluation episodes run on a DEDICATED
seed namespace (EVAL_SEED_NS + i, EVAL_SEED_NS = 9_000_000)
that is disjoint by construction from the worlds' rollout
stream namespace (2000 + seed) — the evaluation source is
never touched by training. Both sides of a pair are scored
on the SAME episode seeds (paired comparison; ties are exact
for identical actors). Cost (episodes spent) is recorded per
proof — the L17 discipline, never hidden.

Actors are anything with act_probs(state) -> action probs;
episodes are played GREEDILY (argmax) so scoring is
deterministic and replayable (NFR-1)."""
import numpy as np

from .defaults import GATE_DEFAULTS, RL_DEFAULTS

EVAL_SEED_NS = 9_000_000        # dedicated namespace floor


def _pe(policy, key):
    policy = policy or {}
    return policy.get(key, RL_DEFAULTS[key])


class EvalEpisodeProvider:
    def __init__(self, world_cls, world_seed, eval_seed_base=0,
                 **world_kwargs):
        self.world_cls = world_cls
        self.world_seed = int(world_seed)
        self.eval_seed_base = int(eval_seed_base)
        self.world_kwargs = dict(world_kwargs)

    def _episode_seed(self, i):
        return EVAL_SEED_NS + 1000 * self.eval_seed_base + i

    def align_to(self, live_world):
        """96 E-6 (G-7, FR-3.4): mirror the LIVE world's regime
        so gate evidence scores the CURRENT service conditions.
        A fresh eval world resets _t_global to 0; when the live
        world is past its boundary/arrival threshold, zero that
        threshold in world_kwargs so eval episodes start in the
        live stage. Pre-threshold (or thresholdless) worlds are
        left untouched — today's behavior stays the default."""
        for attr in ("boundary", "arrival_step"):
            thr = getattr(live_world, attr, None)
            if thr is not None and \
                    live_world._t_global >= int(thr):
                self.world_kwargs[attr] = 0

    def _run_episode(self, actor, ep_seed):
        w = self.world_cls(seed=self.world_seed,
                           **self.world_kwargs)
        obs = w.reset(seed=ep_seed)
        total = 0.0
        done = False
        while not done:
            a = int(np.argmax(actor.act_probs(obs)))
            obs, r, done = w.step(a)
            total += float(r)
        return total

    def evaluate_pair(self, incumbent, candidate, policy):
        n = int(_pe(policy, "rl.eval_episode_budget"))
        window = int(_pe(policy, "rl.eval_window"))
        n = min(n, window) if window else n
        seeds = [self._episode_seed(i) for i in range(n)]
        s_inc = [self._run_episode(incumbent, s) for s in seeds]
        s_cand = [self._run_episode(candidate, s) for s in seeds]
        return {"score_inc": float(np.mean(s_inc)),
                "score_cand": float(np.mean(s_cand)),
                "n": n,
                "cost": {"episodes": 2 * n},     # L17: recorded
                "provenance": {"episode_seeds": seeds,
                               "world": self.world_cls.kind,
                               "world_seed": self.world_seed}}

    def spec(self):
        return {"stream": "eval_episodes",
                "world": self.world_cls.kind,
                "world_seed": self.world_seed,
                "seed_namespace": EVAL_SEED_NS}


def gate_adjudicate(provider, incumbent, candidate, policy):
    """Paired-return adjudication (FR-3.4): adopt iff
    score_cand > score_inc * tol for gains (tol default 1.0 =
    strictly better; the existing holdout-gate law's shape).
    Returns verdict + audit (cost, provenance) — refusals are
    verdicts, never silence."""
    out = provider.evaluate_pair(incumbent, candidate, policy)
    tol = float(_pe(policy, "rl.eval_tol"))
    base = out["score_inc"]
    bar = base * tol if base >= 0 else base / tol
    adopt = bool(out["score_cand"] > bar)
    return {"adopt": adopt,
            "score_inc": out["score_inc"],
            "score_cand": out["score_cand"],
            "audit": {"episodes": out["cost"]["episodes"],
                      "n": out["n"], "tol": tol,
                      "provenance": out["provenance"]}}


def gate_eval_stream(policy):
    """The gate.eval_stream selector (FR-4.1) — reads the
    registered key; default labeled_slice."""
    policy = policy or {}
    return policy.get("gate.eval_stream",
                      GATE_DEFAULTS["gate.eval_stream"])


def adjudicate_with_policy(policy, incumbent, candidate,
                           provider=None):
    """Key-driven stream dispatch (FR-4.1; box TR-F3):
    eval_episodes -> paired-return adjudication through the
    provider (refuses loudly without one); labeled_slice ->
    DECLINED here with the owner named — the existing
    holdout gate owns that stream (single-gate law: one
    comparison logic per stream, never two)."""
    stream = gate_eval_stream(policy)
    if stream == "eval_episodes":
        if provider is None:
            raise ValueError(
                "gate.eval_stream=eval_episodes requires an "
                "EvalEpisodeProvider; got provider=None")
        out = gate_adjudicate(provider, incumbent, candidate,
                              policy)
        out["stream"] = "eval_episodes"
        return out
    return {"stream": "labeled_slice",
            "refusal": "labeled_slice adjudication is owned "
                       "by the existing holdout gate (teach/"
                       "grow lifecycle paths); this dispatcher "
                       "serves eval_episodes only"}
