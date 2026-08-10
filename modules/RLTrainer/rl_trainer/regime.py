"""Trainer-regime dispatch (doc 89 FR-3.6; doc 86 §3.5
LAW-1/LAW-4): trainers are co-resident; EVIDENCE TYPE drives
the dispatch — labeled rows -> teach, reward-only experience
-> rl; a mixed batch follows the registered labeled-first
default. FR-3.6 completions (fix F-7): rl.regime is the
POLICY OVERRIDE (auto|teach|rl — forces the phase regardless
of evidence); rl.interleave = [t, r] is the refresher
interleave — on mixed evidence the phase follows the
registered teach:rl repeating cycle instead of labeled-first.
Every decision is a phase_switch audit event naming evidence
counts and the deciding rule."""
from .defaults import RL_DEFAULTS


class RegimeDispatcher:
    def __init__(self, policy=None):
        self.policy = dict(policy or {})
        self.audit = []
        self._last = None
        self._cycle_pos = 0

    def _p(self, key):
        return self.policy.get(key, RL_DEFAULTS[key])

    def dispatch(self, labeled_rows=None, reward_records=None):
        n_lab = len(labeled_rows or [])
        n_rew = len(reward_records or [])
        if n_lab == 0 and n_rew == 0:
            raise ValueError("no evidence: dispatch needs "
                             "labeled rows or reward records")
        override = str(self._p("rl.regime"))
        inter = self._p("rl.interleave")
        if override in ("teach", "rl"):
            to, rule = override, f"override:{override}"
        elif inter is not None and n_lab > 0 and n_rew > 0:
            t_n, r_n = int(inter[0]), int(inter[1])
            pos = self._cycle_pos % (t_n + r_n)
            to = "teach" if pos < t_n else "rl"
            self._cycle_pos += 1
            rule = f"interleave:{t_n}:{r_n}"
        else:
            to = "teach" if n_lab > 0 else "rl"
            rule = "labeled-first"
        self.audit.append({"kind": "phase_switch",
                           "from": self._last, "to": to,
                           "rule": rule,
                           "evidence": {"labeled": n_lab,
                                        "reward": n_rew}})
        self._last = to
        return to
