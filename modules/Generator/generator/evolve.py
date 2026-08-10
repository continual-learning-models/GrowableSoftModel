"""Evolution controller: the `teach` operation + drift detection (M2).

teach = train a candidate -> evaluate on held-out -> promote iff it passes the
gate, else keep the live version (auto-rollback). Reward = downstream
performance (SEAL), never training loss.

Drift-awareness (M2, REQUIREMENTS 2.3/5):
- The gate scores on the most recent `gate_recent_n` held-out examples, and the
  live version is ALWAYS re-evaluated fresh on that slice (no cached score for
  gating) — so when reality moves, the live score drops too and a candidate
  that learned the new reality can win. The gate protects quality against
  today's reality, not yesterday's.
- check_drift() compares the active version's current recent-slice metric with
  its score recorded at promotion time; decay beyond `drift_tolerance` flags
  `needs_reteach` — the institutionalized reflex the orchestrating LLM acts on.
"""
from __future__ import annotations

import time
from typing import Optional

from .config import Config
from .data import stamp_examples
from .registry import ModelRegistry
from .model_manager import ModelManager
from .trainer import Trainer
from .evaluator import Evaluator


class Evolve:
    def __init__(self, config: Config, registry: ModelRegistry,
                 model_manager: ModelManager, trainer: Trainer, evaluator: Evaluator):
        self.config = config
        self.registry = registry
        self.mm = model_manager
        self.trainer = trainer
        self.evaluator = evaluator

    def teach(self, model_id: str, examples: list[dict], mode: str = "sft",
              window: Optional[int] = None,
              recent_n: Optional[int] = None) -> dict:
        """recent_n (M2, symmetric with check_drift): judge the gate on only
        the most recent N held-out examples. Essential after drift — on a
        mixed-era held-out set, an adapted candidate and the stale live
        version can tie (each aces its own era), and the strict gate would
        block adaptation; gating on the recent slice judges against reality
        as it is now."""
        if mode != "sft":
            raise NotImplementedError(
                "only supervised teaching (mode='sft') is implemented; "
                "preference/RL teach modes are deferred.")
        examples = stamp_examples(list(examples))
        rn = recent_n if recent_n is not None else self.config.gate_recent_n
        live = self.registry.active(model_id)

        # Fresh evaluation of the LIVE version on the current recent slice —
        # never a cached score (M2: judge against reality as it is now).
        live_eval = self.evaluator.eval_version(model_id, live, recent_n=rn)

        candidate = self.trainer.train_candidate(
            model_id, examples, parent_version=live,
            window=window if window is not None else self.config.train_window_n)
        cand_eval = self.evaluator.eval_version(model_id, candidate, recent_n=rn)
        # Record the candidate's score at (potential) promotion time — this is
        # the drift baseline for later check_drift() calls.
        self.registry.record_score(model_id, candidate, cand_eval["metric"])

        promoted = self.evaluator.gate(cand_eval, live_eval)
        if promoted:
            self.registry.set_active(model_id, candidate)

        return {
            "model_id": model_id,
            "candidate_version": candidate,
            "candidate_metric": cand_eval["metric"],
            "live_version_before": live,
            "live_metric_before": live_eval["metric"],
            "promoted": promoted,
            "active_version": self.registry.active(model_id),
            "n_examples": len(examples),
            "gate_recent_n": rn,
            "window": window if window is not None else self.config.train_window_n,
        }

    def check_drift(self, model_id: str, recent_n: Optional[int] = None) -> dict:
        """Has reality moved away from the active version's experience?

        Compares the active version's metric on the current recent held-out
        slice against its score recorded at promotion time. Cheap and
        idempotent — meant to be called periodically by the orchestrating LLM
        (or a scheduler); `needs_reteach: true` is the signal to collect fresh
        examples and call teach().
        """
        rn = recent_n if recent_n is not None else self.config.gate_recent_n
        active = self.registry.active(model_id)
        current = self.evaluator.eval_version(model_id, active, recent_n=rn)
        baseline = self.registry.get_score(model_id, active)

        drifted = (baseline is not None
                   and current["metric"] < baseline - self.config.drift_tolerance)
        info = {
            "model_id": model_id,
            "active_version": active,
            "recent_metric": current["metric"],
            "recent_n_evaluated": current["n"],
            "baseline_metric": baseline,          # score at promotion time
            "drift_tolerance": self.config.drift_tolerance,
            "drifted": drifted,
            "needs_reteach": drifted,
            "checked_at": time.time(),
        }
        self.registry.record_drift(model_id, info)
        return info
