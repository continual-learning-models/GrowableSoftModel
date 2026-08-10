# P-loop RL Trainer — Operations Runbook (plan 84 D-P6)
2026-07-29 · Track B (modules/RLTrainer) · acceptance basis
doc 89 (FR-3/FR-4/FR-6/FR-7/NFR-8).

## What it is
The weight-RL capability: PPO and GRPO trainers (blueprint-
ported, 9/9 verified vs SB3/gymnasium/MABWiser) that train
the REAL growable organ through the pseudo-target adapter —
zero substrate change; growth operators apply BETWEEN updates
with the quasi-static exactness contract (TX-01: deepen
bitwise, widen <= machine-eps rider footprint).

## Enable / disable
Default: OFF — nothing runs unless a caller drives it (the
module is never imported by serving paths; both-off is
byte-identical, FR-4.3). Configuration rides the EXISTING
set_policy surface (growth_params):
  rl.trainer            ppo | grpo         (default ppo)
  rl.lr /rl.gamma /rl.lam /rl.clip /rl.vcoef /rl.ent_coef
  rl.n_epochs /rl.batch_size /rl.max_grad_norm /rl.adam_eps
  rl.target_kl          None | >0 (KL early stop)
  rl.horizon            rollout length per update
  rl.eval_episode_budget / rl.eval_window / rl.eval_tol
  gate.eval_stream      labeled_slice | eval_episodes
  rl.kl_ref_coef        0 = off; KL-to-incumbent anchor
  rl.regime             auto | teach | rl (phase override)
  rl.interleave         None | [t, r] refresher cycle
Invalid values refuse loudly at BOTH policy doors (create /
set_policy), naming the key (TB-P11/TS-P01).
APPLICABILITY (fix F-5): on the TrainerPlug MLP carrier all
keys act; on the ORGAN path the organ's own optimizer owns
lr/adam_eps/max_grad_norm (those three do not apply there);
everything else acts on both paths.

## Driving a training run
  from rl_trainer.runner import OrganPPORunner
  from rl_trainer.worlds import StagedExpansionWorld
  run = OrganPPORunner(StagedExpansionWorld(seed=0), seed=1,
                       hidden=10, policy={"rl.trainer": "ppo"})
  run.train_rounds(30)            # env_return records accrue
Growth mid-training: apply operators between rounds
(organ.deepen / organ.grow) — exactness guaranteed; adoption
SHOULD go through the eval-episode gate:
  from rl_trainer.eval_provider import (EvalEpisodeProvider,
                                        gate_adjudicate)
  verdict = gate_adjudicate(provider, incumbent_actor,
                            candidate_actor, policy)
Quarantine: eval episodes live in the dedicated seed
namespace (9M+), disjoint from rollout streams; episode cost
is recorded per proof (L17).

## Regimes (teach <-> rl)
Evidence type drives the trainer (RegimeDispatcher, FR-3.6):
labeled rows -> teach; reward records -> rl; mixed ->
labeled-first. Every switch is a phase_switch audit event.
Both retention directions are validated (TX-06 + battery
L-B4): teach knowledge holds across rl phases; reward
behavior holds across teach phases.

## Budgets / overhead
Trainer cost is the caller's compute; serving paths are
untouched. Eval-gate cost = 2 x rl.eval_episode_budget
episodes per proof (recorded in the verdict audit).

## Verification anchors (referee data)
- scripts/verify_rl_math.py (22), verify_rl_vs_industry.py
  (14), verify_quasistatic_full.py (19),
  verify_vs_authoritative_libs.py (9 statistical cases) —
  ALL PASS in the logs shipped with this release (the
  quasi-static set grew from 16 to 19 checks during
  development).
- TB-P07 blueprint bit-identity; TB-P08 organ-adapter
  3-route; TX-01..TX-06 cross-loop; battery
  experiments/rl-battery (L-B1..L-B4 ALL PASS, 3 seeds).

## Failure playbook
- refusal dicts name the offending key/value — fix the
  policy, do not bypass.
- fewer than 2 complete episodes in a grpo rollout => the
  round is skipped (stats say so); raise rl.horizon.
- battery line failure => experiment-failure protocol
  (analyze -> written correction round), never ad-hoc
  retries.


## R3 additions (plan 96, 2026-07-30)
- Rollback semantics: `preference.rollback_mode` is now
  consulted by the facade rollback verb — default `keep`
  preserves learned preference lessons across version
  rollbacks (doc 83 M1); `revert` restores the with-model
  table.
- Explore quota windows: `preference.quota_window:
  "batches:N"` refreshes the explore budget every N
  batches (integer N >= 1; cursor survives snapshot
  restore).
- KL anchor: call `set_kl_reference(...)` on
  PPOTrainer/OrganPPORunner with the promoted incumbent
  for the strict LAW-3(ii) fixed reference; without it the
  per-step/per-round snapshot (trust-region flavor)
  applies. `rl.kl_ref_coef` 0 keeps the path closed.
- Regime-aligned evaluation: after a world crosses its
  boundary/arrival threshold, call
  `provider.align_to(live_world)` so gate episodes score
  the current regime.
- Audit tail: `runner.drain_audit()` returns-and-clears
  collected P-loop events (rl_round + handed verdicts);
  persist via the facade's internal `_rl_audit_write` to
  the model's `rl_audit.jsonl` (JSON lines, replayable).
