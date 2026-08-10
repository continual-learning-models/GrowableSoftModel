# Track-B D-P5 Validation Battery — REGISTERED SPEC
2026-07-29 · registered BEFORE any verdict run (house law).
Acceptance basis: doc 89 FR-3.0/3.4/3.5/3.6, FR-4.3, FR-9;
relative-comparison protocol (no absolute contest lines);
evaluation follows the D-5 metric convention (return-per-round
time matrix + adoption/transition counts + retention curve).

## ARMS
Trainers ppo | grpo (rl.trainer key) × worlds staged_expansion
| stationary | sensor_arrival; seeds 0,1,2 per cell; horizon
256 steps/round, 30 rounds/run (hidden 10 organs). Growth arm
(staged + sensor, ppo): at the world boundary a trial copy
grows (grow j=0 hidden 6), brief adaptation (2 rounds), the
eval-episode gate adjudicates (rl.eval_episode_budget 6);
adopt only on the gate's verdict. Governed-silence arm
(stationary): a bitwise-clone candidate adjudicated 5x.
Mixed-regime arm (stationary, ppo): teach(60 supervised
steps) -> rl(10 rounds) -> teach(30) per TX-06 protocol.

## REGISTERED LINES (verdicts; >=2/3 seeds each)
 L-B1 LEARNING: per trainer x world, final mean-20 eval
      return > the untrained-baseline actor's mean return
      on the SAME quarantined eval seeds.
 L-B2 GROWTH UNDER GATE (staged, ppo): the growth arm's
      final mean-20 >= the no-growth arm's (same seeds) —
      relative comparison, gate-adjudicated adoptions only.
 L-B3 GOVERNED SILENCE (stationary): clone candidates win 0
      of 15 adjudications (3 seeds x 5) — no free adoptions.
 L-B4 MIXED REGIME: TX-06 floors at battery scale — oracle
      agreement after rl phase > chance+0.10 AND closing
      return > 0.8 x chance-return floor.
 REPORT (not gated, FR-2.2 discipline): ppo vs grpo final
      means per world — data feeds the close-out
      trainer-default line; means with per-seed values
      printed (statistics discipline).

## OUTPUTS
runs/RESULTS_FAST.json + runs/RESULTS_FULL.json (full
matrices) + runs/VERDICT.md (lines,
per-seed evidence). Any failed line -> experiment-failure
protocol (analyze -> written correction round), never ad-hoc
retries.

## J-1 STACKED ARM (registered 2026-07-29 BEFORE its run)
Both RL loops ON together, staged_expansion world, seeds
0,1,2: P-loop trains the organ (ppo, 10 rounds); at the
boundary the ENABLED preference part (mean_clip, b0,
min_count 1) ranks {grow, deepen}; winner trial-grown +
2-round adaptation; eval-episode gate adjudicates; on
adoption the S-loop is credited from the eval-return delta
(one-metric law); 10 more rounds.
 L-J1-a pipeline completes 3/3 seeds (no refusal, no error);
 L-J1-b the credited bucket exists with w>0 in 3/3 (S-loop
        learned from the P-loop world);
 L-J1-c post-verdict training stays sound: final mean-20
        return > untrained baseline in >=2/3 seeds;
 L-J1-d every adoption (if any) carries a gate audit with
        episode cost (L17) — zero unaudited adoptions.


## R2 ARMS (fix F-8; registered 2026-07-29 BEFORE their run)
 ARM-KL (plan D-P5 'KL-to-incumbent option as a registered
 comparison'): stationary world, ppo, seeds 0,1,2 —
 rl.kl_ref_coef 0.0 vs 0.5; lines:
   L-R2K-a both arms learn (final > untrained baseline,
           >=2/3 seeds each);
   L-R2K-b anchored arm's policy drift (L1 param distance
           from the round-10 snapshot to final) is SMALLER
           than free arm's in >=2/3 seeds (the anchor
           protects the incumbent — direction check, no
           absolute line).
 ARM-IL (interleave vs none): mixed-evidence phase schedule
 rl.interleave=[2,1] vs labeled-first default, dispatcher
 level, seeds n/a (deterministic): line L-R2I = the
 registered cycle is produced exactly and every switch is
 rule-named in the audit (TR-F7 at battery scale: 12-step
 schedule check).

## R3 ARM (plan 96 E-9 / N-2; registered 2026-07-30 BEFORE
## its run)
 ARM-ILT (interleave-vs-none at TRAINING level; caller-
 scripted per LAW-3 ownership — the dispatcher only NAMES
 phases, the caller executes them): stationary world, seeds
 0,1,2. Both arms first learn a supervised probe mapping
 (teach phase: organ train_step on 32 labeled rows, 30
 steps), then:
   interleaved arm: 6 macro-cycles driven by the
     rl.interleave=[1,1] dispatcher decision sequence —
     each teach turn = 5 supervised steps on the SAME rows,
     each rl turn = 1 train_rounds(1);
   rl-only arm: same total rl budget (6 train_rounds), NO
     teach turns after the initial phase.
 Lines:
   L-R3T-a retention: interleaved arm's final teach-probe
     MSE < rl-only arm's in >=2/3 seeds (direction check —
     mixed evidence protects the supervised mapping);
   L-R3T-b both arms' rl capability moved: mean recent
     return after training > the untrained baseline of the
     same seed in >=2/3 seeds for BOTH arms (interleaving
     must not kill learning).
 GROWTH-ARM ALIGNMENT (G-7 re-verify): growth_arm's gate
 provider now calls prov.align_to(grow.world) — the
 candidate is judged under the CURRENT (post-boundary)
 regime; L-B2 line re-run under alignment.

## R3 ARM-ILT run-1 FAILURE ANALYSIS + REVISION (2026-07-30,
## experiment-failure protocol: analyze first, then re-design;
## run-1 result KEPT in runs/R3_ARM_RESULTS.json)
 RUN-1 RESULT: L-R3T-a false 3/3 (interleaved MSE WORSE).
 MECHANISM (root cause, confirmed in code): teach and RL
 both shape the SAME organ head h (pseudo-target adapter,
 doc 86 SS10 scheme (c)); run-1's probe mapping was an
 ARBITRARY function unrelated to the world's reward, so the
 two phases pulled h toward CONFLICTING targets — each
 teach re-fit re-inflated RL advantages, so the ending rl
 turn perturbed a freshly-fit surface (worse retention than
 the converged rl-only arm). The arm measured head-conflict
 under adversarial evidence, NOT the N-2 regime semantics
 (mixed evidence about ONE task).
 ARM-ILT-v2 (registered BEFORE its run): identical schedule
 and budgets, but the labeled evidence is TASK-CONSISTENT:
 teach rows = (state, oracle one-hot) pairs of the SAME
 world (32 rows from sample_state, oracle at stage 0).
 Lines unchanged in direction:
   L-R3T-a' retention: interleaved final oracle-probe MSE
     < rl-only arm's in >=2/3 seeds;
   L-R3T-b' both arms beat the same-seed untrained floor on
     mean recent return in >=2/3 seeds.
 Run-1 verdict is recorded as the CONFLICT-EVIDENCE control
 (informational line, no pass bar).
