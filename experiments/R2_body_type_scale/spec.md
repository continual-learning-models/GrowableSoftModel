# R2 spec — body-type value AT SCALE

Committed BEFORE the driver; executes TEST_PLAN_SCALE_SOUND v1.0
R2. Re-adjudicates the scale-void BT1/BT2; R-verdicts govern.

## Configuration (qualified, frozen)
- Host: reference root, d_in=16, hidden=6000 (own 108,001);
  steps 2500; growth at 500 (root:1) and 900 (root:2);
  post-shaping clause: E(step 500) <= 40% of E(step 100), else
  VOID. Seeds 7/8/9; n=1024 train with feature dropout 0.2 +
  label noise 0.1; eval 2048 CLEAN distinct-stream; lr=1e-2.
  SPU OFF in both arms (single variable: body type).
- Scene S16 (selection): y = sum_i softmax(x[9:16])_i * x[1:8]_i
  (8 keys, 8 values).
- Scene P16 (plain): y = sin(3x1) + 0.6cos(5x2) + 0.3x3x4
  (features 9..16 distractors).
- Arms (param-matched, FROZEN): A = attention bodies at defaults
  (833 params each; hierarchy 130x); B = reference bodies with
  hidden=46 (829 params, -0.5%; hierarchy 130x).

## Bars
- RB1: A beats B on held-out clean MSE >= 2/3 seeds on S16.
- RB2: A <= 1.20 x B on P16 >= 2/3 seeds (band frozen).
- RB3: host output bitwise unchanged at every grow; scripted
  types verified (introspection + refine[attention] ledger tag);
  finite outputs; ZERO scale_violation events; post-shaping
  clause met in every run.

## Execution (parallel template)
Arms within a (scene, seed) cell are INDEPENDENT here (equal
steps, no budget coupling), so the pool parallelizes all 12
(scene, seed, arm) runs; per-run result lines print as each
completes. Library code untouched; each run is internally
serial and bit-deterministic.

## Pre-registered interpretations
RB1 FAIL -> attention bodies do not pay even in their predicted
regime at qualified scale — supersedes the toy BT1 PASS. RB1
PASS + RB2 FAIL -> in-regime-only value; user guidance updated.
