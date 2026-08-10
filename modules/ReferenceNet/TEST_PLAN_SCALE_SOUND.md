# TEST PLAN — Scale-Sound Re-Examination (R-rounds)

Status: DRAFT v1.0 for owner review. NOT started. Purpose:
re-adjudicate every SCALE-VOID value verdict under the owner's
scale rulings (2026-07-08): host own params >= 100,000 AND
>= 100x every grown body AND grown only after the base has
trained into shape; topology changes slow and small. The scale
guard must stay SILENT throughout — zero scale_violation events
is itself a bar in every round.

## The qualified configuration (frozen here, verified feasible)
- Host: reference root, d_in=16, hidden=6000 -> OWN params
  108,001 (clears the 100k floor).
- Bodies: reference hidden=5 -> 91 params (1187x hierarchy);
  attention at defaults -> ~833 params (130x hierarchy).
- Data: n_train=1024 with the house corruption (feature dropout
  0.2 + label noise 0.1); held-out eval 2048 CLEAN samples,
  distinct stream. Larger n than the toy rounds because a 100k
  host must face enough data for capacity to matter at all.
- Steps: 2500; growth at steps 500 and 900 — far above
  grow_min_host_steps AND demonstrably post-shaping (the spec
  requires the host's E to have fallen >= 60% from its early
  plateau before the first grow; verified per run, else VOID).
- Seeds 7/8/9. Measured cost ~5 ms/step -> ~0.2 min per run;
  every round below stays under ~15 min of compute. All laws use
  8 active + 8 distractor input features (the corruption and
  distractors keep the 100k host from trivially interpolating).

## R1 — SPU value on reference bodies at scale
(re-adjudicates PS1/PD1, which were toy-scale)
- Scene R1a: y = sin(3x1) + 0.6cos(5x2) + 0.3x3x4 + distractors.
- Scene R1b: same, and the step-500 newborn is deepened at step
  700 (inside its enrollment window) — the deepened-unit case.
- Arms: SPU on (defaults) vs SPU off + budget-fair extra steps
  (house analytic ratio, whole-model denominator).
- Bars: RV1 value (on beats off >= 2/3 seeds, per scene);
  RV3 no-collapse (house clause); RV4 mechanics + the owner's
  execution-check report (units processed in expected windows,
  deepened unit carries blocks>=1 events) + guard silence.

## R2 — body-type value at scale (re-adjudicates BT1/BT2)
- Scene S16: y = sum_i softmax(x[9:16])_i * x[1:8]_i (8-way
  selection — attention's regime), plus corruption.
- Scene P16: the R1a plain law.
- Arms: attention bodies vs param-matched reference bodies
  (hidden matched to attention n_params within +/-10%; the exact
  values computed and FROZEN in the R2 spec before any run).
- Bars: RB1 attention wins S16 >= 2/3 seeds; RB2 parity band
  <= 1.20x on P16 >= 2/3 seeds; RB3 mechanics (bitwise entry,
  scripted-type execution, finiteness, guard silence).

## R3 — SPU on attention bodies at scale
(re-adjudicates AT1/AT3; runs LAST, gated on R2 confirming the
bodies themselves pay at scale)
- Scene: S16. Bodies: attention at both grow sites.
- THREE arms, all pre-declared (separating scale from
  calibration — the toy round confounded them):
  A1 SPU on at DEFAULTS; A2 SPU on CALIBRATED for token masks
  (spu_p_mask=0.125, spu_eta=0.005, spu_gamma=2.0 — declared
  here, never tuned after a run); B SPU off + budget-fair.
- Bars: RA1-default (A1 beats B >= 2/3), RA1-calibrated (A2
  beats B >= 2/3) — reported separately; RA3 collapse watch
  (median exit s_after/s_entry >= rho_floor across events; the
  toy round's 0.436 was the failure signature); RA4 mechanics +
  execution + guard silence.

## Discipline (unchanged house rules)
- One spec per round committed BEFORE its driver; FAIL branches
  pre-written; verdicts verbatim; no renegotiation, no post-hoc
  tuning (R3's calibration is pre-declared HERE).
- Execution-check reports (owner rule) in every round.
- Rounds run in order R1 -> R2 -> R3; each closes with its
  REPORT and commit before the next starts.
- Interpretation guardrail: these rounds SUPERSEDE the
  scale-void toy verdicts; where a toy verdict and an R-verdict
  disagree, the R-verdict governs and the report says so.
