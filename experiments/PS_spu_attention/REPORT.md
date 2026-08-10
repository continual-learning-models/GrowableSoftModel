# T6 exam report (verdicts verbatim; spec committed first)

## Verdicts
- AT1 (value): **FAIL 0/3.** The budget-fair SPU-off arm won every
  seed (0.405 vs 0.311; 0.348 vs 0.252; 0.569 vs 0.357). The
  pre-registered branch is recorded verbatim: self-processing
  does not pay on attention bodies at matched budget on this
  scene — at DEFAULT policy values.
- AT3 (no collapse): **FAIL.** 8/400 processed events on seed 7
  end below (rho_floor/2) x s_entry (min exit ratio 0.160), and
  the MEDIAN exit ratio is 0.436 — the hinge fires chronically
  (disc_hits 2-3 of 4 steps) and loses. Diagnosis: whole-TOKEN
  masking is a far stronger perturbation than node masking
  (p_mask=0.25 deletes a quarter of the input tokens), so at the
  default spu_eta the invariance gradient dominates the
  anti-collapse term and pushes the body toward flattening. This
  mechanism also explains AT1: the processing actively hurt.
- AT4 (mechanics): **PASS** — budgets respected, outputs finite,
  execution confirmed (200 events per body, both windows, all
  seeds; every event body_type=attention).

## Process note (on the record)
The first driver run hardcoded the fairness denominator as the
bare root (49 params) instead of the house convention
(net.n_params(), whole model) — over-granting the control ~28x;
it timed out before producing any result and was fixed BEFORE any
bar was adjudicated (commit history).

## Standing and follow-up (recorded, NOT executed)
The realization is mechanically sound and stays user-selectable;
the DEFAULTS are NOT validated for attention bodies — docs say so
explicitly. Candidate calibrations for a future pre-registered
round (owner-gated): per-type defaults (lower spu_eta and/or
lower p_mask for token masks, higher spu_gamma), or masking
d_model channels instead of whole tokens (a gentler perturbation
closer to the reference semantics). Per protocol, nothing is
tuned post hoc inside this round.

## Addendum (owner ruling, scale hierarchy)
The negative verdicts are CONFOUNDED: the 49-param host hosted 673-param bodies (~1/14 — hierarchy inverted; DESIGN 4c now requires host >= 100x body AND >= 100k own params). AT1/AT3 remain on the record for THIS configuration; the future calibration round must be scale-sound before any final conclusion about SPU on attention bodies.

## OWNER RULING (2026-07-08): SCALE-VOID
Same ruling: the configuration is below any meaningful scale; AT1/AT3 are VOID as conclusions about the mechanism (the diagnosis of token-mask strength remains a useful observation); AT4 mechanics stand. Scale-sound re-exam required.
