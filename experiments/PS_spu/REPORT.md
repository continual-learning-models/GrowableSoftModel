# PS battery — adjudicated verdicts (quoted verbatim)

Driver run: 2026-07-07, seeds 7/8/9, CPU, deterministic.
Bars applied exactly as pre-registered in spec.md; no bar
renegotiated.

## Verdicts

- **PS1 (value): PASS 3/3.** On the noisy/partial scenes the
  WITH arm beats the budget-fair WITHOUT arm on held-out clean
  MSE in every seed — 0.670 vs 2.313, 0.699 vs 1.466, 0.886 vs
  1.820 — with the WITHOUT arm granted 1107 extra plain steps
  (1807 vs 700) by the spec's analytic fairness formula
  (r = 1.58). The local process objective converts functional
  robustness into large held-out value under corrupted training
  data at matched (indeed generous) compute.
- **PS2 (robustness ratio): FAIL (0/3), applied verbatim.** The
  pre-registered metric — degradation RATIO
  mse_corrupted/mse_clean — is worse for WITH in all seeds
  (1.155/0.935/1.065 vs 0.775/0.876/0.825). Descriptive note
  (not a bar): the ABSOLUTE corrupted-eval error is
  substantially better for WITH in 3/3 seeds (e.g. seed 7:
  0.774 vs 1.793); the ratio penalizes the WITH arm's much
  lower clean denominator. As pre-registered: the finding is
  recorded as-is.
- **PS3 (no collapse): PASS.** Across all runs and scenes, no
  event ended with spread below (rho_floor/2) x s_entry
  (event-level clause, s_after recorded per event), and no
  final model has a constant-output leaf on the eval batch.
- **PS4 (mechanics): FAIL on the cost-ceiling clause, applied
  verbatim.** Budget clause PASS (zero overruns; steps <= S_max
  in every event); summary clause PASS (one summary per
  processed step). The wall-clock overhead clause FAILS:
  measured +55/59/55% per step vs the pre-registered <= 25%
  ceiling (a post-battery self-review removed a genuine waste —
  a full-model predict and tree walk paid even on steps with no
  eligible leaf — dropping overhead from the initial +64%; the
  remaining cost is the active loop itself; verdicts and all MSE
  values are bit-identical across the fix, confirming the
  determinism contract). Finding: the 25% ceiling was mis-calibrated at spec
  time — the spec's own fairness formula prices the loop at
  r = 1.58 step-equivalents (158%), and at kilobyte scale
  interpreter constants dominate the measured 64%. The declared
  price was exceeded; recorded as a spec-calibration lesson, not
  renegotiated.
- **CONTROL (clean scene) bonus reading: WITH also won on clean
  data (pre-registered as bonus, not bar).** The asymmetry
  prediction (pays under corruption, neutral when clean) was
  conservative: the mechanism paid in both regimes at this
  scale.

## Reading

The mechanism's value case (PS1) is strong and budget-fair; its
safety case (PS3) held with the real event-level check; the two
FAILs are a metric-construction artifact (PS2's ratio) and a
cost-declaration error (PS4's ceiling), both recorded verbatim
per the honest-map tradition. Artifacts: results/summary.json,
results/verdicts.json, results/events_*.jsonl (committed).

## Addendum — enrollment-window re-adjudication (same day)

Owner ruling after reading the event record: self-processing must
not begin at birth (the stone-carving principle — rough shape
first); the window became `[spu_warmup_steps, spu_newborn_steps)`
with defaults [100, 300), clock-based by explicit ruling (a shape
signal can be transiently false; the clock is deterministic and
auditable). The battery was re-run under the new default; bars
unchanged. Verdicts identical (PS1 PASS 3/3, PS2 FAIL verbatim,
PS3 PASS, PS4 cost-ceiling FAIL, clean bonus PASS), and PS1's
margins IMPROVED on every seed — held-out MSE 0.654/0.587/0.776
(was 0.670/0.699/0.886 under [0, 200)) against the same controls
2.313/1.468/1.822: the delayed enrollment measurably pays.
Per-event interference is slightly larger (+0.0011/+0.0008/+0.0004
mean — mid-life units carry more function to perturb) while the
end result is better; both facts recorded.

## Addendum 2 — relative convergence tolerance (same day)

Owner audit caught a design inconsistency: the convergence
threshold was an ABSOLUTE value (spu_tau = 1e-4) applied to
J_inv, a quantity living in each unit's own output scale — the
same inconsistency the discrimination floor had already been
cured of (relative rho_floor). Measured consequence: the absolute
default fired on ~0% of events (8/1200) — near-vacuous and
scale-biased. Ruling: relative tolerance, the mature
numerical-analysis practice — converged when
J_inv <= spu_tau_rel x s_entry^2 (default 0.01: disagreement
below 1% of the unit's own output variance). Battery
re-adjudicated: verdicts identical (PS1 PASS 3/3 — 0.655/0.608/
0.788 vs the same controls; PS2/PS4 FAIL verbatim; PS3 PASS;
clean bonus PASS); the tolerance now genuinely fires (6% of
events converge early, mean steps 3.88 vs the flat 4.0 of a
vacuous threshold) — the valve works and is fair across scales.
