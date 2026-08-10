# BACKEND CONFORMANCE KIT — spec (BK1..BK10)

Committed BEFORE any backend exists (DEV_PLAN_BACKEND B0).
Passing the kit is the DEFINITION of a qualified backend; a new
backend (e.g., "xla") qualifies by passing this kit with zero
core changes. Tolerances (DESIGN 6b): trajectory relative error
<= 1e-5 (float64) / 1e-3 (float32); INTEGER structure EXACT.

- BK1 kernel parity: each contract kernel, on pinned inputs,
  matches the numpy judge within tolerance (per dtype).
- BK2 trajectory parity: a fixed 200-step reference-net training
  run (grow at 60, deepen at 120) — held-out predictions within
  tolerance of the numpy run at steps 100/150/200.
- BK3 integer structure EXACT: same run — ledger event sequence,
  SPU processed/skip counts and windows, growth sites, block
  counts identical to numpy (randomness is numpy-sourced, so any
  divergence is a defect, not noise).
- BK4 on-device bit-replay: the same seed twice on the same
  device -> bitwise-identical parameters.
- BK5 serving purity: predict mutates nothing (params, caches,
  events) and works after to_numpy round-trip.
- BK6 persistence: save/pickle produces device-free artifacts;
  load onto numpy AND onto the backend under test; predictions
  match within tolerance; OLD artifacts (pre-backend) load.
- BK7 lifecycle on-device: grow/deepen/remove via the K9 surgery
  round-trip — entry exactness BITWISE on the host output at
  every event; removal restores bitwise.
- BK8 boundary surface: scale guard, pricer fingerprint,
  structure(), build_report run against a backend model without
  error; fingerprint is deterministic and value-sensitive.
- BK9 SPU discipline: budgets respected; Adam moments bitwise
  untouched by self-processing; exact-zero c/bout gradient
  conventions hold (== 0.0).
- BK10 verdict agreement: a mini pre-registered battery (2
  seeds, small scale) — every PASS/FAIL verdict identical to the
  numpy judge's.
- BK14 categorical + causal on device (GSM-I2; replaces the
  BK12 refusal clause whose boundary is now CLOSED): both host
  modes train on torch with judge trajectory parity within the
  dtype band and exact label agreement.
- BK13 loop-operator lifecycle: exact entry BITWISE per
  device/dtype (the L_out=0 argument); trained looped-scope
  trajectory parity vs the judge within dtype tolerance with
  projection-counter equality; k_used INTEGER-equal on the
  pinned cases at both dtypes (pinned away from the stop
  threshold — off-pinned near-threshold flips are disclosed);
  serving purity on device; device-free pickle serves on the
  judge.
