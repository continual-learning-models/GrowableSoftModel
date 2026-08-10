# Growth-Preference Operations Runbook (plan 84 D-6)
2026-07-28 · product-standard ops procedures (doc 83 §4.8;
doc 89 NFR-8). Battery-verified numbers marked [B].

## 1. Enable / disable
ENABLE (explicit policy act; default is provably inert):
  sy.set_policy(m, growth_params={
      "preference.rule": "thompson",       # consensus default
      "preference.bucket_spec": "b1"})     # required for enabled
  SMS: model_policy(m, {"growth_params": {...}}) — same keys.
  Recommended enabled-mode default: thompson (doc 89 NFR-7);
  mean_clip = deterministic fallback. The D-5 selection battery
  has the final word — its record is SoftModelSystem's
  experiments/preference-battery (BATTERY_SPEC.md defines B-3
  SELECTION; committed verdicts: runs/B1_VERDICT.json,
  runs/B2_VERDICT*.json); consult it before overriding.
DISABLE: preference_reset(m, confirm=true) (SMS door) — removes
  every preference.* key, clears the table, audits; next life is
  byte-identical to fixed (TS-03 standing regression). Setting
  {"preference.rule": "fixed"} disables WITHOUT clearing the
  learned table (paused, resumable).

## 2. Inspect / audit
  preference_inspect(m): stats per bucket (w/m/v/n_raw), rule +
  bucket_spec echo, prior fingerprint, quota, last draws, health
  (b1_fallback_count — a rising count means slope sourcing is
  failing: check probe budget), policy echo, audit tail.
  Draw-volume control: preference.audit_draw_mode
  full (the default; batteries use it) | sampled:N (recommended
  for high-volume production, e.g. sampled:8);
  sampling logs every Nth draw + ALL boundary cases and carries
  rng checkpoints, so replay stays exact (TB-16 referee).

## 3. Fleet prior — build + load
  BUILD (offline): prior_fold(ledgers=[...jsonl], out=path)
  (SMS verb / facade preference_prior_fold). Ledger rows:
  {"bucket", "advantage", "world_type"}. The tool z-standardizes
  per world_type BEFORE pooling (cross-world scale law, M8) and
  records strata; artifact schema prior-v2.
  LOAD: set preference.prior_path (+ prior_weight_cap, default
  12 — keep it light so within-life evidence can move
  decisions); loading audits prior_load with the artifact sha.
  Unreadable artifact => refusal + inert continue (P-4), never
  a crash.

## 4. Rollback / persistence
  The table rides the model store (organ pickle) — no separate
  artifact. preference.rollback_mode: keep (default — lessons
  survive rollbacks; negative evidence is evidence) | revert
  (table restored with the model). Corrupt/unknown snapshot at
  load => refusal + rebuild from the credit-event stream
  (P-1; TB-14/16 referees) — never silent zeroing.

## 5. Schema migration
  Snapshot schema pref-v1; prior artifact prior-v2. On a future
  bump: register a pure migration function (TB-16 pattern),
  amend doc 83 §4.8 P-1 through change control FIRST. Unknown
  versions always refuse-and-rebuild, never guess.

## 6. Overhead budget (P-6, binding)
  Decision-time: one bucket lookup + one draw + one multiply per
  candidate. The ONLY material cost is probing: b1 slope
  sourcing doubles trial probes per candidate (disclosed per
  decision as pref_slope_probe_steps / gain10 records).
  [B] Battery-measured: see SoftModelSystem's
  experiments/preference-battery
  runs (B-1 vs B-2 wall-clock per life; recorded in the battery
  report). Any implementation exceeding the P-6 shape is a
  defect.

## 7. Failure modes → operator actions
  - refusal at set_policy: bad key value; the message names the
    key and range (doc 83 §4.6 table; PARAMETER_REFERENCE 5b).
  - preference_restore_refusal in the audit tail: snapshot
    schema unknown/corrupt — run preference_reset (rebuild path)
    or restore a store version.
  - b1_fallback_count rising: slope probes unavailable —
    verify probe budget policy; b0 operation is legal but
    coarser (the pre-study's k2 lesson: prefer b1).
  - explore_activation absent over long enabled lives with all-
    negative candidates: quota exhausted (one per life by
    default) — intentional; raise preference.explore_quota only
    by deliberate policy.
