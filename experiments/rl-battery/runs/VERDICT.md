# D-P5 Battery VERDICT — 2026-07-29 (FULL, 3 seeds)
Registered spec: ../BATTERY_SPEC.md (registered before runs).
Data: RESULTS_FULL.json (per-seed curves, all matrices).

| Line | Content | Verdict | Evidence (per-seed) |
|---|---|---|---|
| L-B1 | learning above untrained baseline, all 6 trainer x world cells | **PASS** (3/3 seeds in all 6 cells) | e.g. ppo/stationary 21.0 vs 8.8; grpo/stationary 24.3/22.8/20.7 vs 12.5/8.8/9.2; sensor cells narrower but 3/3 above |
| L-B2 | gate-adjudicated growth >= no-growth (staged world) | **PASS** (2/3: s1 20.33>19.83, s2 14.67>=14.67; s0 14.0<16.0) | adoptions: s0 1, s1 1, s2 0 (gate refused s2's candidate — governance working, not a defect) |
| L-B3 | governed silence: clone candidates win 0/15 | **PASS** (0+0+0) | |
| L-B4 | mixed-regime floors both directions | **PASS** (3/3): agreement 1.00/1.00/0.98 held across the rl phase; closing returns 18.25/23.50/19.50 all above the 8.5 floor | |
| ALL | | **PASS** | |

TRAINER REPORT (not gated; FR-2.2 data line for close-out;
per-seed values in RESULTS_FULL.json — means of 3 seeds):
  staged_expansion : ppo 17.78  grpo 17.06
  stationary       : ppo 23.44  grpo 22.61
  sensor_arrival   : ppo 15.00  grpo 15.72
All 18 cells above their untrained baselines (18/18, not just
the registered 2/3). ppo edges staged/stationary, grpo edges
sensor; differences are within seed spread. Close-out default
recommendation stays ppo (industry battle-tested general
default per doc 89 NFR-7); the data records grpo as fully
competitive on these worlds.
