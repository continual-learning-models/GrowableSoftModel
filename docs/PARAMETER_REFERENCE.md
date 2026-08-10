# Parameter Reference — every controllable parameter of the system

The single catalog (S9.6 D-DOC). One table per method family.
"How to set" names the surface; the ACCESS TIERS are:

- **Tier 1 (standard users + AI): SoftModelSystem** — the product
  gate. Birth-time: `train_converge(..., policy={...})`.
  Life-time: `model_policy(model_id, {...})`. Both reflected to
  the SMS CLI (`sms-cli <ws> model_policy '{...}'`) and SMS MCP.
- **Tier 2 (advanced, direct lib)**: the facade (`System`), the
  lib MCP server (`python3 -m mcp.mcp_server`), the lib CLI —
  `create_model(policy=...)` at birth, `set_policy(...)` in life.

All validation lives in the LIB, on every surface (facade, lib
CLI, lib MCP, SMS verbs/CLI/MCP — one choke point): unknown
top-level keys are refused loudly, the refusal naming every
offending key (the name gate of design 104; T-1..T-19,
tests/unit/test_policy_key_gate.py). Family validators run at
BOTH doors: spu_* (keys AND values, engine-validated),
innovation_* (keys AND values — full range/enum validation),
numeric_head (enum), substrate value (registry), and
everything inside `growth_params` (keys; plus
preference.*/rl.* values); substrate_params is type- and
signature-checked. att_* validates NAMES only — no att value
validator exists. The remaining base-key VALUES (the 23
scalar training/self-study/refound keys) are not yet
range-checked (FR-5, pending); the "values" column documents
the intended ranges.
A per-model value is stored in
that model's own `policy.json` (plain JSON, one file per model)
and installed onto the organ at the next study/load.

## 1. Model policy (base keys — `DEFAULT_POLICY`, core/lifecycle.py)

| key | meaning | default | values | acts |
|---|---|---|---|---|
| substrate | model body family | "mlp" | see `list_substrates` | birth |
| numeric_head | numeric output head | "point" | "point"/"dist" | birth |
| | 60A: "dist" (value + uncertainty) serves on mlp, transformer AND growable_attention (each declares support in its SUPPORTED_HEADS whitelist); sequence refuses at the door; undeclared substrate+head combinations always refuse loudly, never crash | | | |
| gate_tol | numeric-match tolerance of the gate (lifecycle AND factory evaluator — one rule; practice_reach stays independent) | 0.5 | >= 0 | live |
| substrate_params | constructor knobs for the chosen substrate (section 2) | {} | per-substrate | birth ONLY |
| widen_sat | siting: container saturation to consider widening | 0.5 | (0,1] | live |
| uniform_factor | siting: top-unit instability < f x mean = uniform | 2.0 | >= 1 | live |
| escalate_disposed | siting: consecutive gate-disposed refines -> widen | 2 | >= 1 | live |
| selfstudy_steps | self-study block study steps (was hardcoded past study_steps) | 200 | > 0 | live |
| selfstudy_lp_flat | self-study LP-flat threshold | 0.02 | > 0 | live |
| selfstudy_sat_demand | self-study growth-demand saturation | 0.35 | (0,1] | live |
| selfstudy_quiz_n | self-quiz sample size | 96 | >= 1 | live |
| selfstudy_var_eps | variation self-check input perturbation | 0.05 | > 0 | live |
| selfstudy_var_flag | variation inconsistency flag factor | 3.0 | > 0 | live |
| retention_sag | self-review sagging-trend threshold | -0.03 | <= 0 | live |
| refound_inv_alarm | Phi trigger: inversion alarm level | 0.5 | (0,1] | live |
| refound_inv_consecutive | Phi trigger: consecutive probes | 3 | >= 1 | live |
| refound_disposed_alarm | Phi trigger: disposed-growth count | 2 | >= 1 | live |
| refound_lp_flat | Phi trigger: flat-LP threshold | 1e-3 | > 0 | live |
| growth_params | per-model growth-policy overrides (section 5) | {} | keys of section 5 | life |
| max_params_mult | growth budget: params cap = initial x this | 10 | int >= 1 | grow gate |
| max_depth | growth budget: per-branch depth cap | 4 | int >= 1 | grow gate |
| gate_recent_n | commit gate judges on the recent-N holdout slice | None | int/None | commit |
| practice_reach | practice acceptance reach (x TOL) | 1.0 | float | practice |
| consolidation_lr | practice consolidation learning rate | 1e-3 | float | practice |
| study_steps | default steps per study call | 200 | int | study |
| attempt_sigma | attempts: perturbation scale | 0.05 | float | attempts |
| n_attempts | attempts: number of variants | 8 | int | attempts |

ROUTED FAMILIES (validated at the door, installed at study/load):
`spu_*` keys (section 4) and `att_*` keys (section 3) may be set
directly in the model policy / set_policy / model_policy.

## 2. substrate_params (birth-time constructor knobs)

Signature-derived per substrate; `d_in`/`hidden`/`mode`/`vocab`
are BIRTH-DERIVED from the data's own shape (self-shaping) and
never accepted. `seed` overrides the global config seed for THIS
model (per-model reproducibility).

| substrate | accepted keys |
|---|---|
| mlp / mlp_plus | lr, seed, new_class_bias, nll_clamp |
| transformer / transformer_plus / sequence | lr, seed, d_model, n_layers, n_heads, backend, window, inner_lr_factor, new_class_bias |
| growable_attention | lr, seed, d_model, n_layers, heads_spec, causal, selfproc, window, inner_lr_factor, new_class_bias, backend |

growable_attention specifics: `heads_spec` = per-layer head-width
lists, e.g. `[[1,3],[2,1]]` (ragged by design); `backend` =
None (follow the system compute policy, set_compute_policy) |
a registered backend name | a backend instance — CPU/GPU
selection, ga-backend-v1 batch (internal design doc 32; note: after
long divergent training histories, near-tie structural
rankings may differ between backends within float tolerance
— arithmetic physics, not a defect); `selfproc` (bool,
default false) switches the entropy-band discipline on at birth;
`causal` (bool) selects the sequence-token path.
Param-interface batch (internal design docs 21-23): `window` (int >= 1,
default 16) — causal context length / positional-table size;
`inner_lr_factor` (number > 0, default 0.3) — two-timescale
inner-net learning-rate factor (transformer-family hosts; the
sequence host keeps its own class default 0.1);
`new_class_bias` (number <= 0, default -10) — vocabulary-entry
logit bias (old-class leakage bounded by ~e^bias; categorical
hosts).
`nll_clamp` (number > 0, default 10) — numeric-dist variance-logit
clamp (mlp host; passed per kernel call, backend singleton never
mutated). Generator factory config keys: `gate_tol` (>= 0, default
0.5) and `max_rules` (>= 1, default 32) — the factory gate
tolerance and the rule-mining cap.

## 3. Attention policy (`att_*`, 13 keys — POLICY,
## core/substrates/growable_attention.py)

Per-model via the routed `att_*` keys; module POLICY is the
default table (and stays the direct-construction override surface
for experiment drivers). Installed as `organ._att_policy`; every
in-class read is instance-first (`_pol`).

| key | meaning | default |
|---|---|---|
| att_birth_heads | heads per layer at birth | 1 |
| att_birth_dh | head width at birth / head_add | 1 |
| att_widen_m | columns added per widen | 1 |
| att_kappa | widen trigger: max u_h >= kappa x mean | 2.0 |
| att_beta | add trigger: PR >= (1-beta) x H | 0.1 |
| att_lambda | J_att weight in the training objective | 0.05 |
| att_warmup | no self-processing before this step | 100 |
| att_h_lo | entropy band lower factor (x log F_i) | 0.2 |
| att_h_hi | entropy band upper factor (x log F_i) | 1.0 |
| att_selfproc_heads | allow-set of heads (None = all) | None |
| att_head_age_min | head age gate for discipline + widen | 100 |
| att_probe_steps | gate probe epoch: training-lane steps before adjudication (doc 18) | 20 |
| att_window | widen-failure memory window (events) | 200 |

## 4. SPU policy (`spu_*`, 14 keys + 1 optional —
## DEFAULT_SPU_POLICY, engine/spu/spu_policy.py)

Per-model via the routed `spu_*` keys; validated by the ENGINE
(`validate_spu_policy`); installed as `organ._spu_policy` at
study/load. `set_policy(spu_enabled=True)` is the one on/off
surface.

| key | meaning | default |
|---|---|---|
| spu_enabled | master switch | False |
| spu_scope | which inner nets self-process | "newborn_inner_nets" |
| spu_warmup_steps | enrollment age (no SPU before) | 100 |
| spu_newborn_steps | retirement age (newborn window end) | 300 |
| spu_S_max | max self-processing count per event | 4 |
| spu_K | perturbation copies | 4 |
| spu_p_mask | hidden-unit drop probability | 0.10 |
| spu_eta | local SGD step size | 0.01 |
| spu_clip | update-norm cap (fraction of ||theta||) | 0.05 |
| spu_tau_rel | relative convergence tolerance | 0.01 |
| spu_gamma | discrimination weight | 1.0 |
| spu_rho_floor | relative spread floor | 0.5 |
| spu_n_min | minimum batch rows to run | 8 |
| spu_every | process every m-th evolution step | 1 |
| spu_objective | OPTIONAL: "mask" (default) or "analytic" (closed-form expectation; leaf-only, blocks fall back) | "mask" |

## 5. Growth policy (43 keys — DEFAULT_GROWTH_POLICY,
## reference_net/growthpolicy)

Per-model via `growth_params` (a dict inside the model policy);
installed as `organ._growth_policy`; the seven net.py read sites
are instance-first. Unknown keys refused loudly at the door.

| key | meaning | default |
|---|---|---|
| extrapolator | learning-curve extrapolation part | "domhan2015" |
| forecastability | forecastability scorer part | "spectral_entropy" |
| changepoint | changepoint detector part | "bocpd" |
| backtest | backtest protocol part | "rolling_origin" |
| pricer | growth pricer part | "zero_attach_v1" |
| combiner | decision combiner part | "threshold_policy" |
| seed | growth-machinery seed | 0 |
| forecastability_min | minimum forecastability score | 0.25 |
| changepoint_max | maximum changepoint probability | 0.2 |
| backtest_max_err | maximum backtest error | 0.25 |
| max_rel_ci_width | maximum relative CI width | 0.5 |
| eta_target | cross-scale target step eta_t (t = h - eta*dL/dh) | 0.5 |
| instrument_min_len | direction-instrument validity floor (min history) | 64 |
| bocpd_recent | BOCPD recent-mass window | 32 |
| stall_k | stall detection: window count | 3 |
| stall_eps | stall detection: epsilon | 0.05 |
| saturation_norm | saturation gradient-norm floor | 1e-4 |
| energy_floor | energy floor for site eligibility | 1e-3 |
| min_energy_points | minimum energy history points | 64 |
| min_ledger_events | minimum ledger events to decide | 3 |
| min_window_rows | minimum window rows | 64 |
| probe_steps | probe-training steps | 300 |
| probe_lr | probe-training learning rate | 0.05 |
| refine_hidden | hidden size of a refined body | 8 |
| max_blocks | widen budget: max blocks | 8 |
| log_path | growth log path (None = default) | None |
| growth_mode | "adaptive" or "widen_only" | "adaptive" |
| grow_body_type | inner-body family: "reference"/"attention" | "reference" |
| grow_attention_d_model | attention body: d_model | 8 |
| grow_attention_layers | attention body: layers | 1 |
| grow_attention_heads | attention body: heads | 2 |
| grow_attention_ffn | attention body: FFN width | 16 |
| grow_port_type | growth coupling port (doc 35 D7): "fullwidth" only for new growth; "legacy_scalar" load-only | "fullwidth" |
| grow_body_out_width | fullwidth body vector output width k_g (int >= 1) | 1 |
| grow_min_host_ratio | scale guard: host params >= ratio x body | 100.0 |
| grow_min_host_params | scale guard: absolute host floor | 100000 |
| grow_min_host_steps | scale guard: host age floor (steps) | 100 |
| grow_scale_guard | guard mode: "warn"/"refuse" | "warn" |
| loop_enabled | loop operator master switch | False |
| loop_m | loop width (L_in rows) | 8 |
| loop_K_max | Picard iteration budget | 32 |
| loop_tol | loop convergence tolerance | 1e-6 |
| loop_rho_max | contraction cap (certificate) | 0.6 |

## 5b. Growth-preference policy (`preference.*`, 23 keys —
## PREFERENCE_DEFAULTS, reference_net/growthpolicy/preference.py;
## plan 84 D-4, doc 83 §4.6)

Per-model via `growth_params` (same door as section 5; the
registry set accepts the names, the facade validates the VALUES
loudly). Default rule `fixed` is PROVABLY INERT (byte-identity
tests TI-02/TS-03); enabling is an explicit policy act.

| key | meaning | default | range |
|---|---|---|---|
| preference.rule | selection rule | fixed | fixed, mean_clip, thompson, ucb, eps_greedy |
| preference.rule_mix | raw-score blend weights | {} | dict rule->w, sum<=1 |
| preference.bucket_spec | context granularity | b1 | b0, b1 (b2 reserved) |
| preference.decay | EMA fold decay | 0.98 | (0.5, 1] |
| preference.credit_weights | K-window credit weights | [1, .5, .25] | 1..5 positive |
| preference.clip_lo | envelope lower clip | 0.5 | (0, 1] |
| preference.clip_hi | envelope upper clip | 2.0 | [1, 10] |
| preference.min_count | floor before deviation | 3 | >=1 |
| preference.explore_quota | below-line trials per window | 1 | >=0 |
| preference.quota_window | quota window | life | life, batches:N |
| preference.slope_probes | two-horizon slope sourcing | True | bool (ignored under rule=fixed) |
| preference.slope_cuts | b1 band boundaries | [-0.01, 0.01] | sorted floats |
| preference.eps | eps rule exploration prob | 0.1 | [0, 1) |
| preference.ucb_c | ucb optimism constant | 1.0 | >0 |
| preference.rollback_mode | table on rollback | keep | keep, revert |
| preference.prior_path | fleet-prior artifact | "" | path |
| preference.prior_weight_cap | prior weight cap | 12 | >0 |
| preference.bocpd_coupling | drift deepening on/off | False | bool |
| preference.bocpd_depth | deepening rescale rho | 0.5 | (0, 1] |
| preference.ts_seed_offset | draw-stream seed offset | 10000 | int |
| preference.audit_draw_mode | draw-event volume | full | full, sampled:N |
| preference.world_type | prior-load stratum selector | supervised | nonempty string |
| preference.explore_draw_min | explore favorability threshold | 1.2 | >1 |

## 6. Verb arguments (call-scoped parameters)

Facade (Tier 2) signatures; SMS mirrors train/serve flows.

| verb | signature |
|---|---|
| create_model | (model_id, description='', holdout=None, policy=None, substrate=None) |
| study | (model_id, examples, steps=None) |
| teach | (model_id, examples, window=None, recent_n=None) |
| grow | (model_id, k_nodes=2, hidden=16, body_type=None) |
| grow_attention | (model_id, layer=0, tol=1.05) |
| set_attention_selfproc | (model_id, on, heads=None) |
| set_policy | (model_id, **updates) |
| preference_inspect | (model_id) — read-only preference state dump (stats, echoes, tails) |
| preference_reset | (model_id) — revert to inert; SMS door adds confirm=true consent |
| preference_prior_fold | (ledgers, out=None) — offline fleet-prior fold tool (prior-v1) |
| widen | (model_id, container='root', k=2) |
| loop | (model_id, container='root', m=None) |
| remove_loop | (model_id, container='root') |
| refound | (model_id, steps=4000, mode='fresh') |
| run_self | (model_id, block_budget, suites=None, allow_growth=False) |
| run_course | (model_id, curriculum, policy=None) |
| infer | (model_id, input_, working=False, version=None) |
| predict_dist | (model_id, input_, working=False, version=None) |
| practice_update | (model_id, inputs, passed) |
| attempts | (model_id, inputs) |
| add_feature | (model_id, name, default=0.0) |
| add_holdout | (model_id, examples) |
| commit | (model_id, note='') |
| reset | (model_id) |
| rollback | (model_id, to) |
| evaluate | (model_id, suites=None, recent_n=None) |
| trajectory | (model_id, current_stage=None) |
| growth_report | (model_id, top=6) |
| check_drift | (model_id, recent_n=None) |
| discoveries | (model_id, version=None) |
| get_versions | (model_id) |
| card | (model_id) |
| attribution | (model_id, suites) |
| self_review | (model_id, probe_inputs=None) |
| list_models | () |
| list_substrates | () |
| recommend_substrate | (sample_examples) |
| store | (model_id) |
| deepen | (model_id, m=None, position=None, recipe=None, recipe_params=None, zero_side=None, scope=None, force=False) |
| remove_block | (model_id, k, force=False) |
| remove_grown | (model_id, key, force=False) |
| propose | (model_id, move, args=None) |
| plan_validate | (model_id, plan) |
| plan_run | (model_id, plan, examples=None, steps_between=10) |
| trial | (model_id, move, args=None, budget_steps=6, examples=None) |
| describe | (model_id) |
| assess | (model_id) |

Growth-control verbs (59B): `deepen` — the delta operator, one
verb for every organ kind: hosts insert a whole layer (position
None = END = global deepening; scope refused loudly), network
models append/insert a composition block; `scope=[...]` confines
the block to a sub-scope and attaches the closure advisory to
the response as `warning` (multi-scale growth adds capacity in
place; deepen adds a processing stage — the two are distinct
axes). `force` bypasses refusing GATES only, never params
budgets. `remove_block`/`remove_grown` — audited negative
mirrors (network family; host shrink = version rollback via
commit/get_versions/rollback). `propose` — FR-14 dry-run (gate
verdicts + cost estimate, no mutation). `plan_validate`/
`plan_run` — FR-17 rule plans, inline dict or parameter-file
path; training rows via `examples` (None = pure structural
mode). `trial` — FR-15 bounded probe of (move, args), report
{losses, realized_gain, wall_ms}, unconditional rollback;
requires `examples`. `describe`/`assess` — read-only anatomy /
dynamics reports (JSON-safe).

Key verb parameters: `grow.body_type` — explicit inner-body
family (None -> the model's growth policy decides);
`grow_attention.tol` — held-out gate tolerance (accept iff after
<= tol x before); `set_attention_selfproc.heads` — allow-set
entries `h` or `[l, h]`.

## 7. Worked examples

### 7.1 Grow a network with a chosen body type
```python
# Tier 2 (facade)
sy.set_policy("m", growth_params={"grow_body_type": "attention"})
sy.grow("m", k_nodes=1)            # or: sy.grow("m", body_type="attention")
```
```bash
# Tier 1 (SMS CLI)
sms-cli ws model_policy '{"model_id":"m","updates":{"growth_params":{"grow_body_type":"attention"}}}'
```

### 7.2 Enable SPU with a custom K
```python
# Tier 2
sy.set_policy("m", spu_enabled=True, spu_K=8)
```
```bash
# Tier 1
sms-cli ws model_policy '{"model_id":"m","updates":{"spu_enabled":true,"spu_K":8}}'
```

### 7.3 Build a growable_attention model with substrate_params
### and switch its discipline on
```python
# Tier 2
sy.create_model("m", holdout=rows, substrate="growable_attention",
                policy={"substrate_params": {"d_model": 16,
                                             "heads_spec": [[1, 3]],
                                             "seed": 7}})
sy.set_attention_selfproc("m", True)          # or heads=[[0, 1]]
```
```bash
# Tier 1 (birth-time via train_converge)
sms-cli ws train_converge '{"model_id":"m","snapshot":"s-...","substrate":"growable_attention","policy":{"substrate_params":{"d_model":16,"seed":7},"att_lambda":0.1}}'
```

Freshness: tests/unit/test_param_reference.py (T25) asserts every
key of every live defaults dict appears here, every facade verb
appears with its signature, and the Tier-1 forms (model_policy /
train_converge policy=) are documented — this catalog cannot rot.

## Innovation self-assessment (selfassess-v1; internal design docs 26-30)

MDL compression-progress self-assessment (core/selfassess.py).
All keys live in the MODEL POLICY (C2 channel); component OFF
unless innovation_slice_mode is set. Facade verb:
innovation_report(model_id) -> read-only report (refusal when
not enabled). MCP tool: innovation_report {model_id}. Public
experiment entry point: core.selfassess.install(organ, policy).

| key | default | meaning |
|---|---|---|
| innovation_method | "mdl" | criterion part (registry; not hot-swappable) |
| innovation_slice_mode | None | None=OFF; level_tag / length_bucket / task_family |
| innovation_slice_min_obs | 256 | positions before a slice gets verdicts |
| innovation_progress_window | 2 | cycles for the progress test |
| innovation_progress_eps | 0.01 | relative flatness threshold |
| innovation_ema_alpha | None | optional cycle-mean smoothing (0<a<1) |
| innovation_trial_cycles | 2 | cycles a trial trains before verdict |
| innovation_cost_form | "const" | const | half_log_n (doc 27 3.4) |
| innovation_cost_per_param | 0.002 | nats/param (const form; pilot-set) |
| innovation_amortize_h | 1.0 | cost amortization horizon (>=1) |
| innovation_no_harm_eps | 0.005 | mastered-slice no-harm tolerance |
| innovation_ladder_order | (widen, head, structure) | escalation order |
| innovation_class_fails | 2 | consecutive fails advancing the ladder |
| innovation_backoff_levels | ((2,2),(4,4)) | (rejections, cadence) pairs |
| innovation_allow_mse | False | admit MSE organs (const form only) |


## Base-Design Adjustment batch (adj-p0..adj-a5, docs 51-56)

- LEDGER CONVENTION AMENDMENT (52 SR-22, owner-approved):
  every growth event now records its four spec objects
  verbatim under the `specs` key (the event's birth
  certificate: structure/wiring/placement/birth). Pre-batch
  records lack the key (absence = pre-adjustment event);
  readers treat it as optional. This is the batch's ONE
  declared ledger-shape change.
- No new policy keys were introduced by Part A. The family
  optimizer-kernel constants inside foundation/coupling.py
  (Adam 0.9/0.999/1e-8 with the 0.1/0.001 mul-association;
  instability EMAs 0.95/0.05) are pre-existing kernel
  doctrine, extracted verbatim (PR-4 classification:
  DERIVED-from-family-doctrine; never growth parameters).

### B3 control-surface policy keys (docs 55 s3-B3/B5/B6)

PRODUCT ROUTING (59C stage 0 = doc 61 G8; +60D): these
fourteen keys (ten of 59C + the four 60D aspect keys) are
NOT members of DEFAULT_GROWTH_POLICY (the dict's key-set is
pinned by write-once fixtures); they are registered in the
EXTENDED_GROWTH_KEYS set (reference_net/growthpolicy) and pass
through EVERY served tier's policy door — facade/CLI/MCP
`set_policy(growth_params={...})` and the SMS product panel
`model_policy(updates={"growth_params": {...}})` — installed
onto the organ at every load, persisted in policy.json
[boxes: test_g8_policy_key_registration.py incl. the
half-speed EXACT identity through the product gate].

| key | default | class | meaning |
|---|---|---|---|
| growth_auto_snapshot | True | EMPIRICAL (safety default) | pre-event snapshot before every growth/removal (FR-13); off = fully manual history |
| growth_snapshot_keep | 8 (_DEFAULT_KEEP, censused) | EMPIRICAL | in-memory snapshot retention window — the FINITE-rollback guarantee (FR-18); disk persistence is unbounded |
| gate_deepen_mode | "advise" | USER-SET (threshold/mode values are the user's own choice; no derivation implied — owner ruling 2026-07-23) | G-DEEPEN enforcement strength: advise/warn/refuse; force=True overrides (C-5) |
| gate_seam_min_width | 1 | USER-SET (permissive default; user's own choice, no derivation implied) | G-SEAM (58 D-5.1): minimum junction width for a NEW growth event (grow / grow_site); consulted pre-mutation in preset_rho/preset_site |
| gate_seam_mode | "advise" | USER-SET | G-SEAM enforcement: advise/warn/refuse; force overrides (C-5) |
| gate_nest_mode | "advise" | USER-SET | OD-3 nesting gate (58 D-5.2): growing INSIDE a grown body checks min(new junction, body mouth) vs gate_seam_min_width; mouth = the body's ledger-resident nested_birth record (grand-nesting deeper mouths are not walked — stated limitation); advise/warn/refuse; force overrides |
| gate_scope_min_width | 1 | USER-SET (permissive default) | G-SCOPE (58 D-5.3): minimum |scope| for scoped deepening; global deepen (scope=None) is never gated |
| gate_scope_mode | "advise" | USER-SET | G-SCOPE enforcement: advise/warn/refuse; force overrides (C-5) |
| train_lr_scales | (absent) | USER-SET (ADVANCED; no default value exists — absent = off = bitwise current behavior) | 59 R-2: per-region UPDATE multipliers {region: float}; regions: encoder/readout/block:k/loop (reference Network), layer:l/head:l:h/embed/out (attention hosts); union grammar validated loudly pre-mutation, grammar-valid non-live names inert (S2 propagation safety); s=0 accepted (C-5) with same-object bitwise stillness; axiom guidance: quasi-still (small nonzero) rather than zero; see GUIDE s14 |
| gate_widen_mode | "advise" | USER-SET | G-WIDEN advisory (58 D-5.4): fires when width_demand.met is True (no width demand) at grow; advise (default, silent) / warn / refuse; force overrides. TIER LAW (E6-1): direct compose.grow (L1) is the ungated expert surface — all gates live in the method tier |
| gate_aspect_min | 0.0 (OFF) | USER-SET (permissive default: nothing blocks until the user sets a floor; no derivation implied) | G-ASPECT (60D): minimum admissible aspect ratio aspect_after = width / depth_after at every deepen/insert_layer (reference Network: trunk H over 1+blocks+1; hosts: d_model over L+1; loop blocks are iteration, excluded from depth; >= ADMITS the boundary). LLM-scale reference: Kaplan-basin aspect ratios sit roughly ~40..200; Levine: useful depth grows ~log(width); small-scale constants are experiment-determined — values are yours |
| gate_aspect_mode | "advise" | USER-SET | G-ASPECT enforcement: advise (silent) / warn / refuse; force overrides (C-5); fires PRE-snapshot at the organ entry (no junk snapshot on refusal); mirrored in propose/plan_validate (plan_validate walks the plan's CUMULATIVE depth — shift-left) |
| aspect_auto | "widen_first" | USER-SET (INERT until gate_aspect_min > 0) | 60D D-7 AUTO-COMPLIANCE — the system itself never violates the floor and never hits the gate, no human monitoring: widen_first (default rule = the industry-mature width-first practice) widens by exactly the deficit k = ceil(floor x depth_after) - width BEFORE deepening (plans: ONE up-front pre-widen to the final stage count — every instant compliant a fortiori); defer skips the crossing deepen cleanly ("deferred_aspect"); off = gate-only manual flight. Hosts always defer with the boundary named (no d_model-widen operator exists). Responses always carry the automation's actions (auto_widened / deferred_aspect_steps — silent automation is banned, C-5); every auto-widen is ledgered with provenance aspect_auto; budgets are NEVER bypassed |
| aspect_auto_max_widen | 64 | USER-SET (safety bound) | per-event cap on automatic widening; a deficit above the cap refuses loudly BEFORE any mutation (clean state); params budgets always apply on top |

RECOMMENDED PRODUCTION PROFILE (60D S-c; proactive prevention —
production must not rely on hitting the gate): set a floor and
leave the automation on, e.g. `growth_params={"gate_aspect_min":
2.0, "gate_aspect_mode": "refuse", "aspect_auto": "widen_first"}`
— the gate is then the never-hit backstop, the shape law holds
by construction at every grown instant, and `assess` census
`aspect` / `depth_headroom` give drift-free observability
(headroom = int(width // floor) - stages; None while the gate
is off). Threshold VALUES remain the user's own parameters.

Trigger provenance: every growth/shrink ledger event carries
`trigger` = "caller" | "policy" (FR-12 mixed control audit);
rollback events are audit-only records (never gain events).

## growth_params write semantics (72B D-1, v1.7)
`set_policy(..., growth_params=G)` MERGES G one level deep:
given keys update/insert; a key with value None is REMOVED
(explicit deletion sentinel); `{}` changes nothing. Installed
governance keys therefore SURVIVE unrelated writes (e.g. an
E-6 protection window no longer discards an installed aspect
stack). The logged policy event carries the effective
post-merge dict. Budget note (72B D-2): every growth entrance
(facade verbs, plan_run, trial, propose/plan_validate via the
G-BUDGET row) judges the WHOLE act — primary move plus any
aspect-auto widen rider — against
initial_params x max_params_mult BEFORE any mutation.

## 7. Reward-learning keys (pointer)

The `growth_params` door also accepts the `rl.*` family
(19 keys) and `gate.eval_stream`, validated loudly by
`rl_trainer.defaults.validate_rl_policy`. Their defaults and
enums are documented in docs/RL_TRAINER_RUNBOOK.md.
