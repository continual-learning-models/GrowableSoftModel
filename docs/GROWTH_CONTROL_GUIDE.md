# Growth Control Guide

Complete interface reference for the growth-control surface
(one-shot batch, docs 51–56; tags adj-p0..os-b4). Audience:
human operators AND AI callers. EVERY example below is a
VERIFIED test case — the owning test box is cited; nothing
here is undemonstrated usage.

Principles in force: complete human controllability (C-5 —
every automatic mechanism has an off switch, a runtime
interrupt and manual-override precedence; assessments and
gates are ADVISORY toward a human command); total plasticity
(nothing is ever frozen); provenance (every growth/shrink
event carries its four-spec birth certificate and its trigger).

## 0. SERVED TIERS (59B/59C): who reaches this surface

Everything in this guide is reachable at every SERVED tier —
raw library functions are a developer surface, never the
user's (59B s0). One verb vocabulary everywhere:

| tier | reach | example (each a verified box) |
|---|---|---|
| facade `System` | `deepen, remove_block, remove_grown, propose, plan_validate, plan_run, trial, describe, assess` | `s.deepen("m", m=4)` [F-1, test_facade_growth_verbs.py] |
| lib CLI | same verbs, auto-dispatched JSON | `cli deepen '{"model_id":"m","m":4}'` [C-1, test_cli_growth_e2e.py] |
| lib MCP | tools of the same names (+ `store` summary) | tools/call deepen [M-2, test_mcp_growth_tools.py] |
| SMS product | ordinary: `deepen` (no scope/force), `propose/plan_validate/plan_run/trial/describe/assess`, `widen/grow/commit/get_versions/rollback`; ADVANCED: `deepen_scoped/remove_block/remove_grown` (+`force`), generic `advanced(verb, args)` | `op.deepen("m")` [S-1, SMS tests/growth] |

Notes binding on every served text: multi-scale growth adds
capacity in place; **deepen adds a processing stage** — two
distinct axes (M-5 terminology sweep). Warnings raised by an
operation are RETURNED in the response under `"warning"`
(CLI also re-prints them to stderr). Product-tier structural
undo = the version system (`commit`/`get_versions`/
`rollback`); the in-session snapshot ring below is a
lib-session instrument. Facade `plan_run`/`trial` take
training rows as `examples`; the SMS mirrors take `snapshot`
and resolve it to rows at the product door.

## 1. Deepening — `Network.deepen(...)`

    deepen(m=None, position=None, recipe=None,
           recipe_params=None, scope=None, zero_side=None,
           force=False)

| parameter | meaning | default |
|---|---|---|
| m | new layer width | host width H |
| position | seam index 0..len(blocks); "end" appends | end |
| recipe | birth values: "random", "zero", "copy_layer", "interleave_neighbors", or a registered custom name | random |
| recipe_params | e.g. {"source_index": 1} for copy_layer | {} |
| scope | index set: the layer lives on this subset at birth (out-of-scope read cols / write rows born exactly zero) | ALL |
| zero_side | "Bout" = function-preserving birth; "none" = full-value birth (function changes, your declared choice) | Bout |
| force | C-5 unconditional override past any gate refusal | False |

The DEFAULT call is the historical operator, bit-identical.
[boxes: test_layer_preset_b1.py — every-seam preservation,
recipes, full copy, scope; gate modes in
test_control_surface_b3.py]

Examples (verified):

    net.deepen()                       # historical delta
    net.deepen(position=1)             # insert at seam 1,
                                       # function preserved
                                       # BITWISE
    net.deepen(position=2,             # owner's case: insert
        recipe="copy_layer",           # between blocks 2 and 3
        recipe_params={"source_index": 1},
        zero_side="none")              # COMPLETE copy of block
                                       # 1 — non-preserving by
                                       # declared choice
    net.deepen(scope=[0, 2, 5])        # local layer on an
                                       # arbitrary subset

Custom initial values (C-5: values are yours):

    from reference_net.foundation.recipes import register_recipe
    register_recipe("mine", lambda shapes, rng, ctx:
                    {"Bin": my_array})
    net.deepen(recipe="mine")

## 2. Attention hosts — `insert_layer(...)`

    host.insert_layer(position, recipe="random",
                      source_index=None, zero_side="default")

GrowableAttentionSubstrate and TransformerSubstrate. Default
birth: LN identity + zero attention out-projections + zero FFN
second matrix -> the inserted layer contributes EXACTLY
nothing (bitwise) at ANY position. recipe="copy_layer" +
zero_side="none" = complete SOLAR-style copy (non-preserving,
declared). Per-layer keys renumber internally; grown FFN port
sites keep identity (below the seam unchanged, above shifted).
Head methodology is untouched. Events in host.growth_events.
[boxes: test_attn_insertion_b2.py]

## 3. Widening (existing move, unchanged)

`net.grow(j, hidden=...)` and `host.grow_site(path, hidden=)`
— the rho/site operators, bit-identical to growthport-v1;
now preset-backed with ledgered specs.
[boxes: P0 recapture; test_foundation_compose.py]

## 4. Reading the structure — `describe(host)`  (FR-19)

    from reference_net.instrument import describe
    rep = describe(net)      # read-only, JSON-serializable

Returns components (trunk/blocks/loop/attention layers) with
PERMANENT ids (kind#seed — never renumbered), current
positions, shapes, params; and every coupling with its key,
A-shape and spans. ADDRESSABILITY CLOSURE: everything the
report shows can be designated in commands (copy sources,
seams, removal keys, scopes) — see it, address it.
[box: test_control_surface_b3.py describe+closure]

## 5. Assessing the dynamics — `assess_growth(host)` (FR-11)

    from reference_net.method.gates import assess_growth
    a = assess_growth(net)

Fields: width_demand (gradient-span estimate vs width — the
width-first law's input), seam_margins (junction widths, the
T5 bandwidth law), event_gains (realized gain per ledgered
event), instability, census. Read-only; the GATES consume the
same estimators (what you see is what they judge).
[box: test_control_surface_b3.py assess]

## 6. Dry-run — `propose(host, move, gpp, **args)` (FR-14)

Zero-cost pre-filter: gate verdicts, cost estimate and the
operator's mathematical verdict for a candidate move — with
NO mutation; a would_refuse verdict matches what the real
call would do. EVERY move has a dry-run form (FR-12
symmetry): "deepen", "grow", "remove_grown", "remove_block",
"remove_loop", "insert_layer", "grow_site". Cost comes in
three components on every report (FR-14): cost_params,
cost_mem_bytes (= cost_params x itemsize x 3: parameter +
Adam m + Adam v), cost_step_frac (= cost_params / n_params);
the formulas ride in the report (cost_formulas). Removal
proposes carry the NEGATIVE mirror of the grow accounting.
Hosts without an operator refuse loudly (e.g. remove_block
on an attention host: "host shrink = snapshot rollback").
[boxes: t2/t3 propose tests]

## 7. Bounded trial — `trial(...)` (FR-15)

    from reference_net.growth_store import trial
    rep = trial(net, lambda h: h.deepen(), X, y,
                budget_steps=6)

snapshot -> apply -> train <= budget -> measure ->
UNCONDITIONAL rollback. Report: losses, realized_gain,
steps_run. The model is bit-identical to before the trial;
applying for real is your separate decision. Abortable
mid-run. [boxes: trial tests]

## 8. Snapshots & rollback (FR-13/FR-18)

Automatic: every growth/removal event is preceded by a
snapshot (policy key growth_auto_snapshot, default ON;
retention growth_snapshot_keep, default 8 — rollback is
guaranteed within this FINITE window). Manual:

    from reference_net.growth_store import snapshot, rollback
    rec = snapshot(net, tag="before-experiment")
    ...                        # deepen, train, change your mind
    rollback(net, rec)         # EXACT bitwise restore (params,
                               # optimizer, counters); next move
                               # is YOURS — nothing follows
                               # automatically

Disk persistence beyond the window:

    from reference_net.growth_store import GrowthStore
    store = GrowthStore("growth_store/model-A")
    store.save_snapshot(rec)   # write-once file (truth)
    rec2 = store.load_snapshot(rec["id"])

[boxes: test_growth_store_b3.py]

## 9. Removal (shrink, FR-18)

`net.remove_block(k)`, `net.remove_grown(j)`, `remove_loop()`
stand under the same governance: pre-event snapshot,
reversible within the window, ledgered with the NEGATIVE
parameter mirror. The 3-D attention hosts have NO removal
operators — their shrink path IS the finite-window snapshot
rollback (FR-18); plan/propose removal moves on a host
refuse loudly. [boxes: removal test, t10 host parity]

## 10. Plans — rule-input automatic mode (FR-12/FR-17)

Automatic mode is YOUR control expressed as RULES. CLOSED
vocabulary this batch: rule types {"schedule", "threshold",
"limit"}; limits {max_events, max_params, stop}; conditions on
{params, blocks}; moves {"deepen", "grow", "remove_grown",
"remove_block", "remove_loop", "insert_layer", "grow_site"}.
Unknown types are refused loudly. Step args pass VERBATIM to
the host method, so a step may carry "force": true — the
C-5 override works inside automatic mode too [box: t7
force-in-plan]. max_params compares the LIVE n_params each
step, so a removal re-opens headroom [box: t7 mixed plan].

    plan = {"steps": [
              {"rule": "schedule", "move": "deepen",
               "args": {"position": 0}},
              {"rule": "threshold", "move": "deepen",
               "when": {"metric": "blocks", "op": "<",
                        "value": 5}, "args": {}}],
            "limits": {"max_events": 2}}
    from reference_net.method.gates import (validate_plan,
        run_plan, load_plan)
    validate_plan(net, plan, gpp)   # per-step dry-run BEFORE
                                    # any compute
    run_plan(net, plan, gpp, X, y, steps_between=10,
             control=ctl)           # ctl = {"pause":..,
                                    #        "abort":..} (C-5)

Plans load from PARAMETER FILES (load_plan(path)) —
file-loaded == programmatic. Plan events carry
trigger="policy"; your direct calls carry trigger="caller";
they interleave freely and your call always wins.
PROVENANCE (FR-17): run_plan writes audit-only plan_adopted
(before the first event) and plan_halted (exact reason +
events_run) records, both carrying the plan's sha — the
same sha GrowthStore.save_plan computes, so the executed
plan is reconstructible from the record alone [box: t4].
[boxes: plan tests, mixed-provenance test]

## 11. Gates & enforcement (FR-12)

Admission gates realize the width-first law (G-DEEPEN input =
width_demand) and the junction-bandwidth facts (seam_margins).
The FULL FAMILY (one uniform mechanism — estimator vs
USER-set threshold, modes advise/warn/refuse, force
override):

  G-DEEPEN  width-first admission     gate_deepen_mode
  G-SEAM    new-event junction width  gate_seam_min_width /
            (grow, grow_site)         gate_seam_mode
  G-NEST    path bottleneck when      gate_seam_min_width /
            growing INSIDE a grown    gate_nest_mode
            body (OD-3)
  G-SCOPE   |scope| at scoped deepen  gate_scope_min_width /
            (global deepen ungated)   gate_scope_mode
  G-WIDEN   advisory when there is    gate_widen_mode
            NO width demand at grow
  G-ASPECT  aspect ratio (width /     gate_aspect_min /
            depth_after) at every     gate_aspect_mode
            deepen & insert_layer
            (60D; >= admits; loop
            excluded from depth;
            scoped deepen judged on
            the GLOBAL shape)

G-ASPECT runnable example (= the verified T-3/T-5 boxes,
tests/unit/test_aspect_ratio_gate.py; mlp trunk H=16):

    s.set_policy("m", growth_params={
        "gate_aspect_min": 6.0, "gate_aspect_mode": "refuse",
        "aspect_auto": "off"})
    s.deepen("m", m=4)   # blocks=1 -> depth_after=3:
    # {"refusal": "G-ASPECT: aspect 5.33 = width 16 / depth 3
    #   < gate_aspect_min 6.0 (width-first; see
    #   PARAMETER_REFERENCE guidance)"}
    s.propose("m", "deepen", {"m": 4})
    # ...["gates"]["G-ASPECT"] -> {"aspect_after": 5.333...,
    #   "floor": 6.0, "met": False, ...}, would_refuse True

All thresholds are USER PARAMETERS (policy keys / parameter
files; PERMISSIVE defaults — nothing blocks until you set
values; how you derive them is your business, outside the
system). force=True executes past any refuse (C-5) and the
event records forced=True; assessments never block a human
command. TIER LAW: direct compose.grow (L1) is the ungated
expert surface — gates live in the method tier only.
[boxes: t5 gate-family tests; test_aspect_ratio_gate.py]

### 11b. AUTO-GROWTH (60D D-7): set the floor, the system
### self-complies

With `aspect_auto` (default "widen_first", inert until a floor
is set) the system NEVER meets its own gate — autonomous
growth is shape-correct at every instant, with no human
monitoring in the loop:

    s.set_policy("m", growth_params={
        "gate_aspect_min": 6.0, "gate_aspect_mode": "refuse",
        "aspect_auto": "widen_first"})
    s.deepen("m", m=4)
    # -> widens by exactly the deficit k = ceil(6x3) - 16 = 2
    #    FIRST (ledgered with provenance aspect_auto), then
    #    deepens; response carries {"auto_widened": 2}; final
    #    aspect 18/3 = 6.0 >= floor. Serve value-exact across
    #    the act (zero-extension algebra).
    s.plan_run("m", {"steps": [...deepen steps...]})
    # -> ONE up-front pre-widen to the plan's FINAL stage
    #    count (every intermediate instant compliant a
    #    fortiori), then the plan runs with ZERO refusals.

Reading headroom (how many more deepens fit under the floor):

    s.assess("m")["census"]          # {"aspect": ...,
    # "depth_headroom": int(width // floor) - stages, ...}
    # headroom is None while the gate is off

Controllability (C-5): `aspect_auto="off"` restores gate-only
manual flight; `"defer"` skips crossing deepens cleanly
("deferred_aspect"); hosts always defer with the boundary
named (no d_model-widen operator exists yet); direct manual
verbs are never blocked or re-attributed; params budgets are
NEVER bypassed — an auto-widen that cannot fit refuses loudly
and nothing mutates. RECOMMENDED PRODUCTION PROFILE:
PARAMETER_REFERENCE B3 (floor + refuse + widen_first — the
gate becomes the never-hit backstop).
[boxes: T-14..T-18, tests/unit/test_aspect_ratio_gate.py]

## 12. Monitoring (FR-16)

    from reference_net.instrument import (monitor_configure,
        monitor_export)
    monitor_configure(net, cadence=50, window=256)
    ...train...
    monitor_export(net)      # JSON time series of assessments

Off by default; never serialized into artifacts; zero float
ops when unarmed. [box: monitor test]

## 13. The event ledger

Every growth/shrink event records its four spec objects
verbatim ("specs": structure/wiring/placement/birth — the
birth certificate), its trigger ("caller"/"policy"), the
gain machinery's fields, and the COST COLUMNS (FR-16):
step_at_event (training-step counter at the event),
steps_since_prev (training steps attributed to the preceding
configuration — deterministic), wall_ms (this operation's
own duration — run-varying by nature; the ledger sits
outside all bit gates). trial() reports carry wall_ms too.
On the 3-D hosts the same records (incl. grow_site's four-
spec certificate) live in host.growth_events. forced=True
appears when and only when an event was forced (C-5).
Rollbacks are audit-only records (never gain events). `GrowthStore.append_event/query_events`
give filtered history; `reindex()` rebuilds the SQLite index
from the files at any time (files are truth).

## 14. New-layer protection window (per-region rates) — ADVANCED

Optional advanced feature: nothing here is needed for normal
use; with no key set the system behaves exactly as before.

After deepening, you may let the NEW layer train full-speed
while the rest of the network stays quasi-still, then release:

    net.deepen()                       # new block at index k
    net._growth_policy = dict(net._growth_policy or
        DEFAULT_GROWTH_POLICY, train_lr_scales={
            "encoder": 0.01,           # input features ~still
            "block:0": 0.01,           # ... every OLD block
            "loop": 0.01})             # (readout stays full:
                                       #  industry norm)
    ...train the window (steps chosen by you)...
    # release: ramp the values back toward 1 over some steps
    # (soft-continuity), then delete the key entirely.

Region names: reference Network "encoder" (W1,b1), "readout"
(W2,c), "block:k", "loop"; attention hosts "layer:l" (all
per-layer tensors incl. in-P attention), "head:l:h" (GA),
"embed" (Bf,Wv), "out" (Wh,bh). Typos are refused loudly
BEFORE any state changes; grammar-valid names that don't
exist on an instance are inert (safe: the policy dict
propagates into grown bodies).

Semantics and facts you should know:
 - The scale multiplies the UPDATE (exactly lr x s). It is
   applied at write-back; kernels and optimizer state are
   untouched.
 - s = 0 is accepted (your control, C-5) and keeps parameters
   bitwise still. THEORY GUIDANCE (total-plasticity axiom):
   prefer quasi-still (e.g. 0.01) over exact zero.
 - Adam's moments KEEP ACCUMULATING while a region is slowed:
   on release there is a mild kick that decays in ~10 steps.
 - Instability instruments (update EMAs) read artificially
   calm for slowed regions during the window.
 - Floating point: below ~1e-16 relative, a tiny s IS zero
   bitwise (absorption threshold).
 - Structural events (insert_layer) renumber layers: rewrite
   your table after them (no silent remapping).
 - Grown bodies read their OWN policy reference: mutating a
   shared dict reaches them; rebinding the host's policy does
   not (set body._growth_policy for per-body schedules).
 - Whether a protection window HELPS is scenario-dependent:
   zero-birth insertion needs no protection for stability
   (there is no shock); the window's purpose is giving new
   capacity a chance to capture function. Judge with
   trial() + the gain ledger. [boxes: T-B1..B11]

## Growth preference (learning strategy layer; plan 84, doc 83)

TOOL FIT: use when a model's growth decisions should LEARN from
its own ledger history (which kinds of growth actually paid) —
the strategy layer stops being memory-less. NOT a gate change:
preferences only reorder proposals; adoption stays the gate's.
INDICATIONS: long-lived organs with recurring growth decisions;
fleets (fold a prior from siblings' ledgers). CONTRA: one-shot
short lives (nothing to learn from; min_count floors it inert);
anything reward-driven at the TRIGGER level (forbidden — triggers
read instruments only, permanent law).
KEYS: PARAMETER_REFERENCE.md section 5b (23 preference.* keys;
default rule=fixed is provably inert — TI-02 byte-identity).
OPS: docs/PREFERENCE_RUNBOOK.md (enable/inspect/reset, fleet
prior, rollback semantics, audit-volume modes, failure modes).
VERBS: preference_inspect / preference_reset (SMS mirrors);
prior_fold offline tool. [boxes: TB-01..16, TP-01..05,
TI-01..07, TS-01..06; batteries B-1..B-4]
