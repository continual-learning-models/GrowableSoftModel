# Quickstart (AI operator)

You are a general AI operating this small-model factory for a
user. Two channels; both are yours.

## Channel 1 — MCP (preferred)

    claude mcp add growable -- python3 -m mcp.mcp_server

(from the repo root). You then see 52 tools:
- the SOFTMODEL tools (create_model, teach, study, grow, commit,
  infer, evaluate, card, list_models, ...) — the tool's method:
  models that shape themselves, grow, and are promoted through a
  reality gate. Prefer these unless the user asks otherwise.
- the standard_* tools — the OPTIONAL industry-standard
  fixed-architecture mode (standard_create/train/evaluate/
  infer/save/load/list). Use when the user explicitly wants
  conventional methods.
Every response includes a `hint` with the natural next step and
a `path` telling you (and the user) where the model lives.

## Channel 2 — agent CLI

    python3 cli/cli.py help          # lists every verb + examples
    python3 cli/cli.py <verb> '<json-args>'

Same verbs, same responses; use when MCP isn't available.

## Operating rules of thumb

- Data rows: softmodel verbs (teach/study/add_holdout) use
  {"input": {...}, "target": ...} — "output" is accepted as an
  alias; standard_train uses {"x": [...], "y": ...}.
- Never invent a verb: `help` / tools-list is authoritative.
- Existing names are loaded, never overwritten (standard_*
  responses flag `existing`; softmodel returns the loaded
  model's data); pick a new name for a new model.
- Models are the user's property: quote the `path` from responses
  when the user asks where their model is.



## Tuning any method (S9)

Every parameter of every method family — network growth, SPU
self-processing, attention — is per-model controllable; the
complete catalog with worked examples is docs/PARAMETER_REFERENCE
.md. Quick forms (this lib = the ADVANCED direct surface; the
standard product gate is SoftModelSystem's train_converge/
model_policy):
- birth: create_model(policy={"substrate_params": {"d_model": 16,
  "seed": 7}}) — constructor knobs, per-model seed
- life: set_policy(model_id, spu_enabled=True, spu_K=8,
  att_lambda=0.1, growth_params={"grow_body_type": "attention"})
- verbs: grow(body_type=...), grow_attention(tol=...),
  set_attention_selfproc(on, heads=...)
Unknown keys refuse loudly, naming the key (all families,
both doors — the 104 name gate). growable_attention
serves numeric AND categorical data (S9.2).

## Growable attention (attention-build)

Create with `substrate="growable_attention"`; heads are born
small and grow on evidence. One governed growth event:

    grow_attention(model_id, layer=0)

(evidence -> suggestion -> held-out gate -> accept/refuse; refuses
on other substrates). Instruments are read-only; self-processing
is per-head switchable.

## Growth control surface (depthgrowth-v1.5: SERVED on every
## channel)

The whole surface is served as SYSTEM VERBS — MCP tools and
CLI verbs of the same names (59B): `deepen` (delta: adds a
processing stage — distinct from multi-scale growth, which
adds capacity in place), `remove_block`, `remove_grown`,
`propose`, `plan_validate`, `plan_run`, `trial`, `describe`,
`assess`, plus the legacy `widen`/`grow`/`loop`/version
verbs. `plan_run`/`trial` take training rows as `examples`;
warnings come back IN the response under `"warning"`.

    tools/call deepen {"model_id": "m", "m": 4}
    cli deepen '{"model_id": "m", "scope": [0,1]}'   # advisory
    tools/call plan_run {"model_id": "m",
                         "plan": "/path/rules.json"}

AUTO-GROWTH shape law (60D): set an aspect-ratio floor once
and the system keeps every grown instant compliant BY ITSELF —
no monitoring, no gate collisions:

    tools/call set_policy {"model_id": "m", "growth_params":
        {"gate_aspect_min": 2.0, "gate_aspect_mode": "refuse",
         "aspect_auto": "widen_first"}}
    # deepen/plan_run now widen-first automatically when a
    # deepen would cross the floor; responses report
    # "auto_widened" / "deferred_aspect_steps"; assess census
    # serves "aspect" and "depth_headroom". This is the
    # recommended production profile (PARAMETER_REFERENCE B3).

Full reference: docs/GROWTH_CONTROL_GUIDE.md (every example is
a verified test case; s0 lists the tier map). The raw
object-level summary below remains for facade-embedding
callers:

    net.deepen()                              # historical, bitwise
    net.deepen(position=p)                    # any seam, preserving
    net.deepen(recipe="copy_layer",           # full copy of a
        recipe_params={"source_index": i},    # designated layer
        zero_side="none", position=p)         # (non-preserving)
    net.deepen(scope=[...])                   # local/expert layer
    host.insert_layer(p)                      # attention hosts,
                                              # preserving at any p
    describe(net)                             # anatomy, permanent
                                              # ids (read-only)
    assess_growth(net)                        # dynamics (read-only)
    propose(net, move, gpp, **args)           # zero-cost dry-run —
                                              # EVERY move: deepen/
                                              # grow/removals/
                                              # insert_layer/
                                              # grow_site; cost in
                                              # params+mem+step-frac
    trial(net, fn, X, y, budget_steps=k)      # budgeted probe,
                                              # auto-rollback
    snapshot(net)/rollback(net, rec)          # bitwise history
    GrowthStore(root)                         # files-truth store +
                                              # SQLite index
    validate_plan/run_plan/load_plan          # rule-input automatic
                                              # mode (closed rule
                                              # set; parameter
                                              # files; ALL moves;
                                              # "force": true in a
                                              # step = C-5 in auto;
                                              # plan_adopted/
                                              # plan_halted audit
                                              # records with sha)

Control doctrine (C-5): every automatic mechanism has an off
switch; pause/abort works mid-plan and mid-trial; a direct
call always wins; force=True is accepted on EVERY entry point
(grow/deepen/loop/removals/insert_layer/grow_site) and
executes past any gate refusal, recorded as forced=True;
every event is ledgered with trigger provenance and cost
columns (step_at_event/steps_since_prev/wall_ms, FR-16).
Gate family (permissive USER-SET thresholds): G-ASPECT,
G-DEEPEN, G-SEAM, G-NEST, G-SCOPE, G-WIDEN — see the
GUIDE s11.
Hosts: grow_site/insert_layer carry the same governance
(auto-snapshot, four-spec records in host.growth_events);
host shrink = snapshot rollback (no removal operators).
