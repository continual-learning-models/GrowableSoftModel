"""One MCP server for the complete system (IWP4). Dependency-free stdio
JSON-RPC 2.0 (the phase servers' proven pattern). ~16 tools = the Model
Protocol. Contract in descriptions: teach/commit are gated (refusals are
outcomes); never advance a curriculum on a FALSE verdict; grow only on
verified plateaus and verify it pays before committing.

Run:      python3 -m mcp.mcp_server
Register: claude mcp add core -- python3 -m mcp.mcp_server
Env:      SOFTMODEL_MODELS_ROOT, SOFTMODEL_BACKEND (frozen Phase-1 config)
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "modules" / "ReferenceNet")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import sys

from core.facade import System

PROTOCOL_VERSION = "2024-11-05"

# Injected into the client model's context at initialize (MCP
# `instructions`): the operating manual for the AI driving the full
# system — lifecycle, growth decisions, and recovery.
SERVER_INSTRUCTIONS = """\
Every controllable parameter (policy keys, substrate_params,
growth_params, att_* keys) is cataloged in
docs/PARAMETER_REFERENCE.md. Innovation self-assessment (MDL,
docs/system/27): innovation_* policy keys; innovation_report tool.
\
You operate growable soft models: small neural models whose weights
AND structure keep learning for life. You supply language and
feature extraction; each model carries one domain's judgment.

LIFECYCLE (working state vs committed version):
- study trains the WORKING state only — nothing is served until
  commit promotes it. teach = study + commit in one call.
- commit gates against the model's OWN holdout: promoted only if it
  beats the live version. A refusal is a safe outcome, not an error.
- REQUIRED ORDER: create_model -> add_holdout (labeled reality kept
  separate from training) -> study/teach -> commit -> infer.
  Without holdout, commits cannot promote and infer stays untrained.
- reset discards the session; rollback re-points versions. Both are
  instant and safe.

CHOOSING A BODY (before create_model, for non-trivial data):
- list_substrates for the registered bodies; recommend_substrate
  with sample examples for a ranked, reasoned recommendation.
  Rule of thumb: weak-relation tables -> mlp; strong-relation
  tables -> transformer; timed signals -> sequence.

GROWTH PROTOCOL (structure is yours to grow, under evidence):
- Watch trajectory (verdict REAL|FALSE_SPIKE|FALSE_SWAP|STUCK) and
  growth_report (most-unstable sites). Never advance a curriculum
  on a FALSE verdict; on STUCK consider remediation or growth.
- grow adds inner networks at unstable nodes; widen (omega) adds
  units; add_feature (sigma) admits a new input mid-stream; refound
  (Phi) rebuilds from the experience store. All are exact-
  preserving in-session edits: evaluate that growth PAYS, then
  commit — the gate still decides.
- Budget refusals (max_params/max_depth from set_policy) are
  outcomes to respect, not errors to fight.
- Every parameter stays trainable for life; there is no freezing.

MAINTENANCE & SELF-STUDY:
- add_holdout as fresh ground truth arrives; check_drift
  periodically; on needs_reteach, teach with window/recent_n.
- run_self GRANTS a budgeted self-study session (store-only
  consolidation; optional allow_growth). It is consent-gated and
  fully logged; commit still gates any promotion. self_review is
  the read-only preview.
- discoveries returns mined IF/THEN regularities of this domain —
  knowledge you cannot have from pretraining; use and cite it.

RECOVERY: unknown model -> list_models/create_model. Repeated
non-promotion -> holdout missing/stale (add_holdout) or judge on
the recent slice (recent_n). untrained infer -> holdout+teach first.
"""

_ARR = {"type": "array"}
_STR = {"type": "string"}
_INT = {"type": "integer"}
_OBJ = {"type": "object"}
_NUM = {"type": "number"}


def _tool(name, desc, props, req):
    return {"name": name, "description": desc,
            "inputSchema": {"type": "object", "properties": props,
                            "required": req}}


TOOLS = [
    _tool("create_model", "Create a model (a name; optional description/"
          "holdout/policy). It shapes ITSELF from taught data: features, "
          "numeric-vs-categorical head (numeric_head='dist' adds "
          "the uncertainty head: answers carry value AND std "
          "on mlp/transformer/growable_attention), vocabulary, "
          "capacity — and later "
          "its STRUCTURE (multi-scale growth; deepening is a separate "
          "operation — see deepen). Optionally choose the "
          "substrate (body family) — see list_substrates; omitted -> "
          "transparent auto-default by detected data form. policy may "
          "carry substrate_params (JSON object): birth-time constructor "
          "knobs for the chosen substrate (e.g. d_model, n_layers, "
          "heads_spec, lr, seed — per-model reproducibility); unknown "
          "keys refuse loudly; d_in/hidden/mode/vocab are birth-derived "
          "and never accepted.",
          {"model_id": _STR, "description": _STR, "holdout": _ARR,
           "policy": _OBJ, "substrate": _STR}, ["model_id"]),
    _tool("study", "In-session supervised learning on the WORKING state "
          "(labeled examples). Not served until commit.",
          {"model_id": _STR, "examples": _ARR, "steps": _INT},
          ["model_id", "examples"]),
    _tool("attempts", "Practice phase 1: attempt 0 is the model's own "
          "answer, then seeded variants. YOU verify them.",
          {"model_id": _STR, "inputs": _ARR}, ["model_id", "inputs"]),
    _tool("practice_update", "Practice phase 2: per problem, a verified "
          "answer ONLY where attempt 0 failed (else null). Consolidation; "
          "cannot degrade the model.",
          {"model_id": _STR, "inputs": _ARR, "passed": _ARR},
          ["model_id", "inputs", "passed"]),
    _tool("teach", "One-call gated learning (study+commit convenience): "
          "trains a candidate version, promotes ONLY if it beats the live "
          "version on held-out reality. window/recent_n per drift rules.",
          {"model_id": _STR, "examples": _ARR, "window": _INT,
           "recent_n": _INT}, ["model_id", "examples"]),
    _tool("grow", "In-session STRUCTURAL edit: the most unstable nodes "
          "(any depth) grow inner networks, function-preserving. Budget "
          "refusals are outcomes. Verify it pays, then commit. "
          "body_type (string, optional): body family for grown inner "
          "bodies; omitted -> the model's growth policy decides.",
          {"model_id": _STR, "k_nodes": _INT, "hidden": _INT,
           "body_type": _STR},
          ["model_id"]),
    _tool("grow_attention", "One governed attention-growth event "
          "on a growable_attention model: evidence -> suggestion "
          "(widen head h / add head) -> held-out gate -> "
          "accept/refuse. The gate probe trains on the training "
          "lane; the holdout is scored only; refuses loudly if "
          "the training store is empty. Refuses on other "
          "substrates. tol (number, "
          "default 1.05) is the held-out gate tolerance.",
          {"model_id": _STR, "layer": _INT, "tol": _NUM},
          ["model_id"]),
    _tool("set_attention_selfproc", "Switch the entropy-band "
          "self-processing discipline on/off for a "
          "growable_attention model; heads (array of h or [l,h]) "
          "optionally restricts the allow-set. Refuses on other "
          "substrates. Warmup/age gates still apply.",
          {"model_id": _STR, "on": {"type": "boolean"},
           "heads": _ARR}, ["model_id", "on"]),
    _tool("commit", "Gate the working state against the incumbent on the "
          "model's OWN holdout; promote as a new version only if better.",
          {"model_id": _STR, "note": _STR}, ["model_id"]),
    _tool("reset", "Discard the session; working state returns to the "
          "committed version.", {"model_id": _STR}, ["model_id"]),
    _tool("rollback", "Point the model back to an earlier version.",
          {"model_id": _STR, "to": _STR}, ["model_id", "to"]),
    _tool("add_holdout", "Append fresh labeled reality to the holdout "
          "stream (the gate's anchor). Rows: {'input': {...}, 'target': ...}; 'output' is accepted as an alias for 'target'.",
          {"model_id": _STR, "examples": _ARR}, ["model_id", "examples"]),
    _tool("widen", "omega: outward growth — add units at any scale "
          "(exact function preservation; budget-checked).",
          {"model_id": _STR, "container": _STR, "k": _INT},
          ["model_id"]),
    _tool("add_feature", "sigma: admit a NEW input feature mid-stream "
          "(zero-weight entry, exact preservation; participation is "
          "earned by training).",
          {"model_id": _STR, "name": _STR,
           "default": {"type": "number"}},
          ["model_id", "name"]),
    _tool("refound", "Phi: rebuild the WORKING organ from the model's "
          "experience store (real data only); commit() gates "
          "promotion as usual.",
          {"model_id": _STR, "steps": _INT, "mode": _STR},
          ["model_id"]),
    _tool("self_review", "Read-only self-report: retention trends, "
          "learning progress, saturation, memory stats, question "
          "list. Changes nothing.",
          {"model_id": _STR, "probe_inputs": _ARR}, ["model_id"]),
    _tool("run_self", "GRANT a self-study session (consent C1-C6): "
          "budgeted blocks of store-only consolidation; optional "
          "growth proposals via allow_growth; fully logged; commit "
          "still gates promotion.",
          {"model_id": _STR, "block_budget": _INT,
           "allow_growth": {"type": "boolean"}},
          ["model_id", "block_budget"]),
    _tool("infer", "USE the model. Serves the COMMITTED version by "
          "default; working=true probes the session state.",
          {"model_id": _STR, "input": _OBJ, "working": {"type": "boolean"},
           "version": _STR}, ["model_id", "input"]),
    _tool("evaluate", "Score on caller suites [{name,X,y}] (appends to the "
          "score matrix; teacher-side measurement, never gates) or, "
          "without suites, on the model's holdout.",
          {"model_id": _STR, "suites": _ARR, "recent_n": _INT},
          ["model_id"]),
    _tool("trajectory", "S4 dashboard: progress/retention/volatility + "
          "verdict REAL|FALSE_SPIKE|FALSE_SWAP|STUCK. Never advance on "
          "FALSE; on STUCK consider remediation or growth.",
          {"model_id": _STR, "current_stage": _INT}, ["model_id"]),
    _tool("growth_report", "Ranked most-unstable nodes across all depths "
          "+ params/depth.", {"model_id": _STR}, ["model_id"]),
    _tool("check_drift", "Has reality moved? (recent holdout slice vs "
          "promotion baseline).", {"model_id": _STR, "recent_n": _INT},
          ["model_id"]),
    _tool("discoveries", "Readable mined regularities (categorical-shaped "
          "models).", {"model_id": _STR, "version": _STR}, ["model_id"]),
    _tool("innovation_report", "Read-only innovation self-assessment "
          "report (per-slice codelength/state, mastered band, config); "
          "enable via innovation_* policy keys.",
          {"model_id": _STR}, ["model_id"]),
    _tool("get_versions", "Version lineage with scores + active pointer.",
          {"model_id": _STR}, ["model_id"]),
    _tool("card", "Model card: learned shape, structure, policy, events.",
          {"model_id": _STR}, ["model_id"]),
    _tool("list_models", "The fleet.", {}, []),
    _tool("attribution", "Which suite each grown node serves (activation "
          "mass); fresh nodes reported inactive.",
          {"model_id": _STR, "suites": _ARR}, ["model_id", "suites"]),
    _tool("list_substrates", "The registered model bodies with machine-"
          "readable selection guidance (data form served, strengths, "
          "typical domains, cost, status). Read this to choose a "
          "substrate. Rule of thumb: weak-relation tables -> mlp; "
          "strong-relation tables -> transformer; timed signals -> "
          "sequence.", {}, []),
    _tool("recommend_substrate", "ADVISORY: give sample examples; the "
          "system detects the data form and probes relation strength, "
          "returning ranked substrate recommendations WITH reasons. "
          "You decide; this changes nothing.",
          {"sample_examples": _ARR}, ["sample_examples"]),
    _tool("set_policy", "Update a model's per-model policy knobs "
          "(growth budgets max_params_mult/max_depth, gate_recent_n, "
          "study_steps, consolidation_lr, ...). ROUTED: spu_* keys "
          "(e.g. spu_enabled, spu_K, spu_objective) are engine-"
          "validated and installed onto the organ at the next "
          "study/load — the one SPU on/off/tuning surface; "
          "growth_params (JSON object) carries per-model growth-"
          "policy overrides, including the aspect-ratio "
          "guardrail keys gate_aspect_min/gate_aspect_mode and "
          "the auto-compliance rule aspect_auto/"
          "aspect_auto_max_widen (60D: set a floor and the "
          "system widens first automatically — every grown "
          "instant stays shape-compliant); att_* keys (e.g. "
          "att_lambda, att_h_lo/att_h_hi) carry per-model "
          "attention-policy overrides for growable_attention "
          "models. Changes are logged events.",
          {"model_id": _STR, "updates": _OBJ}, ["model_id", "updates"]),
    _tool("run_course", "THE AUTOMATIC LOOP: give a graded curriculum "
          "[{name, examples, suite:{X,y}, target}] and a policy; the "
          "runner chains study -> evaluate -> trajectory -> (grow on "
          "STUCK) -> gated commit per stage, remediates bounded FALSE "
          "verdicts, and STOPS with a report on anything needing your "
          "judgment. Same gates as manual operation — no special powers.",
          {"model_id": _STR, "curriculum": _ARR, "policy": _OBJ},
          ["model_id", "curriculum"]),
    _tool("predict_dist", "The model's output DISTRIBUTION for "
          "one input (what infer truncates): numeric -> value "
          "(with numeric_head='dist' at create_model: value AND "
          "std — the uncertainty head, served on mlp/transformer/"
          "growable_attention); categorical/sequence -> labels "
          "+ probabilities. Use "
          "for sampling, uncertainty, generation. Example: "
          "{'model_id': 'm1', 'input_': {'a': 1.0}}.",
          {"model_id": _STR, "input_": _OBJ, "working":
           {"type": "boolean"}}, ["model_id", "input_"]),
    _tool("loop", "Grow a LOOP (lambda) on a scope: a governed "
          "directed cycle whose forward relaxes to a fixed point — "
          "iteration as a third computational resource beside "
          "widen/deepen, for laws that are implicit (no closed "
          "form; e.g. Kepler-type equations). Opt-in: requires "
          "loop_enabled=True in the growth policy; refused on "
          "unshaped hosts. Exact at application. NOTE: the model "
          "then ITERATES at inference on that scope (bounded by "
          "loop_K_max). Example: {'model_id': 'm1', 'container': "
          "'root'}.",
          {"model_id": _STR, "container": _STR, "m": _INT},
          ["model_id"]),
    _tool("remove_loop", "Remove a scope's loop block (audited; "
          "bitwise-restoring if the block was never trained).",
          {"model_id": _STR, "container": _STR}, ["model_id"]),
    # ---- growth-control verbs (59B; served depth/removal/
    # dry-run/plan/trial/inspection surface) ----
    _tool("deepen", "delta: add a PROCESSING STAGE. Multi-scale "
          "growth adds capacity in place; deepen adds a "
          "processing stage — two distinct axes, never the same "
          "thing. One verb for every organ kind: attention/"
          "transformer hosts insert a whole zero-born layer "
          "(position omitted = END = global deepening; scope "
          "refused loudly); network models append/insert a "
          "zero-born composition block (m units). scope (array "
          "of unit indices, advanced use) confines the block to "
          "a sub-scope and attaches the functional-closure "
          "advisory to the response as 'warning' — read it. "
          "Exact at application: the new stage contributes "
          "nothing until trained. force bypasses refusing GATES "
          "only, never params budgets. Verify it pays "
          "(trajectory/evaluate), then commit.",
          {"model_id": _STR, "m": _INT, "position": _INT,
           "recipe": _STR, "recipe_params": _OBJ,
           "zero_side": _STR, "scope": _ARR,
           "force": {"type": "boolean"}}, ["model_id"]),
    _tool("remove_block", "Remove composition block k (audited "
          "negative mirror of deepen; pre-event auto-snapshot; "
          "network models only — host shrink = version rollback "
          "via get_versions/rollback).",
          {"model_id": _STR, "k": _INT,
           "force": {"type": "boolean"}}, ["model_id", "k"]),
    _tool("remove_grown", "Remove the grown body at a site key "
          "(audited negative mirror of grow; pre-event auto-"
          "snapshot; network models only — host shrink = "
          "version rollback).",
          {"model_id": _STR,
           "key": {"description": "site key (int or string) "
                   "from grow's ledger/describe"},
           "force": {"type": "boolean"}}, ["model_id", "key"]),
    _tool("propose", "DRY-RUN a structural move (no mutation): "
          "gate verdicts, would_refuse, cost estimate (params/"
          "memory/step-share) and the operator-calculus verdict. "
          "move in {deepen, grow, remove_grown, remove_block, "
          "remove_loop, insert_layer, grow_site}; args mirror "
          "the real call. Propose before you spend.",
          {"model_id": _STR, "move": _STR, "args": _OBJ},
          ["model_id", "move"]),
    _tool("plan_validate", "Validate a growth plan WITHOUT "
          "running it: per-step dry-run proposals + cumulative "
          "cost + the plan's user limits. plan = inline JSON "
          "object or a plan-FILE path.",
          {"model_id": _STR,
           "plan": {"description": "plan object or file path"}},
          ["model_id"]),
    _tool("plan_run", "Execute a validated rule plan (automatic "
          "mode = YOUR rules in a file: steps of {move, args, "
          "when?}, limits {max_events, max_params}). Halts at "
          "any limit; every event is ledgered trigger='policy'; "
          "examples (labeled rows) train steps_between steps "
          "after each event — omitted = pure structural mode.",
          {"model_id": _STR,
           "plan": {"description": "plan object or file path"},
           "examples": _ARR, "steps_between": _INT},
          ["model_id"]),
    _tool("trial", "BOUNDED PROBE of one move (FR-15): snapshot "
          "-> apply (move,args) -> train at most budget_steps "
          "on examples -> measure -> UNCONDITIONAL rollback. "
          "Returns losses/realized_gain/wall_ms; the model is "
          "bit-restored. Applying for real afterwards is your "
          "separate decision (deepen/grow/plan_run).",
          {"model_id": _STR, "move": _STR, "args": _OBJ,
           "budget_steps": _INT, "examples": _ARR},
          ["model_id", "move", "examples"]),
    _tool("describe", "READ-ONLY anatomy: components (trunk/"
          "blocks/grown bodies/layers) with ids, shapes, "
          "positions, couplings, params_total. JSON-safe; "
          "changes nothing.",
          {"model_id": _STR}, ["model_id"]),
    _tool("assess", "READ-ONLY dynamics instrument panel: "
          "width_demand, seam_margins, event_gains, "
          "instability, census — the same estimators the gates "
          "consume. census carries the shape observables "
          "aspect (width/stages) and depth_headroom (how many "
          "more deepens fit under gate_aspect_min; null while "
          "that gate is off). Changes nothing.",
          {"model_id": _STR}, ["model_id"]),
    _tool("store", "READ-ONLY summary of the model's experience "
          "store (episodic memory): row count, capacity, "
          "quarantined-holdout count. The store feeds refound "
          "and run_self.",
          {"model_id": _STR}, ["model_id"]),
    # ---- the OPTIONAL industry-standard mode (secondary; the
    # tool's method is the softmodel above) ----
    _tool("standard_create", "OPTIONAL industry-standard mode: create a "
          "FIXED-architecture standard model (transformer or mlp) trained "
          "from scratch on your data — no growth, no self-processing "
          "(those are the softmodel tools above). Use when the user "
          "explicitly asks for conventional/standard methods. Example: "
          "{'name': 'demand', 'arch': 'mlp', 'hidden': 32}. Existing "
          "names are loaded, never overwritten.",
          {"name": _STR, "arch": _STR, "examples": _ARR,
           "mode": _STR, "n_layers": _INT, "d_model": _INT,
           "n_heads": _INT, "ffn_hidden": _INT, "hidden": _INT,
           "lr": {"type": "number"}}, ["name"]),
    _tool("standard_train", "Train a standard model with plain supervised "
          "steps. examples=[{'x': [...], 'y': value-or-label}, ...]; the "
          "input width shapes itself from the first batch. Reports "
          "before/after error, held-out score, and the model's path.",
          {"name": _STR, "examples": _ARR, "steps": _INT},
          ["name", "examples"]),
    _tool("standard_evaluate", "Score a standard model on labeled "
          "examples: mse+mae (numeric) or accuracy (categorical).",
          {"name": _STR, "examples": _ARR}, ["name", "examples"]),
    _tool("standard_infer", "USE a standard model: one input row x -> "
          "prediction (+confidence when categorical).",
          {"name": _STR, "x": _ARR}, ["name", "x"]),
    _tool("standard_save", "Persist the standard model's latest state "
          "(v1 keeps latest only — no version chain; gated versioning "
          "lives in the softmodel family).", {"name": _STR}, ["name"]),
    _tool("standard_load", "Restore a standard model from disk.",
          {"name": _STR}, ["name"]),
    _tool("standard_list", "List standard-family models only (softmodel "
          "models have their own list_models).", {}, []),
]


class MCPServer:
    def __init__(self, system: System | None = None):
        self.sys = system or System()

    def _dispatch(self, name, a):
        s = self.sys
        import standard_methods as std
        table = {
            # pass EVERY extra key through: the facade's loud
            # validation must see unknown parameters (charter)
            "predict_dist": lambda: s.predict_dist(
                a["model_id"], a["input_"],
                a.get("working", False)),
            "loop": lambda: s.loop(
                a["model_id"], a.get("container", "root"),
                a.get("m")),
            "remove_loop": lambda: s.remove_loop(
                a["model_id"], a.get("container", "root")),
            "deepen": lambda: s.deepen(
                a["model_id"], m=a.get("m"),
                position=a.get("position"),
                recipe=a.get("recipe"),
                recipe_params=a.get("recipe_params"),
                zero_side=a.get("zero_side"),
                scope=a.get("scope"),
                force=bool(a.get("force", False))),
            "remove_block": lambda: s.remove_block(
                a["model_id"], a["k"],
                force=bool(a.get("force", False))),
            "remove_grown": lambda: s.remove_grown(
                a["model_id"], a["key"],
                force=bool(a.get("force", False))),
            "propose": lambda: s.propose(
                a["model_id"], a["move"], a.get("args")),
            "plan_validate": lambda: s.plan_validate(
                a["model_id"], a["plan"]),
            "plan_run": lambda: s.plan_run(
                a["model_id"], a["plan"],
                examples=a.get("examples"),
                steps_between=int(a.get("steps_between", 10))),
            "trial": lambda: s.trial(
                a["model_id"], a["move"], a.get("args"),
                budget_steps=int(a.get("budget_steps", 6)),
                examples=a.get("examples")),
            "describe": lambda: s.describe(a["model_id"]),
            "assess": lambda: s.assess(a["model_id"]),
            "store": lambda: (lambda st: {
                "rows": len(st), "cap": st.cap,
                "quarantined": len(st._quarantine)})(
                    s.store(a["model_id"])),
            "standard_create": lambda: std.create(
                a["name"], a.get("arch"), a.get("examples"),
                a.get("mode", "numeric"),
                **{k: v for k, v in a.items()
                   if k not in ("name", "arch", "examples",
                                "mode")}),
            "standard_train": lambda: std.train(
                a["name"], a["examples"], a.get("steps", 200)),
            "standard_evaluate": lambda: std.evaluate(
                a["name"], a["examples"]),
            "standard_infer": lambda: std.infer(a["name"], a["x"]),
            "standard_save": lambda: std.save(a["name"]),
            "standard_load": lambda: std.load(a["name"]),
            "standard_list": lambda: std.list_models(),
            "create_model": lambda: s.create_model(
                a["model_id"], a.get("description", ""),
                a.get("holdout"), a.get("policy"), a.get("substrate")),
            "study": lambda: s.study(a["model_id"], a["examples"],
                                     a.get("steps")),
            "attempts": lambda: {"attempts": s.attempts(a["model_id"],
                                                        a["inputs"])},
            "practice_update": lambda: s.practice_update(
                a["model_id"], a["inputs"], a["passed"]),
            "teach": lambda: s.teach(a["model_id"], a["examples"],
                                     a.get("window"), a.get("recent_n")),
            "grow": lambda: s.grow(a["model_id"], a.get("k_nodes", 2),
                                   a.get("hidden", 16),
                                   a.get("body_type")),
            "commit": lambda: s.commit(a["model_id"], a.get("note", "")),
            "reset": lambda: s.reset(a["model_id"]),
            "rollback": lambda: s.rollback(a["model_id"], a["to"]),
            "self_review": lambda: s.self_review(
                a["model_id"], probe_inputs=a.get("probe_inputs")),
            "run_self": lambda: s.run_self(
                a["model_id"], int(a["block_budget"]),
                allow_growth=bool(a.get("allow_growth", False))),
            "widen": lambda: s.widen(a["model_id"],
                                     container=a.get("container", "root"),
                                     k=int(a.get("k", 2))),
            "add_feature": lambda: s.add_feature(a["model_id"], a["name"],
                                                 default=float(a.get("default", 0.0))),
            "refound": lambda: s.refound(a["model_id"],
                                         steps=int(a.get("steps", 4000)),
                                         mode=a.get("mode", "fresh")),
            "add_holdout": lambda: s.add_holdout(a["model_id"],
                                                 a["examples"]),
            "infer": lambda: s.infer(a["model_id"], a["input"],
                                     a.get("working", False),
                                     a.get("version")),
            "evaluate": lambda: s.evaluate(a["model_id"], a.get("suites"),
                                           a.get("recent_n")),
            "trajectory": lambda: s.trajectory(a["model_id"],
                                               a.get("current_stage")),
            "growth_report": lambda: s.growth_report(a["model_id"]),
            # S9.1: dispatch entry was MISSING since S8 (the tool
            # was defined and census-counted but uncallable via
            # tools/call — found while wiring tol; on the record)
            "grow_attention": lambda: s.grow_attention(
                a["model_id"], a.get("layer", 0),
                a.get("tol", 1.05)),
            "set_attention_selfproc": lambda:
                s.set_attention_selfproc(
                    a["model_id"], a["on"], a.get("heads")),
            "check_drift": lambda: s.check_drift(a["model_id"],
                                                 a.get("recent_n")),
            "discoveries": lambda: s.discoveries(a["model_id"],
                                                 a.get("version")),
            "innovation_report": lambda: s.innovation_report(
                a["model_id"]),
            "get_versions": lambda: s.get_versions(a["model_id"]),
            "card": lambda: s.card(a["model_id"]),
            "list_models": lambda: s.list_models(),
            "attribution": lambda: s.attribution(a["model_id"],
                                                 a["suites"]),
            "list_substrates": lambda: s.list_substrates(),
            "recommend_substrate": lambda: s.recommend_substrate(
                a["sample_examples"]),
            "set_policy": lambda: s.set_policy(a["model_id"],
                                               **a["updates"]),
            "run_course": lambda: s.run_course(a["model_id"],
                                               a["curriculum"],
                                               a.get("policy")),
        }
        if name not in table:
            raise ValueError(f"unknown tool: {name}")
        return table[name]()

    def handle(self, msg):
        method, mid = msg.get("method"), msg.get("id")
        if method is None:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32600, "message": "no method"}}
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "core",
                               "version": "2.0.0-dev"},
                "instructions": SERVER_INSTRUCTIONS}}
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": mid, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
        if method == "resources/list":                    # graceful: none
            return {"jsonrpc": "2.0", "id": mid, "result": {"resources": []}}
        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": mid, "result": {"prompts": []}}
        if method == "tools/call":
            try:
                out = self._dispatch(msg["params"]["name"],
                                     msg["params"].get("arguments", {}))
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": json.dumps(out)}],
                    "isError": False}}
            except Exception as exc:              # noqa: BLE001
                return {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"error: {exc}"}],
                    "isError": True}}
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"unknown: {method}"}}


def main():
    server = MCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = server.handle(json.loads(line))
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
