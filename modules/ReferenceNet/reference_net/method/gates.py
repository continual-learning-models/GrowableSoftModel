"""GATES (next-dev-docs/52 v1.7 section 6.2) — L2 admission
guards, invoked by preset orchestration at the DECLARED stage
seams (SR-13); they never enter foundation/. Policy VALUES
arrive as arguments from the host tier (V-A0-2: this module
imports foundation/numpy/stdlib only).

A5 content: the scale-hierarchy guard — VERBATIM code motion
of the historical grow-body guard (via Network._scale_guard,
A2). The math-derived admission gates (G-WIDEN / G-DEEPEN /
G-SEAM, math-research/03) land at stage B3 with the
assessment interface.
"""


def scale_hierarchy_guard(host, body, j, gp2):
    """Owner ruling: the host's OWN params must be >=
    grow_min_host_ratio x body params — the body is a
    fine-scale correction, subordinate to its host. Fires at
    the post-build / pre-couple seam; in refuse mode it raises
    AFTER the seed increment (the P0-captured historical
    ordering, preserved as-is)."""
    nel = host._bk.numel
    own = (nel(host.W1) + nel(host.b1) + nel(host.W2)
           + nel(host.c)
           + sum(nel(b["Bin"]) + nel(b["bb"]) + nel(b["Bout"])
                 for b in host.blocks))
    if host.loop_block is not None:
        own += sum(nel(v) for v in host.loop_block.values())
    ratio = own / max(body.n_params(), 1)
    min_ratio = float(gp2.get("grow_min_host_ratio", 100.0))
    min_params = int(gp2.get("grow_min_host_params", 100000))
    min_steps = int(gp2.get("grow_min_host_steps", 100))
    problems = []
    if host._step_count < min_steps:
        problems.append(
            f"host trained only {host._step_count} steps < "
            f"{min_steps} (grow_min_host_steps): topology "
            f"changes must come after the base has taken "
            f"shape, never from the start")
    if own < min_params:
        problems.append(f"host own params {own} < absolute "
                        f"floor {min_params} "
                        f"(grow_min_host_params)")
    if ratio < min_ratio:
        problems.append(f"host/body ratio {ratio:.2f}x < "
                        f"{min_ratio}x (grow_min_host_ratio)")
    if problems:
        msg = "scale-hierarchy violation: " + "; ".join(problems)
        if gp2.get("grow_scale_guard", "warn") == "refuse":
            raise ValueError(msg)
        import warnings
        warnings.warn(msg + " (recorded)", stacklevel=3)
        # violations are NOT gain events — they get their own
        # ledger (the gain machinery must never adjudicate
        # them; first guard version polluted _pending_gain and
        # broke the instrumentation tests — corrected)
        if not hasattr(host, "_scale_events"):
            host._scale_events = []
        host._scale_events.append(
            {"site": j, "ratio": round(ratio, 2), "own": own,
             "body_params": body.n_params(),
             "step": host._step_count, "problems": problems})
# ---- B3 extension appended to method/gates.py ----
# (doc 55 v1.11 s3-B3/B6: math-derived admission gates,
#  the read-only assessment/proposal interfaces, and the
#  LIMITED-rule-set plans — closed vocabulary, parameter-file
#  loadable. Policy VALUES arrive as arguments; enforcement
#  strength per gate is a policy key in {advise, warn, refuse}
#  — FR-12 full authority; C-5: everything switchable,
#  interruptible, override-able.)
import hashlib as _hashlib
import json as _json

import numpy as _np


# ---------------- estimators (math-research/03) ----------------

def width_demand(host, tol=1e-6):
    """T9 gradient-span floor: the numerical rank of the
    host's gradient-EMA matrix estimates the active-subspace
    dimension r_g; width >= r_g is the width-first
    precondition (G-WIDEN/G-DEEPEN inputs). EMPIRICAL
    tolerance (PR-4: calibration pending)."""
    width = int(getattr(host, "H", getattr(host, "d", 0)))
    ema = getattr(host, "_ema_dw", None)
    if ema is None or not hasattr(ema, "shape"):
        return {"r_hat": None, "width": width,
                "met": None,
                "note": "no matrix gradient-EMA instrument "
                        "on this host kind"}
    M = _np.asarray(host._bk.to_numpy(ema)) \
        if hasattr(host, "_bk") else _np.asarray(ema)
    s = _np.linalg.svd(M, compute_uv=False)
    r = int((s > tol * (s[0] if s.size else 1.0)).sum())
    return {"r_hat": r, "width": width,
            "met": bool(width >= r)}


def seam_margins(host):
    """T5 per-junction bandwidth: for every grown coupling,
    its junction width (rows of A = |span_in| or body
    out_width). The rank cap of anything routed through the
    junction is bounded by this width (math-research/02)."""
    out = []
    site = getattr(host, "_port_site", None)
    sites = ([("root", site)] if site is not None else []) + \
        sorted(getattr(host, "_port_sites", {}).items())
    for where, s in sites:
        for c in getattr(s, "bodies", []):
            A = c["A"]
            out.append({"site": str(where),
                        "key": str(c.get("key")),
                        "junction_width": int(
                            _np.asarray(A).shape[0]),
                        "target_width": int(
                            _np.asarray(A).shape[1])})
    return out


def event_gains(host):
    """Realized per-event gains from the gain ledger (resolved
    entries only) — 'the effect is fading' is visible here."""
    return [{"event": r.get("event"), "site": str(r.get("site")),
             "gain": r.get("gain"),
             "params_added": r.get("params_added")}
            for r in getattr(host, "gain_ledger", [])]


# ---------------- assessment (FR-11, read-only) ----------------

def assess_growth(host, scope=None):
    """The instrument panel: DYNAMICS for control decisions.
    Same estimator functions the gates consume (single
    source); never mutates state (boxed)."""
    inst = None
    if hasattr(host, "instability"):
        try:
            inst = [float(x) for x in host.instability()]
        except Exception:
            inst = None
    # 60D D-6: aspect + headroom (same per-family definitions
    # as the gate, CURRENT stages — pre-op); headroom only
    # exists once the user set a floor, else None
    gpp_a = getattr(host, "_growth_policy", {}) or {}
    width_a, stages_a = _aspect_shape(host)
    floor_a = float(gpp_a.get("gate_aspect_min", 0.0))
    headroom = (max(0, int(width_a // floor_a) - stages_a)
                if floor_a > 0 else None)
    return {"width_demand": width_demand(host),
            "seam_margins": seam_margins(host),
            "event_gains": event_gains(host),
            "instability": inst,
            "census": {"params": int(host.n_params()),
                       "blocks": len(getattr(host, "blocks",
                                             [])),
                       "depth": int(host.layer_depth())
                       if hasattr(host, "layer_depth")
                       else None,
                       "aspect": round(
                           width_a / max(stages_a, 1), 4),
                       "depth_headroom": headroom}}


# ---------------- proposal dry-run (FR-14) ----------------

def _widen_added_params(host, k):
    """72B D-2: the ONE pricing home for the aspect-auto widen
    rider (identical to the facade F-4 formula)."""
    added = k * (host.d_in + 2)
    for b in getattr(host, "blocks", []):
        added += 2 * int(_np.asarray(b["bb"]).size) * k
    lb = getattr(host, "loop_block", None)
    if lb is not None:
        added += 2 * int(_np.asarray(lb["b_l"]).size) * k
    return added


def _budget_row(host, gpp, act_params):
    """72B D-2 G-BUDGET row; None when the cap is absent
    ([RV F7] graceful absence — expert-tier direct calls
    carry no injected cap and see no row)."""
    cap = gpp.get("params_budget_cap")
    if cap is None:
        return None
    post = int(host.n_params()) + int(act_params)
    return {"met": post <= int(cap), "cap": int(cap),
            "post": post}


def propose(host, move, gpp, **args):
    """Pre-spend filter: what WOULD happen — gate verdicts,
    cost estimate (params + derived memory/step-share, 58
    D-2), operator verdict citation — with NO mutation (boxed
    by state-hash). move in {"deepen", "grow", "remove_grown",
    "remove_block", "remove_loop", "insert_layer",
    "grow_site"} (D-3; FR-12 symmetry: every move has a
    dry-run form)."""
    rep = {"move": move, "args": dict(args), "gates": {},
           "would_refuse": False}
    if move == "deepen":
        m = int(args.get("m") or host.H)
        rep["cost_params"] = m * host.H + m + host.H * m
        if args.get("scope") is not None:      # 59 R-1.3
            from .presets import _SCOPE_ADVISORY
            rep["advisory"] = _SCOPE_ADVISORY
        wd = width_demand(host)
        rep["gates"]["G-DEEPEN(width-first)"] = wd
        mode = gpp.get("gate_deepen_mode", "advise")
        rep["gates"]["mode"] = mode
        if wd["met"] is False and mode == "refuse":
            rep["would_refuse"] = True
        ga, ga_refuse = aspect_report(host, gpp)   # 60D mirror
        rep["gates"]["G-ASPECT"] = ga
        if ga_refuse and not args.get("force"):
            rep["would_refuse"] = True
        rep["verdict"] = ("delta form: degree exponent +1, "
                          "rank cap unchanged "
                          "(math-research/02 T8)")
    elif move == "grow":
        hidden = int(args.get("hidden", 16))
        out_w = int(gpp.get("grow_body_out_width", 1))
        rep["cost_params"] = (host.d_in * hidden + hidden
                              + hidden * out_w + out_w
                              + out_w * host.H)
        j = args.get("j")
        dup = j in getattr(host, "_port_js", set()) \
            or j in getattr(host, "inner", {})
        rep["gates"]["duplicate-site"] = {"met": not dup}
        if dup:
            rep["would_refuse"] = True
        rep["verdict"] = ("rho form: degree exponent "
                          "unchanged, rank cap +k "
                          "(math-research/02 T8)")
    elif move in ("remove_grown", "remove_block",
                  "remove_loop"):
        # 58 D-3.1-3.3: negative mirrors of the growth
        # accounting, from the SAME sources the removal
        # carriers write; hosts without removal operators
        # refuse LOUDLY (their shrink path is rollback).
        if not hasattr(host, move):
            raise ValueError(
                f"host {type(host).__name__} has no {move}; "
                "host shrink = snapshot rollback (FR-18)")
        if move == "remove_grown":
            key = args.get("key", args.get("j"))
            port = getattr(host, "_port_site", None)
            # the SLOT (Coupling) carries the full accounting
            # A + body — the R11 removal mirror's own source
            slot = next((s for s in port.bodies
                         if s.get("key") == key), None) \
                if port is not None else None
            if slot is not None:
                rep["cost_params"] = -int(slot.n_params())
            elif key in getattr(host, "inner", {}):
                rep["cost_params"] = -int(
                    host.inner[key].n_params())
            else:
                rep["would_refuse"] = True
                rep["cost_params"] = 0
        elif move == "remove_block":
            k = args.get("k")
            blocks = getattr(host, "blocks", [])
            if isinstance(k, int) and 0 <= k < len(blocks):
                m = blocks[k]["bb"].size
                rep["cost_params"] = -(m * host.H + m
                                       + host.H * m)
            else:
                rep["would_refuse"] = True
                rep["cost_params"] = 0
        else:                                  # remove_loop
            lb = getattr(host, "loop_block", None)
            if lb is None:
                rep["would_refuse"] = True
                rep["cost_params"] = 0
            else:
                m = int(host._bk.numel(lb["b_l"]))
                rep["cost_params"] = -(2 * m * host.H + m)
        rep["verdict"] = ("removal: negative mirror of the "
                          "grow accounting (R11 law)")
    elif move == "insert_layer":
        # 58 D-3.4: 3-D-host whole-layer copy; the estimate is
        # the pickled-TWIN's exact delta (read-only w.r.t. the
        # live host); the box judges it against the REAL call.
        if not hasattr(host, "insert_layer"):
            raise ValueError(
                f"host {type(host).__name__} has no "
                "insert_layer (reference Network deepens "
                "instead)")
        ga, ga_refuse = aspect_report(host, gpp)   # 60D mirror
        rep["gates"]["G-ASPECT"] = ga
        if ga_refuse and not args.get("force"):
            rep["would_refuse"] = True
        import pickle   # safe: round-trips the LOCAL host
        #                 object only (snapshot() precedent)
        twin = pickle.loads(pickle.dumps(host))
        try:
            twin.insert_layer(**args)
            rep["cost_params"] = int(twin.n_params()
                                     - host.n_params())
        except Exception as e:
            rep["would_refuse"] = True
            rep["cost_params"] = 0
            rep["refusal"] = str(e)
        rep["verdict"] = ("full-copy layer insertion "
                          "(55 s3-B2; preservation boxed)")
    elif move == "grow_site":
        # 58 D-3.5 (R4-1 symmetry): closed-form read-only
        # arithmetic — reference body (d*h + h + h*w + w) +
        # coupling A (w x m), the preset's own accounting.
        if not hasattr(host, "grow_site"):
            raise ValueError(
                f"host {type(host).__name__} has no grow_site")
        sp = args.get("site_path", "")
        hidden = int(args.get("hidden", 16))
        if "::" in sp:
            raise ValueError(
                "propose grow_site: deep ('::') paths are not "
                "supported in dry-run; propose on the inner "
                "body instead")
        try:
            l, j = host._parse(sp)
            ok = (0 <= l < host.L and 0 <= j < host.m
                  and (l, j) not in getattr(host, "inner", {})
                  and (l, j) not in getattr(host, "_port_js",
                                            set()))
        except Exception:
            ok = False
        w = int(gpp.get("grow_body_out_width", 1))
        if ok:
            rep["cost_params"] = (host.d * hidden + hidden
                                  + hidden * w + w
                                  + w * host.m)
        else:
            rep["would_refuse"] = True
            rep["cost_params"] = 0
        rep["verdict"] = ("fullwidth FFN-site body "
                          "(growth-port lineage, doc 35)")
    else:
        raise ValueError(f"propose: unknown move {move!r}; "
                         "allowed: deepen, grow, remove_grown,"
                         " remove_block, remove_loop, "
                         "insert_layer, grow_site")
    # ---- D-2: derived cost components on EVERY move ----
    itemsize = int(getattr(host, "dtype_itemsize", 8))
    rep["cost_mem_bytes"] = rep["cost_params"] * itemsize * 3
    rep["cost_step_frac"] = (rep["cost_params"]
                             / max(host.n_params(), 1))
    rep["cost_formulas"] = {
        "cost_mem_bytes": "cost_params*itemsize*3 (param+"
                          "Adam m+Adam v)",
        "cost_step_frac": "cost_params/n_params"}
    # ---- 72B D-2: G-BUDGET (whole-act, rider-inclusive) ----
    if move in ("deepen", "grow", "insert_layer",
                "grow_site") and \
            rep.get("cost_params") is not None:
        act = int(rep["cost_params"])
        ga = rep.get("gates", {}).get("G-ASPECT")
        if (move in ("deepen", "insert_layer") and ga
                and ga.get("met") is False
                and gpp.get("aspect_auto", "widen_first")
                == "widen_first"
                and not hasattr(host, "insert_layer")):
            import math
            k_r = (math.ceil(float(ga["floor"])
                             * int(ga["depth_after"]))
                   - int(ga["width"]))
            if k_r > 0:
                H_new = int(ga["width"]) + k_r
                mm = int(args.get("m") or H_new)
                act = (_widen_added_params(host, k_r)
                       + mm * H_new + mm + H_new * mm)
        gb = _budget_row(host, gpp, act)
        if gb is not None:
            rep["gates"]["G-BUDGET"] = gb
            if not gb["met"]:
                rep["would_refuse"] = True
    return rep


# ---------------- LIMITED-rule plans (FR-17/FR-12) ----------------

_RULE_TYPES = ("threshold", "limit", "schedule")
_LIMIT_KEYS = ("max_events", "max_params", "stop")


def validate_plan(host, plan, gpp):
    """CLOSED vocabulary only (owner: limited rule set):
      {"steps": [{"rule": "schedule"|"threshold",
                  "when": {"metric","op","value"}?,
                  "move": "deepen"|"grow"|"remove_grown"|
                          "remove_block"|"remove_loop"|
                          "insert_layer"|"grow_site",
                  "args": {...}},...],   # args pass VERBATIM
       "limits": {"max_events": int?, "max_params": int?}}
    Args reach the host method unchanged, so a step may carry
    "force": true (C-5 in automatic mode, 58 D-7.1); host-
    capability mismatches refuse LOUDLY at propose time.
    Unknown rule types / fields refused LOUDLY naming them.
    Returns per-step proposals + cumulative cost (FR-14 per
    step, before any compute is spent)."""
    if not isinstance(plan, dict) or "steps" not in plan:
        raise ValueError("plan: needs 'steps'")
    unknown = set(plan) - {"steps", "limits", "name"}
    if unknown:
        raise ValueError(f"plan: unknown field(s) "
                         f"{sorted(unknown)}")
    lim = plan.get("limits", {})
    bad = set(lim) - set(_LIMIT_KEYS)
    if bad:
        raise ValueError(f"plan.limits: unknown key(s) "
                         f"{sorted(bad)}")
    out, total = [], 0
    extra_stages = 0        # 60D shift-left: cumulative depth
    for i, st in enumerate(plan["steps"]):
        rule = st.get("rule", "schedule")
        if rule not in _RULE_TYPES:
            raise ValueError(
                f"plan step {i}: unknown rule type {rule!r}; "
                f"allowed: {_RULE_TYPES} (closed limited rule "
                "set)")
        rep = propose(host, st["move"], gpp,
                      **st.get("args", {}))
        if st["move"] in ("deepen", "insert_layer"):
            # 60D T-12: the plan is judged WHOLE — each depth
            # step's G-ASPECT mirror sees the depth the plan
            # has already scheduled before it (>= the
            # instantaneous verdict propose just wrote)
            ga, ga_refuse = aspect_report(
                host, gpp, extra_stages=extra_stages)
            rep["gates"]["G-ASPECT"] = ga
            if ga_refuse and not st.get("args",
                                        {}).get("force"):
                rep["would_refuse"] = True
            extra_stages += 1
        total += rep["cost_params"]
        # 72B D-2 [RV F5]: cumulative primary-cost budget
        # walk (riders are priced by the run pre-pass; the
        # walk marks the crossing STEP INDEX). `total`
        # already includes THIS step; the cumulative row
        # OVERRIDES propose's instantaneous one.
        gb_w = _budget_row(host, gpp, total)
        if gb_w is not None:
            rep["gates"]["G-BUDGET"] = gb_w
            if not gb_w["met"]:
                rep["would_refuse"] = True
        out.append({"step": i, "rule": rule,
                    "proposal": rep})
    return {"steps": out, "cumulative_cost_params": total,
            "limits": lim}


def load_plan(path):
    """Parameter-FILE plans (owner: rules via files, nothing
    hardcoded): file-loaded == programmatic (boxed)."""
    return _json.loads(open(path).read())


def plan_sha(plan):
    """Canonical plan identity — the VERBATIM GrowthStore.
    save_plan formula (58 D-4.1; intentional duplication,
    equality boxed as the drift guard)."""
    text = _json.dumps(plan, indent=1, sort_keys=True)
    return _hashlib.sha256(text.encode()).hexdigest()[:16]


def _audit_append(host, rec):
    """AUDIT-ONLY record dispatch (58 D-4.3/D-4.4): to the
    gain ledger when the host has one, else to the 3-D hosts'
    growth_events; NEVER the pending-gain machinery (the
    rollback-record law)."""
    led = getattr(host, "gain_ledger", None)
    if led is not None:
        led.append(rec)
        return
    if not hasattr(host, "growth_events"):
        host.growth_events = []
    host.growth_events.append(rec)


def _stamp_policy(host):
    """trigger='policy' on the newest event record, whichever
    home the host keeps (D-4.4 dispatch)."""
    led = getattr(host, "gain_ledger", None)
    if led:
        led[-1]["trigger"] = "policy"
    elif getattr(host, "growth_events", None):
        host.growth_events[-1]["trigger"] = "policy"


def apply_move(host, move, args):
    """The ONE move vocabulary (FR-12 symmetry): plan steps and
    bounded trials execute through this same table, so a move
    name means the same operation everywhere."""
    args = dict(args or {})
    if move == "deepen":
        host.deepen(**args)
    elif move == "grow":
        host.grow(**args)
    elif move == "remove_grown":                 # D-7.2
        host.remove_grown(args.get("key", args.get("j")))
    elif move == "remove_block":
        host.remove_block(args["k"])
    elif move == "remove_loop":
        host.remove_loop()
    elif move == "insert_layer":
        host.insert_layer(**args)
    elif move == "grow_site":
        host.grow_site(**args)
    else:
        raise ValueError(f"unknown move {move!r}")


def run_plan(host, plan, gpp, X, y, steps_between=10,
             control=None):
    """AUTOMATIC mode = rule-driven USER control. Executes the
    validated plan; HALTS at any user limit; every event is
    ledgered with trigger='policy'; C-5: `control` dict
    {"pause": bool, "abort": bool} is honored between events
    and a direct manual call ALWAYS wins (nothing here locks
    the host)."""
    validate_plan(host, plan, gpp)
    lim = plan.get("limits", {})
    events, halted = [], None
    # D-4.2: adoption record BEFORE the first event
    sha = plan_sha(plan)
    _audit_append(host, {"event": "plan_adopted",
                         "site": "plan", "plan_sha": sha,
                         "params_added": 0, "gain": None,
                         "trigger": "policy"})
    for i, st in enumerate(plan["steps"]):
        if control and control.get("abort"):
            halted = "aborted"; break
        while control and control.get("pause") \
                and not control.get("abort"):
            return {"events": events, "halted": "paused",
                    "resume_at": i}
        if lim.get("max_events") is not None \
                and len(events) >= lim["max_events"]:
            halted = "limit:max_events"; break
        if lim.get("max_params") is not None \
                and host.n_params() >= lim["max_params"]:
            halted = "limit:max_params"; break
        when = st.get("when")
        if when is not None:
            a = assess_growth(host)
            val = {"params": a["census"]["params"],
                   "blocks": a["census"]["blocks"]}.get(
                       when["metric"])
            if val is None:
                raise ValueError(f"plan step {i}: unknown "
                                 f"metric {when['metric']!r}")
            ok = {"<": val < when["value"],
                  ">": val > when["value"],
                  ">=": val >= when["value"],
                  "<=": val <= when["value"]}.get(when["op"])
            if ok is None:
                raise ValueError(f"plan step {i}: unknown op "
                                 f"{when['op']!r}")
            if not ok:
                continue
        move, args = st["move"], dict(st.get("args", {}))
        # 72B D-2: per-step whole-act budget at the CURRENT
        # state (covers conditional steps the pre-pass could
        # not price; propose's G-BUDGET is rider-inclusive)
        if gpp.get("params_budget_cap") is not None and \
                move in ("deepen", "grow", "insert_layer",
                         "grow_site"):
            prep = propose(host, move, gpp, **args)
            gb_s = prep.get("gates", {}).get("G-BUDGET")
            if gb_s is not None and not gb_s["met"]:
                halted = "params budget"
                events.append({"step": i, "move": move,
                               "refused": "params budget",
                               "cap": gb_s["cap"],
                               "post": gb_s["post"]})
                break
        apply_move(host, move, args)
        _stamp_policy(host)
        events.append({"step": i, "move": move})
        for _ in range(int(steps_between)):
            host.train_step(X, y)
    # D-4.2: halt record on every exit path (completed too)
    _audit_append(host, {"event": "plan_halted",
                         "site": "plan", "plan_sha": sha,
                         "halted": halted or "completed",
                         "events_run": len(events),
                         "params_added": 0, "gain": None,
                         "trigger": "policy"})
    return {"events": events, "halted": halted}


# ---------------- gate family (58 D-5; FR-12/OD-3) ----------

def gate_min_width(gpp, gate, qty, value, thr_key, mode_key,
                   force=False):
    """UNIFORM gate mechanism (the G-DEEPEN template, 58 D-5):
    estimator value -> USER-set threshold (policy key,
    parameter-file loadable, PERMISSIVE default 1 = nothing
    blocks until the user sets values) -> mode key advise/
    warn/refuse (default advise = silent proceed) -> C-5
    force. Threshold VALUES are the user's own parameters;
    NO derivation lives in the system (owner ruling
    2026-07-23). Returns "admitted"/"forced"/"advised"."""
    thr = int(gpp.get(thr_key, 1))
    if int(value) >= thr:
        return "admitted"
    if force:
        return "forced"
    msg = (f"{gate}: {qty} {int(value)} < threshold {thr} "
           f"({thr_key})")
    mode = gpp.get(mode_key, "advise")
    if mode == "refuse":
        raise ValueError(msg)
    if mode == "warn":
        import warnings
        warnings.warn(msg, stacklevel=3)
    return "advised"


# ---------------- aspect-ratio guardrail (60D) ----------------

def _aspect_shape(host):
    """Family width/stages (60D D-2, the measured truth):
    reference Network = trunk H over 1 (trunk) + blocks; 3-D
    hosts = d_model over L. Loop blocks are iteration, not a
    static stage — EXCLUDED from depth. Scoped deepen is judged
    on this GLOBAL shape (D-2 ruling)."""
    if hasattr(host, "insert_layer"):            # 3-D hosts
        return int(host.d), int(host.L)
    return int(host.H), 1 + len(getattr(host, "blocks", []))


def gate_aspect(gpp, width, depth_after, force=False):
    """G-ASPECT (60D D-3, the 58 D-5 template): aspect_after =
    width / depth_after against the USER-set floor
    gate_aspect_min (float; 0.0 = OFF — nothing blocks until
    the user sets a value; >= ADMITS the boundary). Mode
    advise (default, silent) / warn / refuse; C-5 force
    bypasses the gate, never budgets."""
    floor = float(gpp.get("gate_aspect_min", 0.0))
    aspect = width / max(int(depth_after), 1)
    if floor <= 0 or aspect >= floor:
        return "admitted"
    if force:
        return "forced"
    msg = (f"G-ASPECT: aspect {aspect:.2f} = width {width} / "
           f"depth {depth_after} < gate_aspect_min {floor} "
           "(width-first; see PARAMETER_REFERENCE guidance)")
    mode = gpp.get("gate_aspect_mode", "advise")
    if mode == "refuse":
        raise ValueError(msg)
    if mode == "warn":
        import warnings
        warnings.warn(msg, stacklevel=3)
    return "advised"


def aspect_report(host, gpp, extra_stages=0):
    """The G-ASPECT dry-run mirror entry (FR-14 symmetry:
    verdict == reality, boxed). extra_stages simulates depth
    already scheduled ahead of this step (the plan-validate
    cumulative walk — cumulative can only be STRICTER than
    the instantaneous mirror, never looser)."""
    width, stages = _aspect_shape(host)
    depth_after = stages + 1 + int(extra_stages)
    floor = float(gpp.get("gate_aspect_min", 0.0))
    aspect = width / max(depth_after, 1)
    met = floor <= 0 or aspect >= floor
    entry = {"aspect_after": aspect, "floor": floor,
             "met": met, "width": width,
             "depth_after": depth_after,
             "mode": gpp.get("gate_aspect_mode", "advise")}
    return entry, (not met) and entry["mode"] == "refuse"
