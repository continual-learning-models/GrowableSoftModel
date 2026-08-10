"""PRESETS (next-dev-docs/52 v1.7 section 6.1) — the existing
growth operators re-expressed as literal spec bundles plus L2
guard ORCHESTRATION at the declared stage seams (SR-13).

BEHAVIORAL IDENTITY (FR-6/AC-2, adjudicated by the A2 gates):
each preset executes the SAME kernel calls in the SAME order
as the pre-adjustment operator body — including the order and
count of RNG consumptions (SR-4) and the exact refusal
ordering (SR-13, as corrected by the P0 capture: duplicate/
port-type/out_width refusals precede the seed-counter
increment; the scale-guard refusal follows it). The specs are
recorded verbatim in the ledger as the event's birth
certificate (FR-7; SR-22 declared amendment).

Policy VALUES arrive as the `gpp` argument from the host tier
(this module never imports growthpolicy — V-A0-2).

Operator-acceptance verdicts (PR-2, math-research/02 T8):
  rho   — degree exponent unchanged, rank cap +k (widening
          move in disguise; Part B embedding);
  delta — degree exponent +1, rank cap unchanged (the
          depth-bearing move).
"""
import numpy as np

from reference_net.foundation import compose
from reference_net.foundation.specs import (
    ALL, END, NONE, BirthSpec, PlacementSpec, StructureSpec,
    Tap, WiringSpec, specs_as_dict)

# 59 R-1: the scoped-deepen theoretical advisory — a REMINDER
# (never a gate): single source, consumed by preset_layer and
# by the propose dry-run report.
_SCOPE_ADVISORY = (
    "scoped deepening presumes the scope is a functionally "
    "CLOSED subnetwork: the inserted layer learns g(x_S), a "
    "function of the scope's projection only; significant "
    "cross-boundary coupling makes the subproblem "
    "ill-conditioned and bounds the layer's attainable value "
    "by the scope's interface bandwidth (interface-"
    "completeness law). The system does not verify functional "
    "closure -- the caller judges from architectural "
    "knowledge. Birth topology remains lifetime-plastic (the "
    "confinement is an initialization hypothesis, not a "
    "cage).")


def preset_rho(host, j, hidden, body_type, gpp,
               force=False):
    """Reproduces Network.grow (net.py, growthport-v1) —
    scope-input read, full-width zero-born assembly at the
    host's shared port site."""
    # --- pre-mutation refusals (order preserved, SR-13) ---
    port_type = gpp.get("grow_port_type", "fullwidth")
    if port_type != "fullwidth":
        raise ValueError(
            f"grow_port_type {port_type!r} is refused for new "
            "growth: legacy_scalar is a load-only artifact-"
            "compat path (deprecated-defective, doc 35 R3/D7); "
            "new growth uses 'fullwidth'")
    out_width = gpp.get("grow_body_out_width", 1)
    if isinstance(out_width, bool) or not isinstance(
            out_width, (int, np.integer)) or out_width < 1:
        raise ValueError(
            f"grow_body_out_width must be an int >= 1; got "
            f"{out_width!r}")
    grown_js = getattr(host, "_port_js", set())
    if j in host.inner or j in grown_js:
        raise ValueError(f"node {j} already composite")
    # --- gate family (58 D-5): pre-mutation, PERMISSIVE
    #     defaults (factory behavior bit-unchanged); the
    #     scale guard's captured post-increment position is
    #     NOT moved (F1) ---
    from .gates import gate_min_width, width_demand
    gate_min_width(gpp, "G-SEAM", "junction width",
                   int(out_width), "gate_seam_min_width",
                   "gate_seam_mode", force=force)
    _mouth = next((r.get("mouth") for r in
                   reversed(getattr(host, "gain_ledger", []))
                   if r.get("event") == "nested_birth"), None)
    if _mouth is not None:            # growing INSIDE a grown
        #                               body (OD-3 ruled form)
        gate_min_width(gpp, "G-NEST", "path bottleneck width",
                       min(int(out_width), int(_mouth)),
                       "gate_seam_min_width", "gate_nest_mode",
                       force=force)
    if gpp.get("gate_widen_mode", "advise") != "advise" \
            and not force:            # G-WIDEN advisory
        _wd = width_demand(host)
        if _wd.get("met") is True:
            _msg = (f"G-WIDEN advisory: no width demand "
                    f"(r_hat {_wd['r_hat']} <= width "
                    f"{_wd['width']}) (gate_widen_mode)")
            if gpp.get("gate_widen_mode") == "refuse":
                raise ValueError(_msg)
            import warnings
            warnings.warn(_msg, stacklevel=2)
    # --- mutation begins (historical position) ---
    host._seed_counter += 1
    if body_type is None:
        body_type = gpp.get("grow_body_type", "reference")
    sspec = StructureSpec(kind=body_type,
                          params={"d_in": int(host.d_in),
                                  "hidden": int(hidden),
                                  "out_width": int(out_width)},
                          seed=host._seed_counter, lr=host.lr)
    wspec = WiringSpec(reads=[Tap("scope_input", span=ALL)],
                       write={"target": "stream", "span": ALL})
    pspec = PlacementSpec(chain=NONE)
    bspec = BirthSpec(zero_side="coupling", recipe="random")
    body = compose.build(host, sspec)
    # body_type recorded ONLY for non-reference grows: the
    # default ledger stays byte-identical (audit convention)
    event = "refine" if body_type == "reference" \
        else f"refine[{body_type}]"
    if hasattr(host, "_growth_policy"):
        # S2 propagation (docs/system/22 item 5)
        body._growth_policy = host._growth_policy
    if getattr(body, "gain_ledger", None) is not None:
        # 58 D-5.2: LEDGER-resident nesting marker (I-8 — no
        # attribute; survives save/load; audit-only)
        body.gain_ledger.append(
            {"event": "nested_birth", "site": j,
             "mouth": int(out_width), "params_added": 0,
             "gain": None, "trigger": "caller"})
    # --- L2 guard at the post-build / pre-couple seam ---
    host._scale_guard(body, j)
    site = host._ensure_port_site()
    compose.couple(site, body, key=j)
    if not hasattr(host, "_port_js"):
        host._port_js = set()
    host._port_js.add(j)
    compose.record(host, event, j,
                   body.n_params() + int(out_width) * host.H,
                   specs_as_dict(sspec, wspec, pspec, bspec))
    return body


def preset_delta(host, m):
    """Reproduces Network.deepen (net.py, growthport-v1) —
    one ZERO-INITIALIZED residual composition block appended
    to the scope's chain (exact at application: Bout zeros)."""
    m = host.H if m is None else int(m)
    host._seed_counter += 1
    sspec = StructureSpec(kind="block", params={"m": m},
                          seed=host._seed_counter, lr=host.lr)
    blk = compose.build(host, sspec)   # A4: contract-conformant
    #                                    Block, EXACT historical
    #                                    construction (builder)
    wspec = WiringSpec(reads=[Tap("stream", span=ALL)],
                       write={"target": "stream", "span": ALL})
    pspec = PlacementSpec(chain="blocks", position=END)
    bspec = BirthSpec(zero_side="Bout", recipe="random")
    resolved = compose.resolve(host, wspec, pspec)
    k = compose.place(host, blk, resolved)
    host._rebuild_opt()
    compose.record(host, "deepen", "scope",
                   m * host.H + m + host.H * m,
                   specs_as_dict(sspec, wspec, pspec, bspec))
    return k


def preset_site(host, l, j, hidden, gpp, site_factory,
                site_path, force=False):
    """Reproduces growth_port.grow_ffn_body (growthport-v1) —
    per-layer FFN-site growth on the 3-D hosts: layer-stream
    read (the post-LN FFN input rows), full FFN-hidden-width
    zero-born assembly; lr = host.lr x INNER_LR_FACTOR (the
    two-timescale factor, SR-11). Historically UNLEDGERED on
    these hosts (no gain-ledger machinery there) — preserved
    as-is at A2, recorded for close-out."""
    port_type = gpp.get("grow_port_type", "fullwidth")
    if port_type != "fullwidth":
        raise ValueError(
            f"grow_port_type {port_type!r} is refused for new "
            "growth: legacy_scalar is a load-only artifact-"
            "compat path (deprecated-defective, doc 35 R3/D7)")
    out_width = gpp.get("grow_body_out_width", 1)
    if isinstance(out_width, bool) or not isinstance(
            out_width, (int, np.integer)) or out_width < 1:
        raise ValueError(
            f"grow_body_out_width must be an int >= 1; got "
            f"{out_width!r}")
    # D-W7-2 (audit find): out-of-range sites were accepted
    # SILENTLY — a phantom port site is created whose body can
    # never run (forward walks range(L)): dead parameters.
    # Loud-refusal law; the propose dry-run already refused.
    if not (0 <= int(l) < int(host.L)
            and 0 <= int(j) < int(host.m)):
        raise ValueError(
            f"grow_site: site ({l}, {j}) out of range for "
            f"L={host.L}, m={host.m}: {site_path}")
    if (l, j) in host.inner \
            or (l, j) in getattr(host, "_port_js", set()):
        raise ValueError(f"site already composite: {site_path}")
    from .gates import gate_min_width      # 58 D-5.1
    gate_min_width(gpp, "G-SEAM", "junction width",
                   int(out_width), "gate_seam_min_width",
                   "gate_seam_mode", force=force)
    host._seed_counter += 1
    sspec = StructureSpec(kind="reference",
                          params={"d_in": int(host.d),
                                  "hidden": int(hidden),
                                  "out_width": int(out_width)},
                          seed=host._seed_counter,
                          lr=host.lr * host.INNER_LR_FACTOR)
    body = compose.build(host, sspec)
    if hasattr(host, "_growth_policy"):
        body._growth_policy = host._growth_policy   # S2
    if getattr(body, "gain_ledger", None) is not None:
        body.gain_ledger.append(       # 58 D-5.2 (I-8)
            {"event": "nested_birth", "site": (l, j),
             "mouth": int(out_width), "params_added": 0,
             "gain": None, "trigger": "caller"})
    site = site_factory()
    compose.couple(site, body, key=(l, j))
    if not hasattr(host, "_port_js"):
        host._port_js = set()
    host._port_js.add((l, j))
    # 58 D-6.1 (FR-7): the birth certificate — four verbatim
    # specs into the hosts' record home (they carry no gain
    # ledger; growth_events is their audit book, F5 lineage)
    # role stays in the CLOSED L1 set (I-3); the site
    # address lives in the record's own site/site_path fields
    wspec = WiringSpec(reads=[Tap("stream", span=ALL)],
                       write={"target": "stream", "span": ALL})
    pspec = PlacementSpec(chain=NONE)
    bspec = BirthSpec(zero_side="coupling", recipe="random")
    if not hasattr(host, "growth_events"):
        host.growth_events = []
    host.growth_events.append(
        {"event": "grow_site", "site": (l, j),
         "site_path": site_path,
         "n_params": int(body.n_params()
                         + int(out_width) * host.m),
         "specs": specs_as_dict(sspec, wspec, pspec, bspec),
         "trigger": "caller"})
    return body


def preset_loop(host, m):
    """Reproduces Network.loop's construction tail (net.py,
    growthport-v1) — the lambda block, ZERO-initialized on
    L_out (exact at application); the loop guard remains in
    the host operator (pre-mutation position preserved)."""
    host._seed_counter += 1
    sspec = StructureSpec(kind="loop", params={"m": int(m)},
                          seed=host._seed_counter, lr=host.lr)
    wspec = WiringSpec(reads=[Tap("stream", span=ALL)],
                       write={"target": "stream", "span": ALL})
    pspec = PlacementSpec(chain=NONE)   # scope-owned slot, not
    #                                     the blocks chain
    bspec = BirthSpec(zero_side="L_out", recipe="random")
    host.loop_block = compose.build(host, sspec)
    host._rebuild_opt()
    compose.record(host, "loop", "scope",
                   2 * int(m) * host.H + int(m),
                   specs_as_dict(sspec, wspec, pspec, bspec))
    return host.loop_block


def preset_layer(host, m=None, position=None, recipe=None,
                 recipe_params=None, scope=None,
                 zero_side=None, gpp=None, force=False):
    """PRESET_LAYER (doc 55 s3 B1) — whole-layer deepening at
    ANY designated scope: ONE rule, one preset. scope=None
    (ALL) = global (the trunk's full width); an ARBITRARY index
    set = local/expert deepening at birth topology. Position =
    any seam of the residual chain (END default). Birth recipe
    in {random, zero, copy_layer, interleave_neighbors, custom
    registered}; zero_side="Bout" (default) gives EXACT
    function preservation at any seam; zero_side=NONE is the
    full-value birth (e.g. complete copy of a designated
    layer, SOLAR copy-splice) — non-preserving by the caller's
    explicit choice (FR-4 as amended).

    SCOPE REALIZATION (recorded design decision, total-
    plasticity axiom): full-width storage with out-of-scope
    read columns and write rows born EXACTLY ZERO — "full
    connection within S, no external wiring" holds precisely
    AT BIRTH; lifetime training may densify (nothing is ever
    frozen; hard masking would be freezing by another name and
    is not a mechanism of this batch).

    Verdict (PR-2, math-research/02 T8): delta form — degree
    exponent +1, rank capacity unchanged; scoped events are
    delta WITHIN the scope (scope-relative, T8)."""
    m = host.H if m is None else int(m)
    if scope is not None:
        import warnings
        warnings.warn(_SCOPE_ADVISORY, stacklevel=3)  # 59 R-1
    if scope is not None and gpp is not None:
        from .gates import gate_min_width  # 58 D-5.3
        gate_min_width(gpp, "G-SCOPE", "scope width",
                       len(scope), "gate_scope_min_width",
                       "gate_scope_mode", force=force)
    if gpp is not None and not force:
        # L2 admission gate at the pre-mutation seam; strength
        # is the caller's policy key (advise/warn/refuse,
        # FR-12); force=True is the C-5 unconditional override
        # (the event then records the overridden verdict).
        from .gates import width_demand
        mode = gpp.get("gate_deepen_mode", "advise")
        wd = width_demand(host)
        if wd["met"] is False:
            msg = (f"G-DEEPEN width-first: width {wd['width']}"
                   f" < gradient-span estimate {wd['r_hat']}")
            if mode == "refuse":
                raise ValueError(msg)
            if mode == "warn":
                import warnings
                warnings.warn(msg, stacklevel=3)
    position = END if position is None else position
    recipe = "random" if recipe is None else recipe
    zero_side = "Bout" if zero_side is None else zero_side
    host._seed_counter += 1
    sspec = StructureSpec(kind="block", params={"m": m},
                          seed=host._seed_counter, lr=host.lr)
    span = ALL if scope is None else list(int(i) for i in scope)
    wspec = WiringSpec(reads=[Tap("stream", span=span)],
                       write={"target": "stream", "span": span})
    pspec = PlacementSpec(chain="blocks", position=position)
    bspec = BirthSpec(zero_side=("none" if zero_side is None
                                 or zero_side == "none"
                                 else zero_side),
                      recipe=recipe,
                      recipe_params=dict(recipe_params or {}))
    blk = compose.build(host, sspec)      # He interior, zero
    #                                       bb/Bout (historical)
    # ---- birth recipe (interior values) ----
    if recipe not in ("random", "native"):
        ctx = dict(bspec.recipe_params)
        if recipe == "copy_layer" and "source" not in ctx:
            if not host.blocks:
                raise ValueError(
                    "copy_layer needs an existing block "
                    "(or an explicit source)")
            src_i = ctx.pop("source_index", None)
            if src_i is None:
                src_i = (position - 1) if isinstance(
                    position, int) and position > 0 else 0
            ctx["source"] = host.blocks[int(src_i)]
        if recipe == "interleave_neighbors" and \
                "left" not in ctx:
            if not (isinstance(position, int)
                    and 1 <= position <= len(host.blocks) - 1):
                raise ValueError(
                    "interleave_neighbors needs an interior "
                    "seam with blocks on both sides")
            ctx["left"] = host.blocks[position - 1]
            ctx["right"] = host.blocks[position]
        compose.birth(blk, bspec, context=ctx)
        for k in blk.keys():              # recipes return numpy
            blk[k] = host._bk.ingest(np.asarray(blk[k]))
    # ---- zero side (the preservation face; caller's choice) --
    if bspec.zero_side == "Bout":
        # DEVICE-AGNOSTIC shape read (defect D-W6-1, caught by
        # the T-9 mps row: Bout is already ingested —
        # np.asarray crashes on device tensors; .shape exists
        # on every tensor family)
        blk.Bout = host._bk.ingest(
            np.zeros(tuple(blk.Bout.shape)))
    # ---- scope structuring at birth (out-of-scope zeroed) ----
    if scope is not None:
        S = np.asarray(sorted(set(int(i) for i in scope)))
        if S.min() < 0 or S.max() >= host.H:
            raise ValueError(f"scope indices out of range "
                             f"0..{host.H - 1}")
        mask_in = np.zeros(host.H); mask_in[S] = 1.0
        Bin = np.asarray(host._bk.to_numpy(blk.Bin)) * mask_in
        blk.Bin = host._bk.ingest(Bin)
        Bout = np.asarray(host._bk.to_numpy(blk.Bout))
        keep = np.zeros((host.H, 1)); keep[S] = 1.0
        blk.Bout = host._bk.ingest(Bout * keep)
    resolved = compose.resolve(host, wspec, pspec)
    k = compose.place(host, blk, resolved)
    host._rebuild_opt()
    compose.record(host, "deepen", "scope",
                   m * host.H + m + host.H * m,
                   specs_as_dict(sspec, wspec, pspec, bspec))
    return k
