"""The COMPOSITION EXECUTOR (next-dev-docs/52 v1.7 section 5)
— mechanism only, no method knowledge.

    grow(host, structure_spec, wiring_spec, placement_spec,
         birth_spec) -> handle

The stages are PUBLIC, single-purpose functions (AR-5); a
preset may orchestrate them directly in order to interleave
its L2 guards at declared seams (52 SR-13 — the scale guard
consumes the BUILT structure, so a whole-call wrapper cannot
reproduce its position). No stage reads policy; no stage
contains a default that encodes a growth mode (PR-0).

STRUCTURE-KIND BUILDERS (52 AL-6, dependency inversion under
AR-2): foundation holds only this SOCKET; the single existing
registry — bodies.py, additive-only by charter — plugs its
builders in at import time (hosts/method import foundation
and register; foundation imports neither). No second registry
exists; unknown kinds are refused by the socket with the
registered catalog named.
"""

BUILDERS = {}       # kind -> callable(host, structure_spec) -> structure


def _identity_builder(host, spec):
    from .coupling import IdentitySource
    return IdentitySource(spec.params["width"])


BUILDERS["none"] = _identity_builder    # neutral vocabulary:
# a coupling-only event builds NO structure (52 s4 kind="none")


def register_builder(kind, fn):
    """Additive registration (used by bodies.py / hosts)."""
    if kind in BUILDERS:
        raise ValueError(f"builder {kind!r} already registered "
                         "(additive-only)")
    BUILDERS[kind] = fn


# ---------------- the six stages ----------------

def resolve(host, wiring_spec, placement_spec):
    """Designations -> concrete references; loud refusal on
    anything undefined. Returns a plain dict the later stages
    consume."""
    from .specs import NONE
    resolved = {"reads": [], "write": wiring_spec.write,
                "chain": placement_spec.chain,
                "position": placement_spec.position}
    for tap in wiring_spec.reads:
        if tap.source not in ("scope_input", "stream") and not \
                isinstance(tap.source, (int, tuple)):
            raise ValueError(f"resolve: undefined read "
                             f"designation {tap.source!r}")
        resolved["reads"].append(tap)
    if resolved["chain"] not in (NONE, "blocks", "layers"):
        raise ValueError(f"resolve: undefined chain "
                         f"designation {resolved['chain']!r}")
    return resolved


def build(host, structure_spec):
    """Structure instance via the registered builder; contract
    check on the result (conforms must be empty)."""
    from .contract import conforms
    try:
        builder = BUILDERS[structure_spec.kind]
    except KeyError:
        raise ValueError(
            f"build: unknown structure kind "
            f"{structure_spec.kind!r}; registered: "
            f"{sorted(BUILDERS)}") from None
    structure = builder(host, structure_spec)
    missing = conforms(structure)
    if missing:
        raise ValueError(f"build: {structure_spec.kind!r} "
                         f"structure misses {missing}")
    return structure


def birth(structure, birth_spec, context=None):
    """Interior initialization by recipe ("native"/"random" =
    the kind's own seeded constructor already did it — no-op,
    the historical birth bitwise). zero_side placement is the
    COUPLE/PLACE stages' concern (the zero face is the
    coupling matrix or a designated structure face); recipes
    only produce interior values."""
    from .recipes import get_recipe
    fn = get_recipe(birth_spec.recipe)
    values = fn(getattr(structure, "shape_map", lambda: {})()
                if callable(getattr(structure, "shape_map",
                                    None)) else {},
                None, context or birth_spec.recipe_params)
    for k, v in values.items():
        setattr(structure, k, v)
    return structure


def couple(host_site, structure, key, span_in=None,
           span_out=None):
    """Register the structure at the target's coupling site —
    the zero-born trainable link (P-2). Delegates to the
    site's add_body (which constructs the Coupling; A zero-born
    = the default zero_side; spans honored when designated)."""
    host_site.add_body(structure, key=key,
                       span_in=span_in, span_out=span_out)
    return host_site.bodies[-1]


def place(host, structure, resolved):
    """Chain insertion where applicable; chain=NONE is a
    structural no-op (SR-15: attachment-only growth)."""
    from .specs import NONE, END
    if resolved["chain"] == NONE:
        return None
    if resolved["chain"] == "blocks":
        pos = resolved["position"]
        if pos == END:
            host.blocks.append(structure)
            return len(host.blocks) - 1
        host.blocks.insert(int(pos), structure)
        return int(pos)
    raise ValueError(f"place: chain {resolved['chain']!r} not "
                     "placeable on this host")


def grow(host, structure_spec, wiring_spec, placement_spec,
         birth_spec, key=None):
    """FR-1: the ONE-SHOT composition of the public stages
    (presets may instead orchestrate the stages directly to
    interleave guards — 52 SR-13). Neutral: every designation
    comes from the specs; nothing here selects a mode."""
    from .specs import NONE, specs_as_dict
    resolved = resolve(host, wiring_spec, placement_spec)
    structure = build(host, structure_spec)
    birth(structure, birth_spec)
    primary = next(t for t in wiring_spec.reads
                   if t.role == "primary") \
        if any(t.role == "primary" for t in wiring_spec.reads) \
        else wiring_spec.reads[0]
    handle = None
    if resolved["chain"] == NONE:
        site = host._ensure_port_site()
        handle = couple(site, structure, key,
                        span_in=primary.span,
                        span_out=wiring_spec.write["span"])
    else:
        handle = place(host, structure, resolved)
    record(host, f"grow[{structure_spec.kind}]", key,
           structure.n_params(), specs_as_dict(
               structure_spec, wiring_spec, placement_spec,
               birth_spec))
    return handle


def record(host, event, site, params_added, specs_dict):
    """Ledger event with verbatim specs (FR-7; SR-22 declared
    convention amendment)."""
    rec = host._ledger_event(event, site, params_added)
    rec["specs"] = specs_dict
    rec.setdefault("trigger", "caller")   # FR-12 provenance
    #   (part of the declared ledger amendment, 52 SR-22:
    #    run_plan overwrites with "policy")
    return rec
