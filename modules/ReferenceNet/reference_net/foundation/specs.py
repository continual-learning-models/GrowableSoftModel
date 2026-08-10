"""The FOUR SPEC OBJECTS (next-dev-docs/52 v1.7 section 4) —
the structured parameters of one growth event:

    grow(host, StructureSpec, WiringSpec, PlacementSpec,
         BirthSpec)

Every design quantity the old operator bodies WELDED (read
source, write target, span, position, birth appearance) is an
ordinary field here. The four dicts enter the ledger verbatim
as the event's birth certificate (FR-7).

DEFAULT POLICY (52 SR-3): mode-determining fields (reads,
write) are REQUIRED and have no defaults — an L1 default read
designation would itself encode a growth mode. Only
mode-neutral fields default: span=ALL (None), position=END,
chain=NONE, recipe="random".

Designation VOCABULARY (52 SR-16: defining the expressible
space is not choosing within it):
    "scope_input"  — the scope's input X_b
    "stream"       — the scope's running hidden stream
    explicit reference — a component id / chain name
Unknown fields and unknown designations are REFUSED loudly
(catalog discipline; V-A2-3).
"""

ALL = None          # full-span marker (structural no-op)
END = "end"         # append placement
NONE = "none"       # no chain placement (attachment-only)

_DESIGNATIONS = ("scope_input", "stream")
_TAP_ROLES = ("primary", "skip")


def _refuse_unknown(kind, given, allowed):
    extra = set(given) - set(allowed)
    if extra:
        raise ValueError(
            f"{kind}: unknown field(s) {sorted(extra)}; "
            f"allowed: {sorted(allowed)} (catalog discipline)")


class _Spec:
    _FIELDS = ()

    def as_dict(self):
        out = {}
        for f in self._FIELDS:
            v = getattr(self, f)
            out[f] = list(v) if isinstance(v, tuple) else v
        return out

    def __repr__(self):
        body = ", ".join(f"{f}={getattr(self, f)!r}"
                         for f in self._FIELDS)
        return f"{type(self).__name__}({body})"


class Tap(_Spec):
    """One read connection: {source, span, role}."""
    _FIELDS = ("source", "span", "role")

    def __init__(self, source, span=ALL, role="primary", **kw):
        _refuse_unknown("Tap", kw, ())
        if not (source in _DESIGNATIONS
                or isinstance(source, (str, int, tuple))):
            raise ValueError(f"Tap: undefined source "
                             f"designation {source!r}")
        if role not in _TAP_ROLES:
            raise ValueError(f"Tap: unknown role {role!r}; "
                             f"allowed: {_TAP_ROLES}")
        self.source, self.span, self.role = source, span, role

    def as_dict(self):
        return {"source": self.source,
                "span": None if self.span is None
                else list(self.span), "role": self.role}


class StructureSpec(_Spec):
    _FIELDS = ("kind", "params", "seed", "lr")

    def __init__(self, kind, params=None, seed=None, lr=None,
                 **kw):
        _refuse_unknown("StructureSpec", kw, ())
        if not isinstance(kind, str) or not kind:
            raise ValueError("StructureSpec: kind is REQUIRED")
        self.kind = kind
        self.params = dict(params or {})
        self.seed = seed
        self.lr = lr           # None -> host-derived (preset tier)


class WiringSpec(_Spec):
    _FIELDS = ("reads", "write")

    def __init__(self, reads, write, **kw):
        _refuse_unknown("WiringSpec", kw, ())
        if not reads:
            raise ValueError("WiringSpec: reads is REQUIRED "
                             "(no default — SR-3)")
        self.reads = tuple(t if isinstance(t, Tap) else Tap(**t)
                           for t in reads)
        if not (isinstance(write, dict) and "target" in write):
            raise ValueError("WiringSpec: write is REQUIRED "
                             "with a 'target' designation")
        _refuse_unknown("WiringSpec.write", write,
                        ("target", "span"))
        self.write = {"target": write["target"],
                      "span": write.get("span", ALL)}

    def as_dict(self):
        return {"reads": [t.as_dict() for t in self.reads],
                "write": {"target": self.write["target"],
                          "span": None
                          if self.write["span"] is None
                          else list(self.write["span"])}}


class PlacementSpec(_Spec):
    _FIELDS = ("chain", "position")

    def __init__(self, chain=NONE, position=END, **kw):
        _refuse_unknown("PlacementSpec", kw, ())
        self.chain = chain     # NONE = attachment-only (SR-15)
        self.position = position


class BirthSpec(_Spec):
    _FIELDS = ("zero_side", "recipe", "recipe_params")

    def __init__(self, zero_side="coupling", recipe="random",
                 recipe_params=None, **kw):
        _refuse_unknown("BirthSpec", kw, ())
        # zero_side: face designation | NONE (owner: zero birth
        # is OPTIONAL; NONE = full-value birth, preservation is
        # the caller's choice — FR-4 as amended)
        self.zero_side = zero_side
        self.recipe = recipe
        self.recipe_params = dict(recipe_params or {})


def specs_as_dict(structure, wiring, placement, birth):
    """The ledger 'specs' field (FR-7; SR-22 declared
    convention amendment): the event's complete provenance."""
    return {"structure": structure.as_dict(),
            "wiring": wiring.as_dict(),
            "placement": placement.as_dict(),
            "birth": birth.as_dict()}
