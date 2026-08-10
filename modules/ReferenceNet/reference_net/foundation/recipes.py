"""Initialization-RECIPE REGISTRY (next-dev-docs/52 v1.7
section 3.3) — birth appearance as pluggable data, nothing
hardwired (FR-4 as amended: the INITIAL VALUES of any new
structure are user-controllable; zero birth is OPTIONAL).

A recipe is a callable
    fn(shape_map, rng, context) -> {name: ndarray}
returning interior parameter values for the structure being
born. `context` gives READ-ONLY access to designated existing
parameters (for copy / interleave forms). The registry
performs NO admissibility judgment (record 50 v1.2): recipes
are descriptions; policy lives in L2.

Built-ins:
  "native"  — the kind's own seeded constructor performs the
              initialization (the HISTORICAL random birth;
              the A2 presets' default, bitwise by identity —
              the recipe body is a no-op).
  "random"  — alias of "native" for the standard forms (the
              documented meaning of the default).
  "zero"    — all-zero interior.
  "copy_layer" — verbatim copy of a designated source
              component's tensors (context["source"]).
  "interleave_neighbors" — LIDAS-form mean of the two
              designated neighbors (context["left"]/["right"]).
Additive only: register_recipe never mutates existing entries.
"""
import numpy as np


def _native(shape_map, rng, context):
    return {}                # constructor already initialized


def _zero(shape_map, rng, context):
    return {k: np.zeros(s) for k, s in shape_map.items()}


def _copy_layer(shape_map, rng, context):
    src = context["source"]
    out = {}
    for k, s in shape_map.items():
        v = np.array(src[k], copy=True)
        if tuple(v.shape) != tuple(s):
            raise ValueError(
                f"copy_layer: source tensor {k} has shape "
                f"{v.shape}, target needs {s}")
        out[k] = v
    return out


def _interleave_neighbors(shape_map, rng, context):
    left, right = context["left"], context["right"]
    out = {}
    for k, s in shape_map.items():
        v = 0.5 * (np.asarray(left[k], dtype=np.float64)
                   + np.asarray(right[k], dtype=np.float64))
        if tuple(v.shape) != tuple(s):
            raise ValueError(
                f"interleave_neighbors: {k} shape {v.shape} "
                f"!= target {s}")
        out[k] = v
    return out


RECIPES = {
    "native": _native,
    "random": _native,
    "zero": _zero,
    "copy_layer": _copy_layer,
    "interleave_neighbors": _interleave_neighbors,
}


def register_recipe(name, fn):
    """Additive registration; refuses to shadow (the core is
    never modified by extension — FR-4)."""
    if name in RECIPES:
        raise ValueError(f"recipe {name!r} already registered "
                         "(additive-only)")
    RECIPES[name] = fn


def get_recipe(name):
    try:
        return RECIPES[name]
    except KeyError:
        raise ValueError(
            f"unknown recipe {name!r}; registered: "
            f"{sorted(RECIPES)}") from None
