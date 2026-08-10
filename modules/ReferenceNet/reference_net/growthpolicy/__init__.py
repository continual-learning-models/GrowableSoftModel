"""Growth-direction policy: predict first, probe to verify, gate to
adopt (DESIGN_DEEPEN sections 5-6). Parts are replaceable behind
registries; selection is an explicit operator act via the policy
dict; the machinery never switches parts on its own (P8)."""
import json
from pathlib import Path

# ---- growth-operator standard vocabulary (T8; 60b) ----
OP_OMEGA = "widen"    # omega
OP_RHO   = "refine"   # rho   (formerly mislabeled "deepen")
OP_DELTA = "deepen"   # delta
GROWTH_OPERATORS = frozenset({OP_OMEGA, OP_RHO, OP_DELTA})

# ---- 59C stage 0 (doc 60 G8): product-gate registry of the
# inline-default policy keys. NOT members of
# DEFAULT_GROWTH_POLICY: adding them to the dict would change
# every policy built as dict(DEFAULT_GROWTH_POLICY, ...) and
# break the write-once recapture fixtures (I-8). The registry
# SET keeps hashes untouched while the facade/lifecycle doors
# accept the keys.
EXTENDED_GROWTH_KEYS = frozenset({
    "gate_deepen_mode",
    "gate_seam_min_width", "gate_seam_mode",
    "gate_nest_mode",
    "gate_scope_min_width", "gate_scope_mode",
    "gate_widen_mode",
    "train_lr_scales",
    "growth_auto_snapshot", "growth_snapshot_keep",
    # 60D aspect-ratio guardrail (G-ASPECT + auto-compliance):
    # floor 0.0 = OFF (permissive default); aspect_auto default
    # "widen_first" is INERT until a floor is set
    "gate_aspect_min", "gate_aspect_mode",
    "aspect_auto", "aspect_auto_max_widen",
})

from . import interfaces
from .interfaces import get, list_available, register  # noqa: F401
from .interfaces import (  # noqa: F401
    GROWTH_MODE_ADAPTIVE, GROWTH_MODE_WIDEN_ONLY,
    VALID_GROWTH_MODES, site_widen)
from . import extrapolate_domhan    # noqa: F401  (self-registers)
from . import forecastability_spectral  # noqa: F401
from . import changepoint_bocpd    # noqa: F401
from . import backtest_rolling     # noqa: F401
from . import pricer_zero_attach   # noqa: F401
from . import combiner_threshold   # noqa: F401
from . import preference           # noqa: F401  (registry
#   role "preference"; inert unless preference.rule != fixed)
# 84 D-4: the preference.* policy namespace enters the product
# doors through the SAME registry-set mechanism as the other
# extended keys (I-8: DEFAULT_GROWTH_POLICY untouched, fixture
# hashes preserved; values validated at the facade).
EXTENDED_GROWTH_KEYS = EXTENDED_GROWTH_KEYS | frozenset(
    preference.PREFERENCE_DEFAULTS)

# The single place an operator edits (DESIGN section 6). Every
# threshold is policy, never a constant baked into a part.
DEFAULT_GROWTH_POLICY = {
    "extrapolator": "domhan2015",
    "forecastability": "spectral_entropy",
    "changepoint": "bocpd",
    "backtest": "rolling_origin",
    "pricer": "zero_attach_v1",
    "combiner": "threshold_policy",
    "seed": 0,
    # cross-scale target step eta_t (PLAN 7.1; param-interface
    # batch docs/system/22 item 5; module default ETA_TARGET=0.5)
    "eta_target": 0.5,
    # instrument validity floors (S5, docs/system/22 items 30-34)
    "instrument_min_len": 64,
    "bocpd_recent": 32,
    # calibrated: power-law learning curves have 1/f-like spectra
    # (score ~0.32); white noise scores < 0.2. 0.25 separates them.
    "forecastability_min": 0.25,
    "changepoint_max": 0.2,
    "backtest_max_err": 0.25,
    "max_rel_ci_width": 0.5,
    "stall_k": 3,
    "stall_eps": 0.05,
    "saturation_norm": 1e-4,
    "energy_floor": 1e-3,
    "min_energy_points": 64,
    "min_ledger_events": 3,
    "min_window_rows": 64,
    "probe_steps": 300,
    "probe_lr": 0.05,
    "refine_hidden": 8,
    "max_blocks": 8,
    "log_path": None,
    # system control (DESIGN 12): adaptive two-direction growth by
    # default; GROWTH_MODE_WIDEN_ONLY restores the pure-linear
    # regime (the machinery never calls deepen; direct API stays
    # legal). Enforced in decide(), above the parts layer.
    "growth_mode": GROWTH_MODE_ADAPTIVE,
    # system control (DESIGN_GROW_BODY_TYPE 2): what kind of inner
    # body rho grows — user-selected, never self-switched (P8).
    # "reference" = the single-hidden-layer body (unchanged
    # default); "attention" = a complete small transformer
    # (reference_net/attention_body.py). Attention sizes below apply
    # only when grow_body_type == "attention"; ranges validated at
    # construction (bodies.make_body refuses violations loudly).
    "grow_body_type": "reference",
    "grow_attention_d_model": 8,
    "grow_attention_layers": 1,
    "grow_attention_heads": 2,
    "grow_attention_ffn": 16,
    # Growth Interface Reform (docs/system/35 D7): the coupling
    # port for NEW growth. "fullwidth" (vector body output u_g
    # assembled across the FULL host width by a trainable
    # zero-born A_g) is the default and the ONLY type selectable
    # for new growth; "legacy_scalar" is deprecated-defective and
    # load-only — grow refuses it loudly. grow_body_out_width =
    # k_g (int >= 1): the body's vector output width; 1 is the
    # neutral library default (capacity is CALLER strategy, never
    # a library constant — owner doctrine).
    "grow_port_type": "fullwidth",
    "grow_body_out_width": 1,
    # scale-hierarchy principle (owner ruling 2026-07-08): a grown
    # inner body is a FINE-scale correction and must be far smaller
    # than the scope hosting it — host's OWN params >= ratio x body
    # params, regardless of either side's type. Guard modes:
    # "warn" (default: proceed, warn, ledger-record the violation)
    # or "refuse" (grow raises loudly). Toy-scale test scenes run
    # under the recorded-warning regime by design; value exams
    # must be scale-sound.
    # (owner ruling addendum: an ABSOLUTE floor as well — a host
    # qualifies to grow inner bodies only from 100k own params up)
    # (owner ruling: topology change must be a SLOW SMALL change —
    # grow only after the base has trained into shape, never from
    # step zero)
    "grow_min_host_ratio": 100.0,
    "grow_min_host_params": 100000,
    "grow_min_host_steps": 100,
    "grow_scale_guard": "warn",
    # -- the loop operator (lambda): grown, governed directed
    #    cycles (DESIGN_LOOP_V2 v2.3). OFF by default: the
    #    default world carries no cycles and stays bitwise. --
    "loop_enabled": False,     # master switch (opt-in option)
    "loop_m": 8,               # lambda width (L_in rows)
    "loop_K_max": 32,          # Picard iteration budget
    "loop_tol": 1e-6,          # convergence tolerance (inf-norm)
    "loop_rho_max": 0.6,       # contraction cap (certificate)
}


def set_growth_mode(mode):
    """SYSTEM INTERFACE: set the global growth mode (user- or
    LLM-callable; the factory tool surface exposes it next round).
    Validated; returns {'old', 'new'} for the caller's log."""
    if mode not in VALID_GROWTH_MODES:
        raise ValueError(f"unknown growth_mode {mode!r}; valid: "
                         f"{sorted(VALID_GROWTH_MODES)}")
    old = DEFAULT_GROWTH_POLICY["growth_mode"]
    DEFAULT_GROWTH_POLICY["growth_mode"] = mode
    return {"old": old, "new": mode}


def get_growth_mode():
    """SYSTEM INTERFACE: the current global growth mode."""
    return DEFAULT_GROWTH_POLICY["growth_mode"]


def decide(scope, policy=None):
    """Run the full selection for one scope; returns the decision
    dict and appends one JSONL record to policy['log_path'] if set.
    The gate-verdict field is present but filled by the factory
    wiring (next round) — this round records decision + applied."""
    pol = dict(DEFAULT_GROWTH_POLICY)
    if policy:
        pol.update(policy)
    from ..net import Network as _Net        # lazy: no import cycle
    if not isinstance(scope, _Net):
        # DESIGN_GROW_BODY_TYPE 4b: structural operators do not
        # apply to non-reference bodies in v1; exclusion sits HERE,
        # upstream of the pricers (whose deepen probe would
        # otherwise deepcopy+deepen the body) and of every applier.
        return {"refusal": "scope body type does not support "
                           "structural operators (v1 boundary, "
                           "DESIGN_GROW_BODY_TYPE 4b)",
                "body_type": getattr(scope, "BODY_TYPE",
                                     type(scope).__name__)}
    mode = pol.get("growth_mode", GROWTH_MODE_ADAPTIVE)
    if mode not in VALID_GROWTH_MODES:
        return {"refusal": f"unknown growth_mode {mode!r}",
                "valid": sorted(VALID_GROWTH_MODES)}
    if mode == GROWTH_MODE_WIDEN_ONLY:
        # system control enforced ABOVE the parts layer (DESIGN 12):
        # short-circuit before any part is assembled — zero extra
        # cost, legacy pure-linear behavior, ordinary logged record.
        site, apply_as, note = site_widen(scope)
        decision = {"arm": OP_OMEGA, "site": site,
                    "apply_as": apply_as,
                    "tier_used": "widen_only_mode",
                    "reasons": ["growth_mode=widen_only: pure-linear "
                                "regime, parts not assembled"],
                    "certificate": {}, "parts": None,
                    "policy_snapshot": {k: v for k, v in pol.items()
                                        if k != "log_path"},
                    "gate_verdict": None}
        if note:
            decision["reasons"].append(note)
        if pol.get("log_path"):
            path = Path(pol["log_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(_jsonable(decision)) + "\n")
        return decision
    parts, names = {}, {}
    for role in ("extrapolator", "forecastability", "changepoint",
                 "backtest", "pricer", "combiner"):
        part = get(role, pol[role])
        if isinstance(part, dict):          # refusal
            return {"refusal": part["refusal"],
                    "available": part["available"]}
        parts[role] = part
        names[role] = pol[role]
    decision = parts["combiner"].decide(scope, parts, pol)
    decision["parts"] = names
    decision["policy_snapshot"] = {
        k: v for k, v in pol.items() if k != "log_path"}
    decision["gate_verdict"] = None          # filled next round
    if pol.get("log_path"):
        path = Path(pol["log_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(_jsonable(decision)) + "\n")
    return decision


def grow_with_policy(net, policy=None):
    """decide -> apply (rho or deepen at reference-net level; omega
    application is core wiring, next round) -> return decision."""
    pol = dict(DEFAULT_GROWTH_POLICY)
    if policy:
        pol.update(policy)
    decision = decide(net, pol)
    if decision.get("refusal"):
        return decision
    if decision["arm"] == OP_DELTA:
        if pol.get("growth_mode") == GROWTH_MODE_WIDEN_ONLY:
            # defense in depth: impossible by construction
            decision["applied"] = None
            decision["reasons"].append(
                "refused: growth_mode=widen_only (defense in depth)")
        elif len(net.blocks) >= pol["max_blocks"]:
            decision["applied"] = None
            decision["reasons"].append(
                f"budget refusal: max_blocks={pol['max_blocks']} "
                "reached (logged, not applied)")
        else:
            net.deepen()
            decision["applied"] = OP_DELTA
    elif decision["apply_as"] == "rho":
        net.grow(decision["site"], hidden=pol["refine_hidden"])
        decision["applied"] = f"rho@{decision['site']}"
    else:
        decision["applied"] = None           # omega: next round
    if decision.get("applied") and \
            preference.preference_enabled(pol):
        preference.on_adoption(net, decision, pol)   # pending
    return decision


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x
