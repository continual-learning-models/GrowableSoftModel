"""Innovation-degree self-assessment (docs/system/26-30).

MDL in compression-progress form: the organ assesses, from its
OWN prequential losses only, whether incoming knowledge is
mere quantity (capacity matter) or a higher level its current
structure cannot express (leap matter). Design doc 27 (method,
five-layer verification spec) and doc 29 (this code level).

Architecture (doc 29 2.0): the CRITERION is a replaceable part
behind a named interface with a registry — the growthpolicy
doctrine copied verbatim: selection is an explicit operator
act via the policy dict; v1 ships exactly one implementation
(MDL); the engine never switches parts on its own.

The engine (SliceLedger + InnovationAssessor) is method-
agnostic and I/O-free: it holds NO organ reference, performs
no file access, and emits audit records as values
(drain_events) — persistence belongs to the service layers
(owner layering ruling, doc 29 3.4d).
"""
from __future__ import annotations

import math
from collections import deque

LADDER_CLASSES = ("widen", "head", "structure")
_STATE_LEARNING = "LEARNING"
_STATE_STALLED = "STALLED"
_STATE_TRIAL = "UNDER_TRIAL"
_STATE_MOI = "MASTERED_OR_INEXPRESSIBLE"
_EPS0 = 1e-9        # numeric-stability epsilon (allowlist)

V27_DEFAULTS = {
    "innovation_method": "mdl",
    "innovation_slice_mode": None,          # None = OFF
    "innovation_slice_min_obs": 256,
    "innovation_progress_window": 2,
    "innovation_progress_eps": 0.01,
    "innovation_ema_alpha": None,           # None = off
    "innovation_trial_cycles": 2,
    "innovation_cost_form": "const",
    "innovation_cost_per_param": 0.002,     # pilot-recalibrated
    "innovation_amortize_h": 1.0,
    "innovation_no_harm_eps": 0.005,
    "innovation_ladder_order": ("widen", "head", "structure"),
    "innovation_class_fails": 2,
    "innovation_backoff_levels": ((2, 2), (4, 4)),
    "innovation_allow_mse": False,
}

# ---------------- criterion role (replaceable part) ----------

_REGISTRY = {}


def register(part):
    _REGISTRY[part.NAME] = part
    return part


def get(name):
    if name not in _REGISTRY:
        raise ValueError(f"unknown innovation criterion "
                         f"{name!r}; available: "
                         f"{list_available()}")
    return _REGISTRY[name]


def list_available():
    return sorted(_REGISTRY)


class InnovationCriterion:
    """Role contract. A part owns ONLY method-specific
    arithmetic; the engine owns everything else."""

    NAME = "abstract"

    def demand_stalled(self, l_now, l_then, cfg):
        raise NotImplementedError

    def score_trial(self, inc_l, cand_l, slice_key,
                    added_params, n_positions, lifetime_n,
                    cfg):
        raise NotImplementedError


@register
class MDLCriterion(InnovationCriterion):
    """Two-part-code / prequential model selection (doc 27
    3.4): accept iff nats saved on the deficit slice exceed
    the description cost of the added parameters."""

    NAME = "mdl"

    def demand_stalled(self, l_now, l_then, cfg):
        p = l_then - l_now
        return abs(p) < float(cfg["innovation_progress_eps"]) \
            * max(l_now, _EPS0)

    def score_trial(self, inc_l, cand_l, slice_key,
                    added_params, n_positions, lifetime_n,
                    cfg):
        gain = (inc_l[slice_key] - cand_l[slice_key]) \
            * float(n_positions)
        form = cfg["innovation_cost_form"]
        if form == "const":
            per = float(cfg["innovation_cost_per_param"])
        else:                                # "half_log_n"
            per = 0.5 * math.log(max(lifetime_n, 2))
        cost = per * float(added_params)
        ok = gain > cost / float(cfg["innovation_amortize_h"])
        return {"slice_ok": bool(ok),
                "gain_nats": float(gain),
                "cost_nats": float(cost),
                "units": "nats"}


# ---------------- ledger (method-agnostic) -------------------

class SliceLedger:
    """Per-slice prequential bookkeeping. The FEEDER must obey
    the first-exposure rule (doc 27 v1.4 G1): only a fresh
    batch's first pre-update loss is a code term."""

    def __init__(self, hist_len):
        self._sums = {}
        self._counts = {}
        self._hist = {}
        self._lifetime_n = {}
        self._hist_len = int(hist_len)
        self._k = 0

    def record(self, slice_key, nats, n):
        if not (nats == nats) or nats in (float("inf"),
                                          float("-inf")):
            raise TypeError(f"non-finite loss for slice "
                            f"{slice_key!r}: {nats!r}")
        self._sums[slice_key] = self._sums.get(slice_key, 0.0) \
            + float(nats) * int(n)
        self._counts[slice_key] = self._counts.get(
            slice_key, 0) + int(n)
        self._lifetime_n[slice_key] = self._lifetime_n.get(
            slice_key, 0) + int(n)

    def close_cycle(self, ema_alpha=None):
        out = {}
        for s, c in self._counts.items():
            if c <= 0:
                continue
            mean = self._sums[s] / c
            h = self._hist.setdefault(
                s, deque(maxlen=self._hist_len))
            if ema_alpha and h:
                mean = float(ema_alpha) * mean \
                    + (1.0 - float(ema_alpha)) * h[-1][1]
            h.append((self._k, mean, c))
            out[s] = mean
        self._sums.clear()
        self._counts.clear()
        self._k += 1
        return out

    def l(self, slice_key):
        h = self._hist.get(slice_key)
        return h[-1][1] if h else None

    def l_at(self, slice_key, back):
        h = self._hist.get(slice_key)
        if not h or len(h) <= back:
            return None
        return h[-1 - back][1]

    def active(self, slice_key, min_obs):
        return self._lifetime_n.get(slice_key, 0) >= int(min_obs)

    def lifetime_n(self, slice_key):
        return self._lifetime_n.get(slice_key, 0)

    def slices(self):
        return sorted(set(self._hist) | set(self._lifetime_n))


# ---------------- validation ---------------------------------

def _validate_innovation(keys):
    v = {**V27_DEFAULTS, **keys}

    def bad(k, why):
        raise ValueError(f"invalid {k}={keys[k]!r}: {why}")

    if v["innovation_method"] not in _REGISTRY:
        raise ValueError(f"unknown innovation_method "
                         f"{v['innovation_method']!r}; "
                         f"available: {list_available()}")
    if v["innovation_slice_mode"] not in (
            None, "level_tag", "length_bucket", "task_family"):
        bad("innovation_slice_mode", "unknown mode")
    for k, lo in (("innovation_slice_min_obs", 1),
                  ("innovation_progress_window", 1),
                  ("innovation_trial_cycles", 1),
                  ("innovation_class_fails", 1)):
        if k in keys and (isinstance(keys[k], bool)
                          or not isinstance(keys[k], int)
                          or keys[k] < lo):
            bad(k, f"int >= {lo} required")
    if not 0 < float(v["innovation_progress_eps"]) < 1:
        bad("innovation_progress_eps", "0 < x < 1")
    a = v["innovation_ema_alpha"]
    if a is not None and not 0 < float(a) < 1:
        bad("innovation_ema_alpha", "None or 0 < a < 1")
    if v["innovation_cost_form"] not in ("const", "half_log_n"):
        bad("innovation_cost_form", "const | half_log_n")
    if not float(v["innovation_cost_per_param"]) > 0:
        bad("innovation_cost_per_param", "float > 0")
    if not float(v["innovation_amortize_h"]) >= 1:
        bad("innovation_amortize_h", "float >= 1")
    if not 0 <= float(v["innovation_no_harm_eps"]) < 1:
        bad("innovation_no_harm_eps", "0 <= x < 1")
    if sorted(v["innovation_ladder_order"]) != \
            sorted(LADDER_CLASSES):
        bad("innovation_ladder_order",
            f"permutation of {LADDER_CLASSES}")
    for pair in v["innovation_backoff_levels"]:
        if len(pair) != 2 or any(
                isinstance(x, bool) or not isinstance(x, int)
                or x < 1 for x in pair):
            bad("innovation_backoff_levels",
                "pairs of int >= 1")
    if not isinstance(v["innovation_allow_mse"], bool):
        bad("innovation_allow_mse", "bool")
    return v


# ---------------- engine -------------------------------------

class InnovationAssessor:
    """Method-agnostic engine: verdict machine, probe
    scheduling (priority, backoff, ladder), no-harm clause,
    audit records, evidence functions. Delegates the stall
    test and the trial score to the selected criterion part.
    Holds NO organ reference; does NO I/O."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.criterion = get(cfg["innovation_method"])()
        self.ledger = SliceLedger(
            int(cfg["innovation_progress_window"]) + 2)
        self._state = {}
        self._rejections = {}
        self._class_fail = {}
        self._ladder_pos = {}
        self._skip = {}
        self._moi_l = {}
        self._mastered = set()
        self._cum_gain = {}
        self._events = []

    # ---------- feeding + clocking ----------
    def observe(self, slice_key, nats, n):
        self.ledger.record(str(slice_key), float(nats), int(n))

    def close_cycle(self):
        closed = self.ledger.close_cycle(
            self.cfg["innovation_ema_alpha"])
        for s in closed:
            self._refresh_state(s)
        for s, l in closed.items():
            self._events.append(
                {"event": "selfassess_assess",
                 "cycle": self.ledger._k - 1, "slice": s,
                 "l": round(l, 6),
                 "state": self._state.get(s, _STATE_LEARNING)})
        return closed

    def _refresh_state(self, s):
        if self._state.get(s) == _STATE_TRIAL:
            return
        if self._state.get(s) == _STATE_MOI:
            # re-test when the codelength moved off the level
            # it was flagged at (world/knowledge changed)
            l = self.ledger.l(s)
            ref = self._moi_l.get(s)
            if l is not None and ref is not None and abs(
                    l - ref) > float(
                    self.cfg["innovation_progress_eps"]) * ref:
                self._state[s] = _STATE_LEARNING
                self._mastered.discard(s)
            return
        if not self.ledger.active(
                s, self.cfg["innovation_slice_min_obs"]):
            return
        l_now = self.ledger.l(s)
        l_then = self.ledger.l_at(
            s, int(self.cfg["innovation_progress_window"]))
        if l_now is None or l_then is None:
            self._state[s] = _STATE_LEARNING
            return
        stalled = self.criterion.demand_stalled(
            l_now, l_then, self.cfg)
        self._state[s] = _STATE_STALLED if stalled \
            else _STATE_LEARNING

    # ---------- verdicts + scheduling ----------
    def cycle_verdicts(self):
        return dict(self._state)

    def mastered_band(self):
        ls = [self.ledger.l(s) for s in self._mastered
              if self.ledger.l(s) is not None]
        return (min(ls), max(ls)) if ls else None

    def _priority(self, s):
        l = self.ledger.l(s)
        band = self.mastered_band()
        if band is None:
            return l                      # empty-band rule (G4)
        return max(0.0, l - band[1])

    def _cadence(self, s):
        cad = 1
        for rej, c in self.cfg["innovation_backoff_levels"]:
            if self._rejections.get(s, 0) >= int(rej):
                cad = int(c)
        return cad

    def next_probe(self):
        if any(v == _STATE_TRIAL for v in self._state.values()):
            return None                    # one live trial
        best, best_p = None, -1.0
        for s, st in self._state.items():
            if st != _STATE_STALLED:
                continue
            self._skip[s] = self._skip.get(s, 0) + 1
            if self._skip[s] < self._cadence(s):
                continue
            p = self._priority(s)
            if p is not None and p > best_p:
                best, best_p = s, p
        if best is None:
            return None
        self._skip[best] = 0
        order = self.cfg["innovation_ladder_order"]
        cls = order[min(self._ladder_pos.get(best, 0),
                        len(order) - 1)]
        return {"slice": best, "ladder_class": cls,
                "priority": float(best_p)}

    def register_spawn(self, slice_key, ladder_class):
        self._state[slice_key] = _STATE_TRIAL
        self._events.append(
            {"event": "selfassess_probe", "slice": slice_key,
             "ladder_class": ladder_class,
             "cycle": self.ledger._k})

    # ---------- acceptance (two-part test + no-harm) ----------
    def score_trial(self, inc_l, cand_l, slice_key,
                    added_params, n_positions):
        r = self.criterion.score_trial(
            inc_l, cand_l, slice_key, added_params,
            n_positions, self.ledger.lifetime_n(slice_key),
            self.cfg)
        harm_eps = float(self.cfg["innovation_no_harm_eps"])
        no_harm = all(
            cand_l.get(m, 0.0) <= inc_l.get(m, 0.0)
            * (1.0 + harm_eps)
            for m in self._mastered
            if m in inc_l and m in cand_l)
        r["no_harm_ok"] = bool(no_harm)
        r["accept"] = bool(r["slice_ok"] and no_harm)
        r["per_slice"] = {s: [inc_l.get(s), cand_l.get(s)]
                          for s in
                          set(inc_l) | set(cand_l)}
        return r

    def register_outcome(self, slice_key, accepted,
                         ladder_class, gain_nats=0.0):
        order = self.cfg["innovation_ladder_order"]
        self._events.append(
            {"event": "selfassess_verdict",
             "slice": slice_key, "accepted": bool(accepted),
             "ladder_class": ladder_class,
             "cycle": self.ledger._k})
        if accepted:
            kind = "small" if ladder_class == "widen" else "big"
            self._cum_gain[slice_key] = self._cum_gain.get(
                slice_key, 0.0) + float(gain_nats)
            self._events.append(
                {"event": "selfassess_innovation",
                 "slice": slice_key, "kind": kind,
                 "ladder_class": ladder_class,
                 "gain_nats": float(gain_nats),
                 "cumulative_gain":
                     self._cum_gain[slice_key],
                 "cycle": self.ledger._k})
            self._state[slice_key] = _STATE_LEARNING
            self._rejections[slice_key] = 0
            self._class_fail[slice_key] = 0
            self._ladder_pos[slice_key] = 0
            return
        self._rejections[slice_key] = self._rejections.get(
            slice_key, 0) + 1
        self._class_fail[slice_key] = self._class_fail.get(
            slice_key, 0) + 1
        if self._class_fail[slice_key] >= int(
                self.cfg["innovation_class_fails"]):
            self._class_fail[slice_key] = 0
            pos = self._ladder_pos.get(slice_key, 0) + 1
            if pos >= len(order):        # ladder exhausted
                self._state[slice_key] = _STATE_MOI
                self._moi_l[slice_key] = self.ledger.l(
                    slice_key)
                band = self.mastered_band()
                l = self.ledger.l(slice_key)
                # conservative band entry (doc 27 3.3 + G4
                # seed rule: an empty band is seeded)
                if l is not None and (band is None
                                      or l <= band[1]):
                    self._mastered.add(slice_key)
                self._ladder_pos[slice_key] = 0
                return
            self._ladder_pos[slice_key] = pos
        self._state[slice_key] = _STATE_STALLED

    # ---------- arrivals + reconfiguration ----------
    def on_arrival(self, slices=None):
        targets = (list(self._state) if slices is None
                   else [str(s) for s in slices])
        for s in targets:
            self._state[s] = _STATE_LEARNING
            self._rejections[s] = 0
            self._class_fail[s] = 0
            self._ladder_pos[s] = 0
            self._skip[s] = 0
            self._mastered.discard(s)
            self._moi_l.pop(s, None)
        self._events.append(
            {"event": "selfassess_arrival",
             "slices": (None if slices is None else targets),
             "cycle": self.ledger._k})

    def config(self):
        return dict(self.cfg)

    def reconfigure(self, partial_keys):
        if "innovation_method" in partial_keys:
            raise ValueError(
                "innovation_method is not hot-swappable; a "
                "criterion change is a new install by design")
        merged = _validate_innovation(
            {**{k: v for k, v in self.cfg.items()
                if k in V27_DEFAULTS}, **partial_keys})
        delta = {k: [self.cfg.get(k), partial_keys[k]]
                 for k in partial_keys}
        self.cfg.update(merged)
        self._events.append(
            {"event": "selfassess_reconfig", "delta": delta,
             "cycle": self.ledger._k})

    # ---------- evidence (read-only) ----------
    def innovation_degree(self, slice_key):
        return {"l": self.ledger.l(slice_key),
                "state": self._state.get(slice_key),
                "priority": (self._priority(slice_key)
                             if self.ledger.l(slice_key)
                             is not None else None),
                "cumulative_gain":
                    self._cum_gain.get(slice_key, 0.0)}

    def innovation_report(self):
        return {"cycle": self.ledger._k,
                "slices": {s: self.innovation_degree(s)
                           for s in self.ledger.slices()},
                "mastered_band": self.mastered_band(),
                "mastered": sorted(self._mastered),
                "config": self.config()}

    def drain_events(self):
        out, self._events = self._events, []
        return out


# ---------------- public installer ---------------------------

def install(organ, policy):
    """THE public entry point (doc 29 2.4/GAP-A): validates
    innovation_* keys, checks mode compatibility (doc 27
    3.1a), attaches organ._selfassess, returns the assessor.
    Directly-constructed experiment organs are first-class
    consumers of this function."""
    keys = {k: v for k, v in (policy or {}).items()
            if k.startswith("innovation_")}
    if not keys or keys.get("innovation_slice_mode") is None:
        return None
    bad = sorted(set(keys) - set(V27_DEFAULTS))
    if bad:
        raise ValueError(f"unknown innovation keys {bad}; "
                         f"valid: {sorted(V27_DEFAULTS)}")
    cfg = _validate_innovation(keys)
    mode = getattr(organ, "mode", None)
    if mode not in ("categorical", "numeric_dist"):
        if not cfg["innovation_allow_mse"]:
            raise ValueError(
                f"organ mode {mode!r} returns MSE, not a "
                "codelength (doc 27 3.1a); set "
                "innovation_allow_mse=True to admit it under "
                "the fixed-sigma reading (const cost form "
                "only)")
        if cfg["innovation_cost_form"] != "const":
            raise ValueError(
                "innovation_allow_mse requires "
                "cost_form='const' (half_log_n units are "
                "broken under MSE by definition)")
    organ._selfassess = InnovationAssessor(cfg)
    return organ._selfassess
