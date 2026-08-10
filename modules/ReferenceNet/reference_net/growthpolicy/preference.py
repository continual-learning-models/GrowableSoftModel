"""GrowthPreference — the S-loop L1 StructureAdapter (doc 83
v1.16 M1-M8; doc 86 §3.0 layering: all shared arithmetic
delegates to evaluative_core; only THIS module knows buckets,
contexts and blobs).

Registry role "preference" (registered like every algorithmic
role); the ONLY existing-code seam is the combiner's tier-2
consultation (doc 83 §4.3, ~10 lines) plus the adoption/window
hooks invoked by the lifecycle.

INERTNESS PRECEDENCE (doc 83 M2 v1.13): rule == "fixed" (the
shipping default) bypasses the ENTIRE path — no bucketer call,
no slope sourcing, no draws, no state; provably byte-identical
(TI-02).

Every constant is a policy key (PREFERENCE_DEFAULTS); audit
events are plain dicts drained by the caller into the standard
decision/audit channel.
"""
import copy
import hashlib
import json
from bisect import bisect_right
from math import sqrt

import numpy as np

from . import evaluative_core as core
from . import OP_DELTA, OP_OMEGA

SCHEMA = "pref-v1"
PRIOR_SCHEMA = "prior-v2"
RULES = ("fixed", "mean_clip", "thompson", "ucb", "eps_greedy")
BUCKET_SPECS = ("b0", "b1")            # b2 reserved (doc 83 M2)
_BLOB_ATTR = "_preference_blob"
_TAIL_MAX = 20

PREFERENCE_DEFAULTS = {
    "preference.rule": "fixed",
    "preference.rule_mix": {},
    "preference.bucket_spec": "b1",
    "preference.decay": 0.98,
    "preference.credit_weights": [1.0, 0.5, 0.25],
    "preference.clip_lo": 0.5,
    "preference.clip_hi": 2.0,
    "preference.min_count": 3,
    "preference.explore_quota": 1,
    "preference.quota_window": "life",
    "preference.slope_probes": True,
    "preference.slope_cuts": [-0.01, 0.01],
    "preference.eps": 0.1,
    "preference.ucb_c": 1.0,
    "preference.rollback_mode": "keep",
    "preference.prior_path": "",
    "preference.prior_weight_cap": 12.0,
    "preference.bocpd_coupling": False,
    "preference.bocpd_depth": 0.5,
    "preference.ts_seed_offset": 10000,
    "preference.audit_draw_mode": "full",
    "preference.world_type": "supervised",
    "preference.explore_draw_min": 1.2,
}


class PreferenceAuditError(Exception):
    """The audit-only law (doc 83 §2.4): governance-derived
    events must never enter preference statistics."""


_KEY_TYPES = {
    # G-6 (plan 96 E-1): TYPE PRE-PASS — every key refuses
    # non-conforming types LOUDLY before any range logic (the
    # §4.5 law; GK-03 gate-enforced; bool is excluded from
    # numeric keys because bool subclasses int).
    "preference.rule": str, "preference.rule_mix": dict,
    "preference.bucket_spec": str,
    "preference.decay": (int, float),
    "preference.credit_weights": (list, tuple),
    "preference.clip_lo": (int, float),
    "preference.clip_hi": (int, float),
    "preference.min_count": int,
    "preference.explore_quota": int,
    "preference.quota_window": str,
    "preference.slope_probes": bool,
    "preference.slope_cuts": (list, tuple),
    "preference.eps": (int, float),
    "preference.ucb_c": (int, float),
    "preference.rollback_mode": str,
    "preference.prior_path": str,
    "preference.prior_weight_cap": (int, float),
    "preference.bocpd_coupling": bool,
    "preference.bocpd_depth": (int, float),
    "preference.ts_seed_offset": int,
    "preference.audit_draw_mode": str,
    "preference.world_type": str,
    "preference.explore_draw_min": (int, float),
}


def _type_ok(v, t):
    if t is bool:
        return isinstance(v, bool)
    if isinstance(v, bool):          # bool never counts as a number
        return False
    return isinstance(v, t)


def validate_preference_policy(policy):
    """Set-policy-time validation (doc 83 §4.5): refusal dict on
    any invalid preference.* value, None when clean. NEVER a
    silent fallback — and NEVER a crash (G-6/GK-03: the type
    pre-pass refuses non-conforming types before range logic)."""
    def _ref(msg):
        return {"refusal": f"preference policy: {msg}"}
    p0 = policy or {}
    # None is the merge-deletion sentinel (facade merge-None
    # semantics, exercised by preference_reset) — never typed
    for _k, _t in _KEY_TYPES.items():
        if p0.get(_k) is not None and not _type_ok(p0[_k], _t):
            return _ref(f"{_k}={p0[_k]!r} has wrong "
                        f"type; expected {_t}")
    for _k in ("preference.slope_cuts",
               "preference.credit_weights"):
        if p0.get(_k) is not None and not all(
                _type_ok(x, (int, float))
                for x in p0[_k]):
            return _ref(f"{_k} elements must be numbers")
    p = policy or {}
    rule = p.get("preference.rule")
    if rule is not None and rule not in RULES:
        return _ref(f"unknown rule {rule!r}; valid {RULES}")
    spec = p.get("preference.bucket_spec")
    if spec is not None and spec not in BUCKET_SPECS:
        if spec == "b2":
            return _ref("bucket_spec b2 is reserved (doc 83 M2),"
                        " not selectable in v1")
        return _ref(f"unknown bucket_spec {spec!r}")
    dec = p.get("preference.decay")
    if dec is not None and not (0.5 < float(dec) <= 1.0):
        return _ref(f"decay {dec} outside (0.5, 1.0]")
    lo = p.get("preference.clip_lo")
    hi = p.get("preference.clip_hi")
    lo_v = float(lo) if lo is not None else None
    hi_v = float(hi) if hi is not None else None
    if lo_v is not None and not (0.0 < lo_v <= 1.0):
        return _ref(f"clip_lo {lo} outside (0, 1]")
    if hi_v is not None and not (1.0 <= hi_v <= 10.0):
        return _ref(f"clip_hi {hi} outside [1, 10]")
    mc = p.get("preference.min_count")
    if mc is not None and int(mc) < 1:
        return _ref(f"min_count {mc} < 1")
    eq = p.get("preference.explore_quota")
    if eq is not None and int(eq) < 0:
        return _ref(f"explore_quota {eq} < 0")
    qw = p.get("preference.quota_window")
    if qw is not None and qw != "life":
        n = str(qw).split(":", 1)
        if not (str(qw).startswith("batches:") and len(n) == 2
                and n[1].isdigit() and int(n[1]) >= 1):
            return _ref(f"quota_window {qw!r} invalid; valid: "
                        "'life' or 'batches:N' with integer "
                        "N >= 1")
    eps = p.get("preference.eps")
    if eps is not None and not (0.0 <= float(eps) < 1.0):
        return _ref(f"eps {eps} outside [0, 1)")
    c = p.get("preference.ucb_c")
    if c is not None and float(c) <= 0:
        return _ref(f"ucb_c {c} <= 0")
    rb = p.get("preference.rollback_mode")
    if rb is not None and rb not in ("keep", "revert"):
        return _ref(f"rollback_mode {rb!r} invalid")
    cap = p.get("preference.prior_weight_cap")
    if cap is not None and float(cap) <= 0:
        return _ref(f"prior_weight_cap {cap} <= 0")
    bd = p.get("preference.bocpd_depth")
    if bd is not None and not (0.0 < float(bd) <= 1.0):
        return _ref(f"bocpd_depth {bd} outside (0, 1]")
    adm = p.get("preference.audit_draw_mode")
    if adm is not None and not (adm == "full"
                                or str(adm).startswith("sampled:")):
        return _ref(f"audit_draw_mode {adm!r} invalid")
    cw = p.get("preference.credit_weights")
    if cw is not None and (len(cw) == 0 or len(cw) > 5
                           or any(x <= 0 for x in cw)):
        return _ref("credit_weights must be 1..5 positive values")
    edm = p.get("preference.explore_draw_min")
    if edm is not None and float(edm) <= 1.0:
        return _ref(f"explore_draw_min {edm} must be > 1")
    wt = p.get("preference.world_type")
    if wt is not None and (not isinstance(wt, str) or not wt):
        return _ref("world_type must be a nonempty string")
    mix = p.get("preference.rule_mix")
    if mix:
        for r in mix:
            if r not in RULES or r == "fixed":
                return _ref(f"rule_mix has invalid rule {r!r}")
        if sum(mix.values()) > 1.0 + 1e-12:
            return _ref("rule_mix weights sum > 1")
    return None


def bucket_of(ctx, policy):
    """M2 ContextBucketer: b0 -> move; b1 -> move|s<band> via
    bisect_right over preference.slope_cuts; slope None under b1
    falls back to the b0 form (the CALLER audits)."""
    spec = policy.get("preference.bucket_spec",
                      PREFERENCE_DEFAULTS["preference.bucket_spec"])
    move = str(ctx.get("move"))
    if spec == "b0":
        return move
    slope = ctx.get("slope")
    if slope is None:
        return move
    cuts = policy.get("preference.slope_cuts",
                      PREFERENCE_DEFAULTS["preference.slope_cuts"])
    return f"{move}|s{bisect_right(list(cuts), float(slope))}"


def assemble_credit_event(event_id, bucket, move, batch,
                          quoted_gain, window_gains, weights):
    """M3 RewardCrediting arithmetic (doc 83 §4.1): credited =
    sum(w*g)/sum(w) over the K windows; advantage = credited -
    quoted (folds via the L0 core)."""
    w = core.credit_weights({"kind": "list",
                             "weights": list(weights)})
    credited = core.credit_fold(list(window_gains), w,
                                normalize=True)
    return {"event_id": event_id, "bucket": bucket, "move": move,
            "batch": batch, "quoted_gain": float(quoted_gain),
            "window_gains": [float(g) for g in window_gains],
            "credited_gain": float(credited),
            "advantage": float(credited - float(quoted_gain))}


def migrate_snapshot(blob):
    """P-1 loading rules: same-version -> verbatim; OLDER known
    version -> registered migration (none exist before pref-v1);
    UNKNOWN/NEWER -> refusal (caller rebuilds from the credit
    stream — never guess)."""
    schema = (blob or {}).get("schema")
    if schema == SCHEMA:
        return blob
    return {"refusal": f"unknown preference snapshot schema "
                       f"{schema!r} (known: {SCHEMA}); refuse and"
                       f" rebuild from credit events"}


class GrowthPreference:
    """The stateful preference part (M1 store + M4 rules + M6
    envelope + M7 quota + M5/M8 hooks). Constructed from a plain
    policy dict (missing keys take PREFERENCE_DEFAULTS)."""

    def __init__(self, policy=None):
        self.policy = dict(PREFERENCE_DEFAULTS)
        if policy:
            self.policy.update(policy)
        self.audit_events = []
        self._stats = {}          # bucket -> [w, m, v, n_raw]
        self._quota_used = 0
        self._quota_win = -1          # M7 batches:N window cursor
        self._fold_cursor = 0
        self._pending = []
        self._credited_tail = []
        self._prior_fingerprint = None
        self._draw_count = 0
        self._b1_fallbacks = 0
        self._estats = [0, 0.0, 0.0, 0.0]   # n, w, m, v (v1.17
        #   reference distribution of credited advantages)
        seed = int(self.policy.get("seed", 0)) + \
            int(self.policy["preference.ts_seed_offset"])
        self._rng = np.random.default_rng(seed)
        path = self.policy["preference.prior_path"]
        if path:
            self._load_prior(path)

    # ---------------- policy shorthands ----------------
    def _p(self, key):
        return self.policy[f"preference.{key}"]

    def _fold_event(self, a):
        n, w, m, v = self._estats
        w2, m2, v2 = core.ema_fold((w, m, v), a,
                                   self._p("decay"))
        self._estats = [n + 1, w2, m2, v2]

    def _ref_stats(self):
        """(mu_ev, sd_ev) of the credited-advantage
        distribution; degenerate when n_ev < 2 (the
        normalized_multiplier applies its own sd tolerance)."""
        n, _w, m, v = self._estats
        if n < 2:
            return 0.0, 0.0
        return m, sqrt(max(v, 0.0))

    # ---------------- prior (M8) ----------------
    def _load_prior(self, path):
        try:
            raw = open(path, "rb").read()
            art = json.loads(raw.decode())
            assert art.get("schema") == PRIOR_SCHEMA
        except Exception as e:                     # P-4: refusal +
            self.audit_events.append(              # inert continue
                {"kind": "prior_refusal",
                 "refusal": f"prior artifact unreadable: {e}"})
            return
        cap = float(self._p("prior_weight_cap"))
        wt = str(self._p("world_type"))
        capped = False
        buckets = (art.get("buckets") or {}).get(wt, {})
        for b, s in buckets.items():
            w = float(s["w"])
            if w > cap:
                w, capped = cap, True
            self._stats[b] = [w, float(s["m"]), float(s["v"]),
                              int(s.get("n", 0))]
        st = (art.get("strata") or {}).get(wt)
        if st and int(st.get("n", 0)) >= 2:
            n_ev = int(st["n"])
            self._estats = [n_ev, float(min(n_ev, cap)),
                            float(st["mu"]),
                            float(st["sd"]) ** 2]
        self._prior_fingerprint = hashlib.sha256(raw).hexdigest()
        self.audit_events.append(
            {"kind": "prior_load",
             "path_sha": self._prior_fingerprint,
             "world_type": wt, "buckets": len(buckets),
             "weight_cap_applied": capped})

    # ---------------- scoring (M4 + M6) ----------------
    def score(self, ctx):
        return self.score_set([ctx])[0]

    def score_set(self, ctxs):
        rule = self._p("rule")
        if rule == "fixed":            # INERTNESS PRECEDENCE:
            return [1.0] * len(ctxs)   # zero side effects
        lo, hi = self._p("clip_lo"), self._p("clip_hi")
        min_count = int(self._p("min_count"))
        mu_ev, sd_ev = self._ref_stats()
        mults = [1.0] * len(ctxs)
        for i, ctx in enumerate(ctxs):
            b = self._bucket_audited(ctx)
            w, m, v, n = self._stats.get(b, (0.0, 0.0, 0.0, 0))
            if n < min_count:
                continue                      # floored: 1.0,
            raw, rule_name, draw_se = self._raw_score(
                rule, w, m, v)             # no side effects
            mults[i] = core.normalized_multiplier(
                raw, mu_ev, sd_ev, lo, hi)
            self._log_draw(b, rule_name, draw_se, raw,
                           mults[i], lo, hi)
        return mults

    def _raw_score(self, rule, w, m, v):
        mix = self._p("rule_mix")
        if mix:
            total, wsum, var_mix = 0.0, 0.0, 0.0
            for r in sorted(mix):
                val, _, se_r = self._single_raw(r, w, m, v)
                total += mix[r] * val
                var_mix += (mix[r] * se_r) ** 2
                wsum += mix[r]
            rest = 1.0 - wsum
            if rest > 1e-12:
                val, _, se_r = self._single_raw(rule, w, m, v)
                total += rest * val
                var_mix += (rest * se_r) ** 2
            # independent component draws => the blend's own
            # Gaussian scale is the weighted quadrature sum
            return total, f"mix({'+'.join(sorted(mix))})", \
                sqrt(var_mix)
        return self._single_raw(rule, w, m, v)

    def _single_raw(self, rule, w, m, v):
        # returns (raw, rule_name, draw_se) — draw_se is the
        # ACTUAL Gaussian scale of this raw's stochastic draw
        # (0.0 for deterministic paths); it is what the audit
        # event logs as "se" (a checker must reproduce draw =
        # mean + se*z from the log).
        if rule == "mean_clip":
            return m, "mean_clip", 0.0
        if rule == "thompson":
            if w <= 0.0:
                return m, "thompson", 0.0     # no evidence at all
            v_ev = max(self._estats[3], 0.0)
            if v_ev <= 0.0:
                return m, "thompson", 0.0     # degenerate reference
            # PRODUCTION-STANDARD fixed-scale Gaussian TS (doc 83
            # v1.19 M4): scale = global reference variance / (w+1);
            # the per-bucket empirical variance NEVER enters the
            # scale — a dust-scale thin Gaussian tail from lucky
            # coinciding outcomes would lock the arm out forever.
            se_t = sqrt(v_ev / (w + 1.0))
            return core.seeded_draw(self._rng, m, se_t), \
                "thompson", se_t
        if rule == "ucb":
            se = sqrt(v / w) if (w > 0 and v > 0) else 0.0
            return m + float(self._p("ucb_c")) * se, "ucb", 0.0
        if rule == "eps_greedy":
            u = float(self._rng.uniform())
            mu_ev, sd_ev = self._ref_stats()
            if u < float(self._p("eps")) and sd_ev > 0.0:
                return (mu_ev + sd_ev
                        * float(self._rng.standard_normal()),
                        "eps_random", sd_ev)  # GAIN units (v1.17)
            return m, "eps_greedy", 0.0
        raise ValueError(f"unknown rule {rule!r}")

    def _bucket_audited(self, ctx):
        spec = self._p("bucket_spec")
        if spec == "b1" and ctx.get("slope") is None:
            self._b1_fallbacks += 1
            self.audit_events.append(
                {"kind": "b1_fallback", "move": ctx.get("move"),
                 "count": self._b1_fallbacks})
        return bucket_of(ctx, self.policy)

    def _log_draw(self, bucket, rule_name, draw_se, raw, mult,
                  lo, hi):
        self._draw_count += 1
        mode = str(self._p("audit_draw_mode"))
        boundary = mult in (float(lo), float(hi)) or \
            rule_name == "eps_random"
        if mode.startswith("sampled:") and not boundary:
            n = int(mode.split(":", 1)[1])
            if self._draw_count % n != 0:
                return
        b_stats = self._stats.get(bucket, (0.0, 0.0, 0.0, 0))
        ev = {"kind": "preference_draw", "bucket": bucket,
              "rule": rule_name, "mean": b_stats[1],
              # "se" = the ACTUAL Gaussian scale of this draw
              # (0.0 for deterministic paths) — a checker must
              # reproduce draw = mean + se*z from the log; the
              # eps_random path draws around mu_ev, not mean.
              "se": float(draw_se),
              "draw": raw, "multiplier_out": mult}
        if mode.startswith("sampled:"):
            ev["rng_checkpoint"] = self._rng_state()
        self.audit_events.append(ev)

    # ---------------- credit (M1 + M3) ----------------
    def credit(self, ev):
        if ev.get("source_kind") == "governance" or \
                ev.get("credited_gain") is None:
            raise PreferenceAuditError(
                "governance-derived event refused (audit-only "
                f"law): {ev.get('event_id')}")
        b = ev["bucket"]
        w, m, v, n = self._stats.get(b, (0.0, 0.0, 0.0, 0))
        w2, m2, v2 = core.ema_fold((w, m, v), ev["advantage"],
                                   self._p("decay"))
        self._stats[b] = [w2, m2, v2, n + 1]
        self._fold_event(float(ev["advantage"]))
        self._fold_cursor += 1
        self._credited_tail.append(ev)
        del self._credited_tail[:-_TAIL_MAX]
        self.audit_events.append(
            {"kind": "preference_update", "bucket": b,
             "advantage": ev["advantage"], "w_after": w2,
             "m_after": m2, "v_after": v2})

    # ---------------- explore (M7) ----------------
    def explore_offer(self, ctx, raw_eff):
        quota = int(self._p("explore_quota"))
        # M7 quota_window: life = one budget forever;
        # batches:N = the budget refreshes each N-batch window
        qw = str(self._p("quota_window"))
        if qw.startswith("batches:"):
            win = int(ctx.get("batch", 0)) // int(
                qw.split(":", 1)[1])
            if win != self._quota_win:
                self._quota_win = win
                self._quota_used = 0
        b = self._bucket_audited(ctx)
        w, m, v, _n = self._stats.get(b, (0.0, 0.0, 0.0, 0))
        v_ev = max(self._estats[3], 0.0)
        if v_ev <= 0.0 or w <= 0.0:
            draw = m
        else:
            # same normative thompson form as M4 (doc 83 v1.19):
            # fixed reference-variance scale, one form defined once
            draw = core.seeded_draw(self._rng, m,
                                    sqrt(v_ev / (w + 1.0)))
        mu_ev, sd_ev = self._ref_stats()
        mult = core.normalized_multiplier(
            draw, mu_ev, sd_ev, self._p("clip_lo"),
            self._p("clip_hi"))
        favorable = mult > float(self._p("explore_draw_min"))
        granted = favorable and self._quota_used < quota
        if granted:
            self._quota_used += 1
        self.audit_events.append(
            {"kind": "explore_activation", "bucket": b,
             "raw_eff": raw_eff, "draw": draw, "mult": mult,
             "granted": granted,
             "quota_left": quota - self._quota_used})
        return granted

    # ---------------- drift hook (M5) ----------------
    def on_changepoint(self, bucket, evidence):
        if not self._p("bocpd_coupling"):
            return
        if bucket not in self._stats:
            return
        rho = float(self._p("bocpd_depth"))
        self._stats[bucket][0] *= rho
        self.audit_events.append(
            {"kind": "bocpd_deepen", "bucket": bucket,
             "run_length_drop": evidence.get("run_length_drop"),
             "decay_applied": rho})

    # ---------------- persistence (M1 + P-1) ----------------
    def _rng_state(self):
        return json.loads(json.dumps(
            self._rng.bit_generator.state))

    def snapshot(self):
        return {"schema": SCHEMA, "rule": self._p("rule"),
                "bucket_spec": self._p("bucket_spec"),
                "stats": {b: {"w": s[0], "m": s[1], "v": s[2],
                              "n_raw": s[3]}
                          for b, s in sorted(self._stats.items())},
                "prior_fingerprint": self._prior_fingerprint,
                "quota_used": self._quota_used,
                "quota_win": self._quota_win,
                "rng_state": self._rng_state(),
                "fold_cursor": self._fold_cursor,
                "pending": copy.deepcopy(self._pending),
                "credited_events_tail":
                    copy.deepcopy(self._credited_tail),
                "b1_fallback_count": self._b1_fallbacks,
                "draw_count": self._draw_count,
                "event_stats": {"n": self._estats[0],
                                "w": self._estats[1],
                                "m": self._estats[2],
                                "v": self._estats[3]}}

    def restore(self, blob):
        out = migrate_snapshot(blob)
        if "refusal" in out:
            return out
        st = out.get("stats")
        if not isinstance(st, dict) or any(
                not isinstance(s, dict)
                or not {"w", "m", "v", "n_raw"} <= set(s)
                for s in st.values()):
            return {"refusal": "corrupt preference snapshot: "
                               "stats malformed; refuse and "
                               "rebuild from credit events"}
        self._stats = {b: [float(s["w"]), float(s["m"]),
                           float(s["v"]), int(s["n_raw"])]
                       for b, s in st.items()}
        self._quota_used = int(out.get("quota_used", 0))
        self._quota_win = int(out.get("quota_win", -1))
        self._fold_cursor = int(out.get("fold_cursor", 0))
        self._pending = copy.deepcopy(out.get("pending", []))
        self._credited_tail = copy.deepcopy(
            out.get("credited_events_tail", []))
        self._prior_fingerprint = out.get("prior_fingerprint")
        self._b1_fallbacks = int(out.get("b1_fallback_count", 0))
        self._draw_count = int(out.get("draw_count", 0))
        es = out.get("event_stats") or {}
        self._estats = [int(es.get("n", 0)),
                        float(es.get("w", 0.0)),
                        float(es.get("m", 0.0)),
                        float(es.get("v", 0.0))]
        if out.get("rng_state"):
            self._rng = np.random.default_rng(0)
            self._rng.bit_generator.state = out["rng_state"]
        return {"ok": True}

    def rebuild(self, credit_events):
        """Replay invariant (doc 83 §4.2): deterministic fold over
        the credit-event stream; snapshot == fold(events)."""
        self._stats = {}
        self._fold_cursor = 0
        self._credited_tail = []
        self._estats = [0, 0.0, 0.0, 0.0]
        for ev in credit_events:
            b = ev["bucket"]
            w, m, v, n = self._stats.get(b, (0.0, 0.0, 0.0, 0))
            w2, m2, v2 = core.ema_fold(
                (w, m, v), ev["advantage"], self._p("decay"))
            self._stats[b] = [w2, m2, v2, n + 1]
            self._fold_event(float(ev["advantage"]))
            self._fold_cursor += 1
            self._credited_tail.append(ev)
        del self._credited_tail[:-_TAIL_MAX]

    # ---------------- product read (FR-6) ----------------
    def inspect(self):
        draws = [e for e in self.audit_events
                 if e["kind"] == "preference_draw"][-_TAIL_MAX:]
        return {"stats": {b: {"w": s[0], "m": s[1], "v": s[2],
                              "n_raw": s[3]}
                          for b, s in sorted(self._stats.items())},
                "rule": self._p("rule"),
                "bucket_spec": self._p("bucket_spec"),
                "prior_fingerprint": self._prior_fingerprint,
                "quota_used": self._quota_used,
                "quota_win": self._quota_win,
                "draws_tail": draws,
                "health": {"b1_fallback_count":
                           self._b1_fallbacks},
                "policy_echo": {k: v for k, v in
                                self.policy.items()
                                if k.startswith("preference.")}}

    # ---------------- test/referee hook ----------------
    def _force_event_stats(self, n, w, m, v):
        """Referee hook: set the reference distribution to
        hand-chosen values."""
        self._estats = [int(n), float(w), float(m), float(v)]

    def _force_stats(self, bucket, w, m, v, n_raw):
        """Referee hook: set a bucket to hand-chosen sufficient
        statistics (worksheets need exact known inputs)."""
        self._stats[bucket] = [float(w), float(m), float(v),
                               int(n_raw)]


# ================= lifecycle helpers (L2-facing) =================

def preference_enabled(policy):
    """The seam guard (M2 inertness precedence): the path exists
    only when a non-fixed rule is configured."""
    return (policy or {}).get("preference.rule", "fixed") != "fixed"


def attach_blob(net, blob):
    setattr(net, _BLOB_ATTR, blob)


def read_blob(net):
    return getattr(net, _BLOB_ATTR, None)


def _build(policy, net=None):
    part = GrowthPreference(
        {k: v for k, v in (policy or {}).items()
         if k.startswith("preference.") or k == "seed"})
    if net is not None:
        blob = read_blob(net)
        if blob:
            r = part.restore(blob)
            if "refusal" in r:
                part.audit_events.append(
                    {"kind": "preference_restore_refusal", **r})
    return part


def rank_with_preference(part, raw_scores, ctxs):
    """The seam algebra (doc 83 §4.3 + §4.4): multipliers from
    score_set; adjusted = score * mult if score > 0 else score;
    when NO candidate is positive, consult explore_offer once —
    on True the offered candidate enters ranking with a minimal
    positive epsilon (trial-eligible; downstream flow NORMAL)."""
    mults = part.score_set(ctxs)
    adj = [s * m if s > 0 else s
           for s, m in zip(raw_scores, mults)]
    out = {"mults": mults, "scores_adj": adj,
           "explore_offer": False, "offered_index": None}
    if all(a <= 0 for a in adj) and ctxs:
        best = max(range(len(ctxs)), key=lambda i: adj[i])
        if part.explore_offer(ctxs[best], raw_eff=adj[best]):
            eps = 1e-9
            adj[best] = eps
            out.update({"explore_offer": True,
                        "offered_index": best,
                        "scores_adj": adj})
    return out


def seam_adjust(scope, parts, policy, decision, fw, fd, e_ref):
    """Called by the combiner at tier-2 (the ~10-line seam's
    target): builds the part from the scope blob, sources the b1
    two-horizon slope when configured (M2 SLOPE SOURCING — the
    second, shorter probe burst through the EXISTING pricer;
    cost disclosed on the decision record), adjusts the two arm
    scores, records pref fields, writes the blob back. Returns
    (arm_scores_adjusted dict, explore_flag)."""
    part = _build(policy, scope)
    slope_w = slope_d = None
    spec = part._p("bucket_spec")
    if spec == "b1" and part._p("slope_probes") and parts:
        try:
            steps = int(policy.get("probe_steps", 300))
            short = max(2, steps // 2)
            pol2 = dict(policy)
            pol2["probe_steps"] = short
            pr2 = parts["pricer"].price(scope, pol2)
            if "refusal" not in pr2:
                E = parts["extrapolator"]
                sw = E.fit(np.asarray(pr2["widen_curve"]),
                           seed=int(policy.get("seed", 0)) + 7)
                sd_ = E.fit(np.asarray(pr2["deepen_curve"]),
                            seed=int(policy.get("seed", 0)) + 8)
                if "refusal" not in sw:
                    # gain_long - gain_short = a_short - a_long
                    slope_w = float(sw.get("asymptote", np.nan)
                                    - fw.get("asymptote", np.nan))
                if "refusal" not in sd_:
                    slope_d = float(sd_.get("asymptote", np.nan)
                                    - fd.get("asymptote", np.nan))
                decision["pref_slope_probe_steps"] = short
        except Exception as e:              # audited degradation
            part.audit_events.append(
                {"kind": "slope_probe_failure", "error": str(e)})
    ctx_w = {"move": OP_OMEGA, "slope": slope_w,
             "quoted_gain": e_ref - fw.get("asymptote", np.inf)}
    ctx_d = {"move": OP_DELTA, "slope": slope_d,
             "quoted_gain": e_ref - fd.get("asymptote", np.inf)}
    s_w = e_ref - fw.get("asymptote", np.inf)
    s_d = e_ref - fd.get("asymptote", np.inf)
    out = rank_with_preference(part, [s_w, s_d], [ctx_w, ctx_d])
    decision["pref_mult"] = {OP_OMEGA: out["mults"][0],
                             OP_DELTA: out["mults"][1]}
    decision["pref_e_ref"] = e_ref
    decision["explore_offer"] = out["explore_offer"]
    decision["pref_audit"] = list(part.audit_events)
    attach_blob(scope, part.snapshot())
    return ({OP_OMEGA: out["scores_adj"][0],
             OP_DELTA: out["scores_adj"][1]},
            out["explore_offer"])


def on_adoption(net, decision, policy):
    """Pending-credit registration at adoption (doc 83 §4.4):
    quoted_gain = the decision's own price read for the applied
    arm; e_before = the current energy reference."""
    part = _build(policy, net)
    arm = decision.get("arm")
    prices = decision.get("prices") or {}
    e_ref = decision.get("pref_e_ref", 0.0)
    quoted = decision.get("pref_quoted_gain")
    if quoted is None:
        fit = prices.get(arm) or {}
        quoted = e_ref - fit.get("asymptote", e_ref)
    ctx = {"move": arm, "slope": None}
    part._pending.append(
        {"event_id": f"pref-{len(part._pending)}-"
                     f"{part._fold_cursor}",
         "bucket": part._bucket_audited(ctx), "move": arm,
         "batch": 0, "quoted_gain": float(quoted),
         "e_before": float(e_ref), "windows": []})
    attach_blob(net, part.snapshot())


def on_window(net, energy, policy):
    """K-window credit pipeline (doc 83 §4.4): called at each
    window close with the realized window energy; when a pending
    event completes K windows, the CreditEvent folds in."""
    part = _build(policy, net)
    weights = list(policy.get(
        "preference.credit_weights",
        PREFERENCE_DEFAULTS["preference.credit_weights"]))
    K = len(weights)
    done = []
    for pend in part._pending:
        pend["windows"].append(float(energy))
        if len(pend["windows"]) >= K:
            done.append(pend)
    for pend in done:
        part._pending.remove(pend)
        gains = [pend["e_before"] - e_i          # CUMULATIVE vs
                 for e_i in pend["windows"][:K]]  # fixed base
                                                  # (M3 v1.17)
        ev = assemble_credit_event(
            pend["event_id"], pend["bucket"], pend["move"],
            pend["batch"], pend["quoted_gain"], gains, weights)
        ev["e_before"] = pend["e_before"]
        part.credit(ev)
    attach_blob(net, part.snapshot())


def apply_rollback(current_snap, model_snap, mode):
    """Rollback semantics (doc 83 M1): keep -> lessons survive
    (current table stands); revert -> the with-model snapshot's
    table is restored."""
    if mode == "revert":
        return copy.deepcopy(model_snap)
    return copy.deepcopy(current_snap)


def watch_changepoint(part, bucket, residual_stream, policy):
    """M5 wiring: the EXISTING BOCPD registry part watches the
    advantage residual stream; a detected break triggers
    on_changepoint (w <- rho*w). Returns True iff fired."""
    from .interfaces import get
    det = get("changepoint", "bocpd")
    if isinstance(det, dict):
        return False
    res = det.detect(np.asarray(residual_stream, dtype=float),
                     threshold=0.2, min_len=16, recent=16)
    if "refusal" in res:
        return False
    if not res.get("passed", True):     # break detected
        part.on_changepoint(bucket, {
            "run_length_drop": res.get("p_recent_change")})
        return True
    return False


def fold_prior(ledger_paths, out_path=None, decay=0.98):
    """M8 fleet-prior fold tool (doc 83 §4.7 CLI backend).
    CROSS-WORLD SCALE LAW: advantages are z-standardized WITHIN
    each world_type stratum (population sd) BEFORE pooling; the
    artifact records the strata and their scales. Rows fold in
    file order via the normative L0 ema_fold."""
    import datetime
    rows, shas = [], []
    for path in ledger_paths:
        raw = open(path, "rb").read()
        shas.append(hashlib.sha256(raw).hexdigest())
        for line in raw.decode().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    strata = {}
    for r in rows:
        strata.setdefault(r.get("world_type", "supervised"),
                          []).append(float(r["advantage"]))
    strata_out = {}
    for wt, advs in strata.items():
        a = np.asarray(advs, dtype=float)
        strata_out[wt] = {"mu": float(a.mean()),
                          "sd": float(a.std()),
                          "n": int(a.size)}
    buckets = {}          # {world_type: {bucket: [w,m,v,n]}}
    for r in rows:
        wt = r.get("world_type", "supervised")
        bkt = buckets.setdefault(wt, {})
        w, m, v, n = bkt.get(r["bucket"], (0.0, 0.0, 0.0, 0))
        w2, m2, v2 = core.ema_fold(
            (w, m, v), float(r["advantage"]), decay)  # RAW units
        bkt[r["bucket"]] = (w2, m2, v2, n + 1)
    art = {"schema": PRIOR_SCHEMA,
           "strata": strata_out,
           "buckets": {wt: {b: {"w": s[0], "m": s[1],
                                "v": s[2], "n": s[3]}
                            for b, s in sorted(bkt.items())}
                       for wt, bkt in sorted(buckets.items())},
           "source_sha": shas,
           "created": datetime.date.today().isoformat()}
    if out_path:
        open(out_path, "w").write(json.dumps(art, indent=1,
                                             sort_keys=True))
    return art


from .interfaces import register as _register        # noqa: E402
_register("preference", "preference_v1", GrowthPreference)
