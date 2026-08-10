"""Lifecycle (IWP2/S2.2): WORKING STATE + GATED COMMITS over the factory.

The git-like write model (SYSTEM_DESIGN V.1 / Integration Plan Part V):
- every model has ONE persistent mutable WORKING COPY;
- `study`, `practice`, `grow` are IN-SESSION EDITS to it (continuous
  coupled learning; growth budget-checked against the per-model policy);
- `commit` evaluates working vs. the incumbent version on the model's OWN
  holdout stream (recent slice) and promotes ONLY if strictly better;
- `reset` discards the session back to the incumbent;
- `infer` serves the COMMITTED active version by default (production
  safety); `working=True` probes the session state (VII.1);
- every action is an event in an append-only per-model events.jsonl
  (event-sourced lineage: versions are commit events; the score matrix is
  derived from evaluate events).

Wraps the frozen factory; never edits it (R-SYS2).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from core._modules import generator, reference_net  # noqa: F401
from generator.data import featurize, read_jsonl, recent_slice
from generator.trainer import infer_shape, auto_hidden
from reference_net.trainer import collect_instability

from core.substrate import MSOrgan  # noqa: F401 (legacy alias)
from core.substrates import get_substrate, load_artifact
from core.wiring import SysFactory

TOL = 0.5   # numeric match tolerance — same rule as the frozen evaluator

from core.selfassess import V27_DEFAULTS as _V27  # doc 29 3.5

DEFAULT_POLICY = {
    **_V27,"substrate": "mlp",
                  "numeric_head": "point",   # GSM-I3: "dist" -> numeric_dist
                  "growth_params": {},       # S9.5 D-N1: per-model
                                             # growth-policy overrides
                                             # (life-time; validated vs
                                             # DEFAULT_GROWTH_POLICY)
                  "substrate_params": {},    # S9 D-G1: birth-time
                                             # constructor pass-through
                                             # (signature-filtered;
                                             # d_in/hidden/mode/vocab are
                                             # birth-derived, never here)

                  "max_params_mult": 10, "max_depth": 4,
                  "gate_recent_n": None, "gate_tol": TOL,
                  "practice_reach": 2 * TOL,
                  "consolidation_lr": 1e-3, "study_steps": 200,
                  "attempt_sigma": 0.05, "n_attempts": 8,
                  # param-interface batch S5 (docs/system/22
                  # items 13-27); defaults = the frozen values
                  "widen_sat": 0.5, "uniform_factor": 2.0,
                  "escalate_disposed": 2,
                  "selfstudy_steps": 200, "selfstudy_lp_flat": 0.02,
                  "selfstudy_sat_demand": 0.35,
                  "selfstudy_quiz_n": 96,
                  "selfstudy_var_eps": 0.05,
                  "selfstudy_var_flag": 3.0,
                  "retention_sag": -0.03,
                  "refound_inv_alarm": 0.5,
                  "refound_inv_consecutive": 3,
                  "refound_disposed_alarm": 2,
                  "refound_lp_flat": 1e-3}

# 105 D-1: the top-level model-policy whitelist, ASSEMBLED from
# the live sources (never hand-listed — T-6 guards this).
# Prefixed keys are admitted here and then ruled on by their
# family validators in the facade (spu_* engine, att_* GA).
VALID_POLICY_KEYS = set(DEFAULT_POLICY)
VALID_POLICY_PREFIXES = ("spu_", "att_")


def _is_num(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _substrate_kwargs(pol, cls, default_seed):
    """S9 D-G1: birth-time constructor kwargs from the model policy.
    substrate_params is filtered against the substrate constructor's
    OWN signature; d_in/hidden/mode/vocab are birth-derived from the
    data's shape (self-shaping doctrine) and never overridable.
    Unknown keys were already refused loudly at create_model (facade);
    here the filter is the safety net. seed defaults to the global
    config seed unless the policy carries a per-model one."""
    import inspect
    sp = dict(pol.get("substrate_params") or {})
    allowed = set(inspect.signature(cls.__init__).parameters) \
        - {"self", "d_in", "hidden", "mode", "vocab"}
    kw = {k: v for k, v in sp.items() if k in allowed}
    kw.setdefault("seed", default_seed)
    return kw


def _install_growth(organ, pol):
    """S9.5 D-N1: install per-model growth-policy overrides onto
    the organ. growth_params={} or absent -> organ untouched (the
    attribute never appears; net.py's instance-first reads fall
    back to the module default bit-identically). Unknown keys were
    refused loudly at the facade; the filter here is the safety
    net. Installed at birth AND every _load_working (life-time
    keys, like the SPU seam)."""
    gp = (pol.get("growth_params") or {})
    if not gp or organ is None:
        return
    from reference_net.growthpolicy import (
        DEFAULT_GROWTH_POLICY, EXTENDED_GROWTH_KEYS)
    good = {k: v for k, v in gp.items()
            if k in DEFAULT_GROWTH_POLICY
            or k in EXTENDED_GROWTH_KEYS}
    if good:
        if "eta_target" in good:
            et = good["eta_target"]
            if not isinstance(et, (int, float)) \
                    or isinstance(et, bool) or not 0 < et <= 1:
                raise ValueError(
                    f"eta_target must be in (0, 1]; got {et!r}")
        merged = {**DEFAULT_GROWTH_POLICY, **good}
        # S2 propagation repair (docs/system/22 item 5): install on
        # the WHOLE composite tree, not the root only — inner scopes
        # previously fell back to module defaults for every key.
        # Growth Interface Reform: the tree includes fullwidth
        # port bodies (both couplings carry reference Networks).
        def _walk(o):
            o._growth_policy = merged
            for ch in getattr(o, "inner", {}).values():
                _walk(ch)
            port = getattr(o, "_port_site", None)
            if port is not None:
                for s in port.bodies:
                    _walk(s["body"])
            for site in getattr(o, "_port_sites", {}).values():
                for s in site.bodies:
                    _walk(s["body"])
        _walk(organ)


def _slice_key(rows, sa):
    """Doc 29 3.2: slice identity for a batch. SLICE-PURE
    batches only — a mixed batch returns None (skipped for
    the ledger; G2). Modes: level_tag -> row field 'level';
    task_family -> row field 'family'; length_bucket (and
    any missing field) -> structural fallback on the input
    length."""
    mode = sa.cfg.get("innovation_slice_mode")

    def one(r):
        if mode == "level_tag" and r.get("level") is not None:
            return f"L{r['level']}"
        if mode == "task_family" and r.get("family") is not None:
            return f"F{r['family']}"
        try:
            return f"T{len(np.atleast_2d(r['input']))}"
        except Exception:
            return None
    keys = {one(r) for r in rows}
    if len(keys) != 1:
        return None
    return keys.pop()


def _install_selfassess(organ, pol):
    """Doc 29 2.4: thin product wrapper over the PUBLIC
    core.selfassess.install (validation + mode check live
    there). No innovation keys / slice_mode None -> the
    attribute never appears (OFF-by-default, A4)."""
    if organ is None:
        return
    from core.selfassess import install as _sa_install
    _sa_install(organ, pol)


def _install_att(organ, pol):
    """S9.5b D-G5: install per-model ATTENTION policy. The model
    policy's att_* keys (validated at the facade against the
    module POLICY key set) are merged over the module table and
    installed as organ._att_policy — every in-class read goes
    through the organ's _pol() helper, instance-first. No att_*
    keys -> the attribute never appears (module POLICY serves,
    bit-identically; two models in one process no longer share
    runtime attention knobs — the audit's process-global bug)."""
    att = {k: v for k, v in pol.items() if k.startswith("att_")}
    if not att or organ is None:
        return
    if getattr(organ, "NAME", "") != "growable_attention":
        return
    from core.substrates.growable_attention import POLICY
    good = {k: v for k, v in att.items() if k in POLICY}
    if good:
        organ._att_policy = {**POLICY, **good}


def _install_spu(organ, pol):
    """S9.4 D-N2: install the model policy's spu_* keys onto the
    organ — the product path was SEVERED (set_policy stored the
    keys; nothing ever installed them; the seam slept). Engine
    validator does ALL validation (loud on bad keys/values, incl.
    the optional spu_objective); SPUNetwork-style family objects
    take it through their own set_spu_policy (same validator).
    No spu keys in the policy -> organ untouched (bit-identical
    default path)."""
    spu = {k: v for k, v in pol.items() if k.startswith("spu_")}
    if not spu or organ is None:
        return
    from engine.spu.spu_policy import validate_spu_policy
    v = validate_spu_policy(spu)
    if hasattr(organ, "set_spu_policy"):
        organ.set_spu_policy(v)
    else:
        organ._spu_policy = v


def _site_level(site_path: str) -> int:
    """The level (depth) of the network that owns the site. Growing the
    site creates structure at level + 1. Works for both hosts:
    mlp 'root[j]'=1, '2[j]'=2, '2/1[j]'=3;
    transformer 'layer0/ffn[3]'=1, '...::root[k]'=2, '...::2[k]'=3."""
    if "::" in site_path:
        base, rest = 1, site_path.split("::", 1)[1]
    elif site_path.startswith("layer"):
        return 1
    else:
        base, rest = 0, site_path
    prefix = rest.rsplit("[", 1)[0]
    hops = 0 if prefix == "root" else prefix.count("/") + 1
    return base + 1 + hops


def _gate_tol(pol) -> float:
    """gate_tol policy read (param-interface batch S3, docs/system/22
    item 6); loud validation, no silent adjustment."""
    t = pol.get("gate_tol", TOL)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
        raise ValueError(f"gate_tol must be a number >= 0; got {t!r}")
    return float(t)


def _match(pred, truth, tol=TOL) -> bool:
    try:
        return abs(float(pred) - float(truth)) <= tol
    except (TypeError, ValueError):
        return str(pred).strip() == str(truth).strip()


class Lifecycle:
    def __init__(self, factory: SysFactory | None = None):
        self.f = factory or SysFactory()
        self.reg = self.f.registry

    # ---------- paths / events ----------
    def _mdir(self, mid):
        return self.reg.model_dir(mid)

    def _log(self, mid, event, **fields):
        row = {"ts": round(time.time(), 2), "event": event, **fields}
        with open(self._mdir(mid) / "events.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    def events(self, mid):
        p = self._mdir(mid) / "events.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def policy(self, mid):
        p = self._mdir(mid) / "policy.json"
        return json.loads(p.read_text()) if p.exists() else dict(DEFAULT_POLICY)

    def _merge_growth_params(self, stored, given):
        """72B D-1: growth_params updates MERGE one level deep
        (installed laws survive unrelated writes); a key whose
        value is None is REMOVED (explicit deletion sentinel);
        given None/{} means no changes ([RV F6])."""
        merged = dict(stored or {})
        for k, v in (given or {}).items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        return merged

    def set_policy(self, mid, **updates):
        if "growth_params" in updates:      # 72B D-1 (plan A)
            updates = {**updates, "growth_params":
                       self._merge_growth_params(
                           self.policy(mid).get("growth_params"),
                           updates["growth_params"])}
        pol = {**self.policy(mid), **updates}
        (self._mdir(mid) / "policy.json").write_text(json.dumps(pol, indent=1))
        self._log(mid, "policy", **updates)
        return pol

    # ---------- lifecycle ----------
    def create(self, mid, description="", holdout=None, policy=None):
        from generator.spec import ModelSpec
        out = self.f.create(ModelSpec(mid, description=description,
                                      holdout=holdout or []))
        (self._mdir(mid) / "policy.json").write_text(
            json.dumps({**DEFAULT_POLICY, **(policy or {})}, indent=1))
        self._log(mid, "create", description=description,
                  policy=(policy or {}))    # doc 29 3.4d:
        # parameter history's first frame — replay =
        # DEFAULT_POLICY + this + ordered policy events
        return out

    # ---------- working state ----------
    def _wdir(self, mid):
        d = self._mdir(mid) / "working"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_working(self, mid):
        wdir = self._wdir(mid)
        if (wdir / "msorgan.pkl").exists():
            organ = load_artifact(wdir)
            meta = json.loads((wdir / "meta.json").read_text())
            pol_ = self.policy(mid)
            _install_spu(organ, pol_)                # S9.4 D-N2
            _install_growth(organ, pol_)             # S9.5 D-N1
            _install_att(organ, pol_)                # S9.5b D-G5
            _install_selfassess(organ, pol_)          # doc 29
            return organ, meta
        active = self.reg.active(mid)
        loaded = None if active == "v0" else self.f.model_manager._load(
            mid, active)
        if loaded is None:
            return None, None
        organ, shape, _ = loaded
        organ = copy.deepcopy(organ)
        meta = {"features": shape["features"], "base_version": active,
                "initial_params": organ.n_params()}
        pol_ = self.policy(mid)
        _install_spu(organ, pol_)                    # S9.4 D-N2
        _install_growth(organ, pol_)                 # S9.5 D-N1
        _install_att(organ, pol_)                    # S9.5b D-G5
        _install_selfassess(organ, pol_)              # doc 29
        self._save_working(mid, organ, meta)
        return organ, meta

    def _save_working(self, mid, organ, meta):
        wdir = self._wdir(mid)
        organ.save(wdir)
        (wdir / "meta.json").write_text(json.dumps(meta))

    # ---------- input building (data-form aware) ----------
    @staticmethod
    def _build_X(meta, inputs):
        if meta.get("data_form") == "sequence":
            return np.asarray(list(inputs), float)
        return np.array([featurize(i, meta["features"]) for i in inputs])

    # ---------- in-session edits ----------
    def study(self, mid, examples, steps=None):
        """Supervised in-session learning. Self-shaping continues: on the
        first batch the organ is BORN from the data's shape; later batches
        may grow the vocabulary (new labels -> add_class)."""
        pol = self.policy(mid)
        steps = steps or pol["study_steps"]
        organ, meta = self._load_working(mid)
        rows = list(examples)
        if organ is None:
            from core.substrates.forms import detect_form
            form = detect_form(rows) or "vector"
            cls = get_substrate(pol.get("substrate", "mlp"))
            if cls is None:
                self._log(mid, "study_refused",
                          reason=f"unknown substrate {pol.get('substrate')}")
                return {"refusal": f"unknown substrate: {pol.get('substrate')}"}
            if cls.DATA_FORM != form:
                self._log(mid, "study_refused",
                          reason=f"substrate {cls.NAME} serves "
                                 f"{cls.DATA_FORM}, data is {form}")
                return {"refusal": f"substrate '{cls.NAME}' serves "
                        f"'{cls.DATA_FORM}' data; got '{form}'"}
            # 60A L2 combination whitelist (replaces the per-case
            # sequence+dist guard): the substrate class DECLARES
            # its supported heads; undeclared combinations refuse
            # at the door — default-refuse, never accept-then-crash
            head = "dist" if pol.get("numeric_head") == "dist" \
                else "point"
            allowed = getattr(cls, "SUPPORTED_HEADS", ("point",))
            if head not in allowed:
                self._log(mid, "study_refused",
                          reason=f"substrate {cls.NAME} has no "
                                 f"'{head}' head")
                return {"refusal":
                        f"substrate '{cls.NAME}' does not support "
                        f"numeric_head='{head}'; supported heads: "
                        f"{sorted(allowed)}"}
            if form == "sequence":
                step_dim = len(np.atleast_2d(rows[0]["input"])[0])
                targets = [str(r["target"]) for r in rows]
                numeric = all(_is_num(t) for t in targets)
                mode = "numeric" if numeric else "categorical"
                vocab = None if numeric else sorted(set(targets))
                h = auto_hidden(step_dim,
                                1 if numeric else len(vocab), len(rows))[0]
                organ = cls(step_dim, h, mode=mode, vocab=vocab,
                            **_substrate_kwargs(pol, cls,
                                                self.f.config.seed))
                meta = {"features": [f"step_dim_{step_dim}"],
                        "data_form": "sequence",
                        "base_version": "v0",
                        "initial_params": organ.n_params()}
            else:
                shape = infer_shape(rows)
                feats = shape["features"]
                mode = shape["mode"]
                if (mode == "numeric"
                        and pol.get("numeric_head") == "dist"):
                    mode = "numeric_dist"          # GSM-I3
                h = auto_hidden(len(feats),
                                1 if mode != "categorical"
                                else len(shape.get("vocab", [])),
                                len(rows))[0]
                organ = cls(len(feats), h, mode=mode,
                            vocab=shape.get("vocab"),
                            **_substrate_kwargs(pol, cls,
                                                self.f.config.seed))
                meta = {"features": feats, "data_form": "vector",
                        "base_version": "v0",
                        "initial_params": organ.n_params()}
            _install_spu(organ, pol)               # S9.4 D-N2 (birth)
            _install_growth(organ, pol)            # S9.5 D-N1 (birth)
            _install_att(organ, pol)               # S9.5b D-G5 (birth)
            _install_selfassess(organ, pol)        # doc 29 (birth)
        X = self._build_X(meta, [r["input"] for r in rows])
        if organ.mode in ("numeric", "numeric_dist"):
            y = np.array([[float(r["target"])] for r in rows])
        else:
            y = np.array([str(r["target"]) for r in rows])
            for lbl in sorted(set(y) - set(organ.vocab)):
                organ.add_class(lbl)              # vocabulary growth
        mse = None
        sa = getattr(organ, "_selfassess", None)
        for step_i in range(steps):
            mse = organ.train_step(X, y)
            # FIRST-EXPOSURE-ONLY prequential term (doc 27
            # v1.4 G1): only the first step's pre-update loss
            # on this fresh batch is a code term.
            if sa is not None and step_i == 0:
                key = _slice_key(rows, sa)
                if key is not None:
                    sa.observe(key, float(mse), len(rows))
        if sa is not None:
            for ev in sa.drain_events():
                self._log(mid, ev.pop("event"), **ev)
        self._save_working(mid, organ, meta)
        # training lane for in-session learning (doc 18, audit
        # B1): the studied rows are appended to the model's study
        # store — the same append-only jsonl pattern as
        # events.jsonl — so governance probes (grow_attention)
        # have training-lane data without ever touching the
        # holdout. Teach-path models get theirs from
        # train_store.jsonl; this is the study-path counterpart.
        with open(self._mdir(mid) / "study_store.jsonl", "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        self._log(mid, "study", n=len(rows), steps=steps,
                  loss=round(float(mse), 6))
        return {"loss": mse, "steps": steps, "params": organ.n_params()}

    def predict_dist_working(self, mid, input_):
        """GSM-I1 working-state arm: the full organ's own
        distribution (numeric point; categorical/sequence
        vocab + softmax probabilities)."""
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"kind": "none", "note": "untrained",
                    "state": "working"}
        X = self._build_X(meta, [input_])
        if organ.mode == "numeric_dist":               # GSM-I3
            v, sd = organ.predict_dist(X)
            return {"kind": "numeric_dist",
                    "value": float(v[0]), "std": float(sd[0]),
                    "state": "working"}
        if organ.mode == "numeric":
            return {"kind": "numeric",
                    "value": float(organ.predict(X)[0, 0]),
                    "state": "working"}
        probs = organ.predict_proba(X)[0]
        return {"kind": "categorical",
                "labels": list(organ.vocab),
                "probs": [float(v) for v in probs],
                "state": "working"}

    def attempts(self, mid, inputs):
        """Practice phase 1: attempt 0 = the model's own answer; seeded
        weight-noise variants after it. Verifier lives with the caller."""
        pol = self.policy(mid)
        organ, meta = self._load_working(mid)
        X = self._build_X(meta, inputs)
        outs = [self._answers(organ, X)]
        rng = np.random.default_rng(self.f.config.seed + len(self.events(mid)))
        for _ in range(pol["n_attempts"] - 1):
            pert = organ.perturb(rng, pol["attempt_sigma"])
            outs.append(self._answers(pert, X))
        return np.array(outs).T.tolist()          # (n, A)

    @staticmethod
    def _answers(organ, X):
        if organ.mode in ("numeric", "numeric_dist"):
            return organ.predict(X)[:, 0]
        return np.array(organ.predict_label(X)[0], dtype=object)

    def practice_update(self, mid, inputs, passed):
        """Practice phase 2 (attempt-0 protocol): caller sends a verified
        answer ONLY where the model's own answer failed; within-reach
        filter (numeric); self-anchored targets; SGD consolidation."""
        pol = self.policy(mid)
        organ, meta = self._load_working(mid)
        X = self._build_X(meta, inputs)
        current = self._answers(organ, X)
        keep = []
        for i, a in enumerate(passed):
            if a is None:
                continue
            if organ.mode in ("numeric", "numeric_dist"):
                if abs(float(a) - float(current[i])) <= pol["practice_reach"]:
                    keep.append(i)
            else:
                keep.append(i)
        beyond = sum(1 for a in passed if a is None)
        if keep:
            targets = current.copy()
            for i in keep:
                targets[i] = passed[i]
            y = (targets.reshape(-1, 1).astype(float)
                 if organ.mode in ("numeric", "numeric_dist")
                 else targets)
            for _ in range(10):
                organ.train_step(X, y, sgd_lr=pol["consolidation_lr"])
            self._save_working(mid, organ, meta)
        self._log(mid, "practice", n=len(inputs), fixed=len(keep),
                  beyond_reach=beyond)
        return {"fixed": len(keep), "beyond_reach": beyond}

    def growth_report(self, mid, top=6):
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"candidates": [], "note": "no working state"}
        sites = organ.growth_sites()[:top]
        return {"candidates": [{"site": sp, "instability": round(sc, 4)}
                               for sp, sc in sites],
                # 'depth' = height of the inclusion tree (tree height)
                "params": organ.n_params(), "depth": organ.depth()}

    def grow_attention(self, mid, layer=0, tol=1.05):
        """Attention-growth event (attention-build S8). Two-lane
        contract (doc 18, audit B1): the model's own quarantined
        holdout is the gate's SCORING data (never trained on); the
        probe epoch AND the decide() evidence batch use a recent
        slice of the model's training lanes (train_store +
        study_store); appends the event row to the model's event
        log. tol (S9 D-G3) is the held-out gate tolerance,
        threaded to the driver."""
        organ, meta = self._load_working(mid)
        if getattr(organ, "NAME", "") != "growable_attention":
            self._log(mid, "grow_attention_refused",
                      reason=f"substrate {getattr(organ, 'NAME', '?')}")
            return {"refusal": "grow_attention serves the "
                    "growable_attention substrate only; this model "
                    f"is '{getattr(organ, 'NAME', '?')}'"}
        hold = read_jsonl(self.reg.holdout_path(mid))
        if not hold:
            return {"refusal": "no holdout registered; the gate "
                    "needs quarantined data (add_holdout first)"}
        hx = np.array([featurize(r["input"], meta["features"])
                       for r in hold])
        if not (0 <= layer < organ.L):
            return {"refusal": f"layer {layer} out of range "
                    f"(model has {organ.L})"}
        from core.substrates.growable_attention import \
            attention_grow_event
        events = getattr(organ, "_att_events", None)
        if events is None:
            events = organ._att_events = []
        # two-lane contract (doc 18, audit B1): probe data = recent
        # rows of the model's training lanes — the teach-path
        # train_store plus the in-session study_store, in that
        # (chronological) order. The holdout is scored only. Both
        # lanes empty -> the event refuses loudly.
        srows = recent_slice(
            read_jsonl(self.reg.weights_dir(mid, meta["base_version"])
                       / "train_store.jsonl")
            + read_jsonl(self._mdir(mid) / "study_store.jsonl"),
            max(len(hold), 1))
        px = (np.array([featurize(r["input"], meta["features"])
                        for r in srows]) if srows else None)
        if px is None:
            self._log(mid, "grow_attention_refused",
                      reason="no training-lane rows")
            return {"refusal": "no training-lane rows; the gate "
                    "needs teach/study data for its evidence and "
                    "probe (the holdout is scored only)"}
        if getattr(organ, "mode", "numeric") == "categorical":
            # S9.2 D-G4: the gate's probe metric must match the
            # mode — error rate (loss-like, lower better) for
            # categorical; targets stay labels.
            hy = np.array([str(r["target"]) for r in hold])
            py = (np.array([str(r["target"]) for r in srows])
                  if srows else None)

            def _err_rate(model, hx_, hy_):
                labels, _ = model.predict_label(hx_)
                return float(np.mean(
                    [a != b for a, b in zip(labels, hy_)]))

            row = attention_grow_event(organ, layer, px, hx, hy,
                                       probe_X=px, probe_y=py,
                                       events=events, tol=tol,
                                       metric_fn=_err_rate)
        else:
            hy = np.array([float(r["target"]) for r in hold])
            py = (np.array([float(r["target"]) for r in srows])
                  if srows else None)
            row = attention_grow_event(organ, layer, px, hx, hy,
                                       probe_X=px, probe_y=py,
                                       events=events, tol=tol)
        self._save_working(mid, organ, meta)
        self._log(mid, "grow_attention",
                  op=row.get("event"), verdict=row.get("verdict"),
                  layer=row.get("layer"), t_att=row.get("t"),
                  tol=tol)
        return row

    def set_attention_selfproc(self, mid, on, heads=None):
        """S9.3 D-G2: switch the entropy-band self-processing
        discipline on/off for THIS model (and optionally restrict
        it to an allow-set of heads: entries are h or [l, h]).
        Refuses on non-attention substrates naming the boundary.
        Warmup and head-age gates still apply — the switch
        enables, never bypasses."""
        organ, meta = self._load_working(mid)
        if getattr(organ, "NAME", "") != "growable_attention":
            self._log(mid, "selfproc_refused",
                      reason=f"substrate {getattr(organ, 'NAME', '?')}")
            return {"refusal": "set_attention_selfproc serves the "
                    "growable_attention substrate only; this model "
                    f"is '{getattr(organ, 'NAME', '?')}'"}
        organ._selfproc_on = bool(on)
        allow = None
        if heads is not None:
            allow = set(tuple(e) if isinstance(e, (list, tuple))
                        else e for e in heads)
        organ._selfproc_heads = allow
        self._save_working(mid, organ, meta)
        self._log(mid, "set_attention_selfproc", on=bool(on),
                  heads=(sorted(map(str, allow))
                         if allow is not None else None))
        return {"model": mid, "selfproc": bool(on),
                "heads": (sorted(map(str, allow))
                          if allow is not None else "all")}

    def grow(self, mid, k_nodes=2, hidden=16, body_type=None):
        """In-session structural edit; function-preserving at parity;
        REFUSED beyond the per-model policy budgets (R4).
        body_type (S9.5 D-N1): explicit body family for the grown
        inner bodies (None -> the model's growth policy decides —
        per-model growth_params.grow_body_type, else the module
        default 'reference')."""
        pol = self.policy(mid)
        organ, meta = self._load_working(mid)
        added = k_nodes * (hidden * organ.d_in + hidden * 2 + 1)
        ok_b, cap = self._whole_act_budget(organ, meta, pol,
                                           added)
        if not ok_b:
            self._log(mid, "grow_refused", reason="params budget")
            return {"refusal": "params budget", "cap": cap,
                    "params": organ.n_params()}
        # Depth cap is PER BRANCH (owner-confirmed semantics): a branch
        # at the cap stops; shallower branches keep growing. Refuse only
        # when NO eligible site remains.
        eligible = [(sp, sc) for sp, sc in organ.growth_sites()
                    if _site_level(sp) + 1 <= pol["max_depth"]]
        if not eligible:
            self._log(mid, "grow_refused",
                      reason="depth budget (no eligible branch)")
            return {"refusal": "depth budget", "depth": organ.depth()}
        grown = []
        for site_path, score in eligible:
            if len(grown) >= k_nodes:
                break
            try:
                organ.grow_site(site_path, hidden=hidden,
                                body_type=body_type)
            except ValueError:            # already composite
                continue
            grown.append(site_path)
        self._save_working(mid, organ, meta)
        self._log(mid, "grow", nodes=grown, params=organ.n_params(),
                  depth=organ.depth())
        return {"grown": grown, "params": organ.n_params(),
                "depth": organ.depth()}

    # ---------- measurement ----------
    def evaluate(self, mid, suites, working=True):
        organ, meta = self._load_working(mid)
        tol = _gate_tol(self.policy(mid))
        accs = []
        for s in suites:
            X = self._build_X(meta, s["X"])
            preds = self._answers(organ, X)
            accs.append(float(np.mean([_match(p, t, tol=tol)
                                       for p, t in zip(preds, s["y"])])))
        self._log(mid, "evaluate", names=[s["name"] for s in suites],
                  stage_accs=accs)
        return {"stage_accs": accs}

    def score_matrix(self, mid):
        return [{"t": i, "stage_accs": e["stage_accs"]}
                for i, e in enumerate(self.events(mid))
                if e["event"] == "evaluate"]

    def _holdout_score(self, mid, organ, meta, recent_n):
        rows = recent_slice(read_jsonl(self.reg.holdout_path(mid)), recent_n)
        if not rows:
            return None, 0
        X = self._build_X(meta, [r["input"] for r in rows])
        preds = self._answers(organ, X)
        tol = _gate_tol(self.policy(mid))
        return float(np.mean([_match(p, r["target"], tol=tol)
                              for p, r in zip(preds, rows)])), len(rows)

    # ---------- gate: commit / reset ----------
    # ---------- total-plasticity verbs (omega / sigma / Phi) ----------
    def widen(self, mid, container="root", k=2):
        """omega: outward growth — new units at any scale, exact
        function preservation, budget-checked like grow (R4)."""
        from core.plasticity.net_ops import widen_at
        from reference_net.net import Network
        pol = self.policy(mid)
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}

        if isinstance(organ, Network):
            # trunk + the 60D D-8 zero-extension of every
            # composition/loop block (real params the operation
            # now adds — the budget counts what happens)
            added = k * (organ.d_in + 2) + sum(
                2 * int(np.asarray(b["bb"]).size) * k
                for b in getattr(organ, "blocks", []))
            _lb = getattr(organ, "loop_block", None)
            if _lb is not None:
                added += 2 * int(np.asarray(_lb["b_l"]).size) * k
        else:
            added = k * getattr(organ, "L", 2) * \
                (2 * getattr(organ, "d", 32) + 1)
        ok_b, cap = self._whole_act_budget(organ, meta, pol,
                                           added)
        if not ok_b:
            self._log(mid, "widen_refused", reason="params budget")
            return {"refusal": "params budget", "cap": cap}
        if isinstance(organ, Network):
            out = widen_at(organ, container, k)
        elif hasattr(organ, "widen_ffn"):
            out = (organ.widen_inner(container, k) if "ffn[" in container
                   else organ.widen_ffn(k))
        else:
            self._log(mid, "widen_refused",
                      reason=f"substrate {getattr(organ, 'NAME', '?')} "
                             "has no widen support")
            return {"refusal": "substrate has no widen support; "
                               "create the model with a *_plus substrate"}
        self._save_working(mid, organ, meta)
        self._log(mid, "widen", container=container, k=k,
                  params=organ.n_params())
        return out

    def _resolve_scope(self, organ, container):
        """Hop-walk 'j/k/...' over .inner (the widen_at convention);
        returns (scope, refusal-or-None)."""
        from reference_net.net import Network
        scope = organ
        if container not in ("", "root"):
            for hop in container.split("/"):
                try:
                    scope = scope.grown_body(int(hop))
                    if scope is None:
                        raise KeyError(int(hop))
                except (KeyError, ValueError, AttributeError):
                    return None, {"refusal":
                                  f"no scope at '{container}'"}
        if not isinstance(scope, Network):
            return None, {"refusal":
                          "loop applies to reference scopes; this "
                          "body type has no composition chain"}
        return scope, None

    def loop(self, mid, container="root", m=None):
        """lambda: grow the governed directed cycle at the end of
        the scope's chain (opt-in; DESIGN_LOOP_V2 v2.3)."""
        from reference_net.net import Network
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if not isinstance(organ, Network):
            self._log(mid, "loop_refused", reason="non-reference host")
            return {"refusal": "loop applies to reference scopes; "
                               "this substrate has no composition "
                               "chain (v0 boundary)"}
        scope, bad = self._resolve_scope(organ, container)
        if bad:
            return bad
        try:
            scope.loop(m)
        except ValueError as e:
            self._log(mid, "loop_refused", reason=str(e))
            return {"refusal": str(e)}
        self._save_working(mid, organ, meta)
        mm = int(scope._bk.numel(scope.loop_block["b_l"]))
        self._log(mid, "loop", container=container, m=mm,
                  params=organ.n_params())
        return {"ok": True, "container": container or "root",
                "m": mm, "params_added": 2 * mm * scope.H + mm,
                "note": "exact at application (L_out = 0); the "
                        "model ITERATES at serving on this scope "
                        "(bounded by loop_K_max)",
                "hint": "train, then card/growth_report show "
                        "k_used stats and projections"}

    def remove_loop(self, mid, container="root"):
        """Remove the lambda block (audited structural edit)."""
        from reference_net.net import Network
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if not isinstance(organ, Network):
            return {"refusal": "loop applies to reference scopes"}
        scope, bad = self._resolve_scope(organ, container)
        if bad:
            return bad
        try:
            m = scope.remove_loop()
        except ValueError as e:
            return {"refusal": str(e)}
        self._save_working(mid, organ, meta)
        self._log(mid, "remove_loop", container=container)
        return {"ok": True, "removed_m": m,
                "hint": "the scope reads H_L directly again"}

    # ---------- growth-control served verbs (59B s2 / 59C s1) ----
    # Every verb keeps the loop-precedent shape: load -> validate
    # -> operate -> save -> log -> return dict. Warnings raised by
    # the lib operation are captured into response["warning"] AND
    # re-emitted in-process (59B 2.4: capture suppresses stderr;
    # channels re-emit; same-process callers still get them).

    @staticmethod
    def _run_captured(op):
        """Returns (result-or-raise, warning-texts)."""
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            try:
                out = op()
            finally:
                pending = list(caught)
        for w in pending:
            _w.warn(w.message)
        return out, [str(w.message) for w in pending]

    @staticmethod
    def _attach(res, texts):
        if texts:
            res["warning"] = texts
        return res

    def _gpp(self, organ):
        from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
        return getattr(organ, "_growth_policy",
                       DEFAULT_GROWTH_POLICY)

    def _whole_act_budget(self, organ, meta, pol, act_params):
        """72B D-2: ONE budget formulation (whole-act,
        pre-mutation). Returns (ok, cap)."""
        cap = meta["initial_params"] * pol["max_params_mult"]
        return (organ.n_params() + int(act_params) <= cap,
                cap)

    def _gpp_cap(self, organ, meta, pol):
        """72B D-2 [RV F4]: COPY-inject the ephemeral budget
        cap for the gate table (never mutate the installed /
        shared dict; the key never persists)."""
        return {**self._gpp(organ),
                "params_budget_cap":
                    self._whole_act_budget(organ, meta,
                                           pol, 0)[1]}

    def _rows_xy(self, organ, meta, rows):
        X = self._build_X(meta, [r["input"] for r in rows])
        if organ.mode in ("numeric", "numeric_dist"):
            y = np.array([[float(r["target"])] for r in rows])
        else:
            y = np.array([str(r["target"]) for r in rows])
        return X, y

    def deepen(self, mid, m=None, position=None, recipe=None,
               recipe_params=None, zero_side=None, scope=None,
               force=False):
        """delta: one served verb for EVERY organ kind (59B 2.1).
        Hosts (insert_layer): whole-layer insertion, position
        None -> END = global deepening; scope refused loudly.
        Network family: organ.deepen passthrough (scoped path
        emits the closure advisory). Budget BEFORE the operation;
        force never bypasses budgets (they are the user's own
        policy), only gates (C-5)."""
        pol = self.policy(mid)
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        is_host = hasattr(organ, "insert_layer")
        if is_host and scope is not None:
            self._log(mid, "deepen_refused", reason="host scope")
            return {"refusal":
                    f"substrate {type(organ).__name__} has no "
                    "scoped deepen (hosts insert whole layers; "
                    "scope applies to reference-network models)"}
        # ---- 60D D-7 P-a: AUTO-COMPLIANCE (default rule =
        # width-first, the industry-mature practice). When the
        # post-op ratio would cross the user's floor, the system
        # widens FIRST by exactly the deficit (Network family)
        # or cleanly DEFERS (hosts: no d_model-widen operator
        # exists — named boundary; or aspect_auto="defer").
        # force = the user's explicit override (C-5): the
        # automation steps aside with the gate. Budgets are
        # NEVER bypassed. The organ-seam gate underneath stays
        # as the defense-in-depth backstop.
        auto_widened = None
        gpp_a = self._gpp(organ)
        floor_a = float(gpp_a.get("gate_aspect_min", 0.0))
        auto_a = gpp_a.get("aspect_auto", "widen_first")
        if floor_a > 0 and auto_a != "off" and not force:
            from reference_net.method.gates import _aspect_shape
            w_a, st_a = _aspect_shape(organ)
            depth_after = st_a + 1
            if w_a / max(depth_after, 1) < floor_a:
                if is_host or auto_a == "defer":
                    note = ("aspect_auto: deepen deferred — "
                            f"aspect {w_a}/{depth_after} would "
                            f"cross gate_aspect_min {floor_a}")
                    if is_host:
                        note += ("; hosts' ratio width is "
                                 "d_model and no d_model-widen "
                                 "operator exists yet, so "
                                 "widen_first defers on hosts "
                                 "(recorded boundary)")
                    self._log(mid, "deepen_deferred_aspect",
                              floor=floor_a)
                    return {"deferred_aspect": True,
                            "note": note}
                import math
                k_a = math.ceil(floor_a * depth_after) - w_a
                max_w = int(gpp_a.get("aspect_auto_max_widen",
                                      64))
                if k_a > max_w:
                    self._log(mid, "deepen_refused",
                              reason="aspect_auto max_widen")
                    return {"refusal":
                            f"aspect_auto: required widen k="
                            f"{k_a} exceeds aspect_auto_max_"
                            f"widen {max_w}; raise the cap or "
                            "lower gate_aspect_min"}
                added = k_a * (organ.d_in + 2) + sum(
                    2 * int(np.asarray(b["bb"]).size) * k_a
                    for b in getattr(organ, "blocks", []))
                lb_a = getattr(organ, "loop_block", None)
                if lb_a is not None:
                    added += 2 * int(
                        np.asarray(lb_a["b_l"]).size) * k_a
                # F-4: the widen+deepen ACT is ATOMIC — the
                # budget judges the WHOLE act BEFORE any
                # mutation (deepen cost at the POST-widen
                # width), so a crossing never leaves a
                # widened-but-undeepened state or a log row
                # for an act that will not persist.
                H_new = w_a + k_a
                mm_d = int(m) if m is not None else H_new
                cost_d = (mm_d * H_new + mm_d
                          + H_new * mm_d)
                ok_b, cap_a = self._whole_act_budget(
                    organ, meta, pol, added + cost_d)
                if not ok_b:
                    self._log(mid, "deepen_refused",
                              reason="params budget "
                              "(widen+deepen act)")
                    return {"refusal": "params budget "
                            "(aspect_auto widen+deepen act)",
                            "cap": cap_a,
                            "params": organ.n_params()}
                from core.plasticity.net_ops import widen_at
                widen_at(organ, "root", k_a)
                # log DEFERRED to the success point (F-4:
                # every log row pairs with the save that
                # makes it true)
                auto_widened = k_a
                auto_widen_params = organ.n_params()
        if is_host:
            from reference_net.method.gates import propose
            pos = int(organ.L if position is None else position)
            est = propose(organ, "insert_layer", self._gpp(organ),
                          position=pos)
            cost = est["cost_params"]
        else:
            mm = organ.H if m is None else int(m)
            cost = mm * organ.H + mm + organ.H * mm
        ok_b, cap = self._whole_act_budget(organ, meta, pol,
                                           cost)
        if not ok_b:
            self._log(mid, "deepen_refused",
                      reason="params budget")
            return {"refusal": "params budget", "cap": cap,
                    "params": organ.n_params()}

        def _op():
            if is_host:
                kw = {}
                if recipe is not None:
                    kw["recipe"] = recipe
                if zero_side is not None:
                    kw["zero_side"] = zero_side
                organ.insert_layer(pos, force=force, **kw)
                return {"deepened": {"position": pos},
                        "depth": int(organ.L)}
            organ.deepen(m=m, position=position, recipe=recipe,
                         recipe_params=recipe_params,
                         zero_side=zero_side, scope=scope,
                         force=force)
            return {"deepened": True,
                    "blocks": len(organ.blocks)}
        try:
            info, texts = self._run_captured(_op)
        except (ValueError, TypeError) as e:
            self._log(mid, "deepen_refused", reason=str(e))
            return {"refusal": str(e)}
        self._save_working(mid, organ, meta)
        if auto_widened is not None:      # F-4: logged only at
            self._log(mid, "widen", container="root",
                      k=auto_widened,
                      params=auto_widen_params,
                      aspect_auto=True)   # the persisted act
        self._log(mid, "deepen", params=organ.n_params(),
                  cost_params=cost, position=position,
                  scoped=scope is not None, forced=force)
        res = self._attach({"params": organ.n_params(), **info},
                           texts)
        if auto_widened is not None:      # C-5 visibility: the
            res["auto_widened"] = auto_widened   # automation's
        return res                        # action is never silent

    def remove_block(self, mid, k, force=False):
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if not hasattr(organ, "remove_block"):
            self._log(mid, "remove_block_refused", reason="host")
            return {"refusal":
                    f"substrate {type(organ).__name__} has no "
                    "remove_block; host shrink = version rollback "
                    "(commit/get_versions/rollback)"}
        p0 = organ.n_params()

        def _op():
            return organ.remove_block(int(k), force=force)
        try:
            _, texts = self._run_captured(_op)
        except Exception as e:
            self._log(mid, "remove_block_refused", reason=str(e))
            return {"refusal": f"remove_block({k!r}): "
                    f"{type(e).__name__}: {e}"}
        self._save_working(mid, organ, meta)
        self._log(mid, "remove_block", k=k,
                  params=organ.n_params(), forced=force)
        return self._attach(
            {"params": organ.n_params(),
             "params_delta": organ.n_params() - p0}, texts)

    def remove_grown(self, mid, key, force=False):
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if not hasattr(organ, "remove_grown"):
            self._log(mid, "remove_grown_refused", reason="host")
            return {"refusal":
                    f"substrate {type(organ).__name__} has no "
                    "remove_grown; host shrink = version rollback "
                    "(commit/get_versions/rollback)"}
        p0 = organ.n_params()

        def _op():
            return organ.remove_grown(key, force=force)
        try:
            _, texts = self._run_captured(_op)
        except Exception as e:
            self._log(mid, "remove_grown_refused", reason=str(e))
            return {"refusal": f"remove_grown({key!r}): "
                    f"{type(e).__name__}: {e}"}
        self._save_working(mid, organ, meta)
        self._log(mid, "remove_grown", key=key,
                  params=organ.n_params(), forced=force)
        return self._attach(
            {"params": organ.n_params(),
             "params_delta": organ.n_params() - p0}, texts)

    def propose(self, mid, move, args=None):
        """FR-14 dry-run through the served surface: the lib
        report verbatim; read-only (never saved)."""
        from reference_net.method.gates import propose as _prop
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        pol = self.policy(mid)                 # 72B D-2

        def _op():
            return _prop(organ, move,
                         self._gpp_cap(organ, meta, pol),
                         **(args or {}))
        try:
            rep, texts = self._run_captured(_op)
        except (ValueError, AttributeError) as e:
            return {"refusal": f"propose {move!r}: {e}"}
        self._log(mid, "propose", move=move,
                  would_refuse=rep.get("would_refuse"))
        return self._attach(rep, texts)

    def _load_plan(self, plan):
        if isinstance(plan, str):
            from reference_net.method.gates import load_plan
            return load_plan(plan)
        return plan

    def plan_validate(self, mid, plan):
        """FR-17 validation walk (read-only): per-step proposals
        + cumulative cost; inline dict or parameter-file path."""
        from reference_net.method.gates import validate_plan
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        try:
            plan = self._load_plan(plan)
            rep, texts = self._run_captured(
                lambda: validate_plan(
                    organ, plan,
                    self._gpp_cap(organ, meta,
                                  self.policy(mid))))
        except (ValueError, KeyError, OSError, TypeError) as e:
            return {"refusal": f"plan_validate: {e}"}
        self._log(mid, "plan_validate",
                  steps=len(plan.get("steps", [])))
        return self._attach(rep, texts)

    def plan_run(self, mid, plan, examples=None,
                 steps_between=10):
        """FR-17 rule-driven execution (automatic mode = user
        control by rules). LIB DIALECT (59B 2.3): training rows
        come as `examples`; examples=None -> pure structural
        mode (steps_between forced to 0)."""
        from reference_net.method.gates import run_plan
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if examples is None:
            X = y = None
            steps_between = 0
        else:
            X, y = self._rows_xy(organ, meta, list(examples))
        # ---- 60D D-7 P-b: shift-left compliance PRE-PASS.
        # The plan vocabulary is a CLOSED set with no widen
        # move (execution revision, disclosed in 60D): under
        # widen_first the system widens ONCE UP FRONT to the
        # width the plan's FINAL stage count needs — every
        # intermediate instant is then compliant a fortiori
        # (early width only raises every ratio); defer (and
        # hosts — no d_model-widen operator, named) instead
        # EXCLUDES the crossing depth steps and reports them.
        auto_widened = None
        deferred_steps = None
        pre_note = None
        try:
            plan = self._load_plan(plan)
            gpp_a = self._gpp(organ)
            floor_a = float(gpp_a.get("gate_aspect_min", 0.0))
            auto_a = gpp_a.get("aspect_auto", "widen_first")
            steps_all = plan.get("steps", []) \
                if isinstance(plan, dict) else []
            n_depth = sum(1 for st in steps_all
                          if st.get("move") in
                          ("deepen", "insert_layer"))
            if floor_a > 0 and auto_a != "off" and n_depth:
                from reference_net.method.gates import \
                    _aspect_shape
                w_a, st_now = _aspect_shape(organ)
                is_host = hasattr(organ, "insert_layer")
                if not is_host and auto_a == "widen_first":
                    import math
                    k_a = max(0, math.ceil(
                        floor_a * (st_now + n_depth)) - w_a)
                    if k_a > 0:
                        max_w = int(gpp_a.get(
                            "aspect_auto_max_widen", 64))
                        if k_a > max_w:
                            self._log(mid, "plan_refused",
                                      reason="aspect max_widen")
                            return {"refusal":
                                    "aspect_auto: required "
                                    f"widen k={k_a} exceeds "
                                    "aspect_auto_max_widen "
                                    f"{max_w}"}
                        pol_a = self.policy(mid)
                        added = k_a * (organ.d_in + 2) + sum(
                            2 * int(np.asarray(b["bb"]).size)
                            * k_a for b in
                            getattr(organ, "blocks", []))
                        lb_a = getattr(organ, "loop_block",
                                       None)
                        if lb_a is not None:
                            added += 2 * int(np.asarray(
                                lb_a["b_l"]).size) * k_a
                        # 72B D-2 (F-4 at the PLAN act):
                        # the widen and every UNCONDITIONAL
                        # depth step it enables are ONE act —
                        # judged whole, before any mutation.
                        # Conditional (when-gated) steps are
                        # priced per-step inside run_plan.
                        H_new_p = w_a + k_a
                        depth_cost = 0
                        for st_p in steps_all:
                            if st_p.get("move") in (
                                    "deepen", "insert_layer") \
                                    and st_p.get("when") \
                                    is None:
                                mm_p = int(st_p.get(
                                    "args", {}).get("m")
                                    or H_new_p)
                                depth_cost += (mm_p * H_new_p
                                               + mm_p
                                               + H_new_p
                                               * mm_p)
                        ok_p, cap_a = self._whole_act_budget(
                            organ, meta, pol_a,
                            added + depth_cost)
                        if not ok_p:
                            self._log(mid, "plan_refused",
                                      reason="aspect budget")
                            return {"refusal":
                                    "aspect_auto widen "
                                    "refused: params budget "
                                    "(whole plan act)",
                                    "cap": cap_a}
                        from core.plasticity.net_ops import \
                            widen_at
                        widen_at(organ, "root", k_a)
                        # F-4: log deferred to the success
                        # point (after run_plan + save)
                        auto_widened = k_a
                        auto_widen_params = organ.n_params()
                else:                    # defer, or any host
                    kept, excl, cur = [], [], st_now
                    for i, st in enumerate(steps_all):
                        if st.get("move") in ("deepen",
                                              "insert_layer"):
                            if not st.get("args", {}).get(
                                    "force") and \
                                    w_a / max(cur + 1, 1) \
                                    < floor_a:
                                excl.append(i)
                                continue
                            cur += 1
                        kept.append(st)
                    if excl:
                        plan = {**plan, "steps": kept}
                        deferred_steps = excl
                        pre_note = ("aspect_auto: crossing "
                                    "depth steps deferred "
                                    f"(gate_aspect_min "
                                    f"{floor_a})")
                        if is_host:
                            pre_note += (
                                "; hosts' ratio width is "
                                "d_model and no d_model-widen "
                                "operator exists yet "
                                "(recorded boundary)")
            rep, texts = self._run_captured(
                lambda: run_plan(organ, plan,
                                 self._gpp_cap(
                                     organ, meta,
                                     self.policy(mid)),
                                 X, y,
                                 steps_between=steps_between))
        except (ValueError, KeyError, OSError, TypeError) as e:
            self._log(mid, "plan_refused", reason=str(e))
            return {"refusal": f"plan_run: {e}"}
        self._save_working(mid, organ, meta)
        if auto_widened is not None:     # F-4: logged only at
            self._log(mid, "widen", container="root",
                      k=auto_widened,
                      params=auto_widen_params,
                      aspect_auto=True)  # the persisted act
        self._log(mid, "plan", events=len(rep.get("events", [])),
                  halted=rep.get("halted"),
                  params=organ.n_params())
        res = self._attach(rep, texts)
        if auto_widened is not None:     # C-5 visibility
            res["auto_widened"] = auto_widened
        if deferred_steps is not None:
            res["deferred_aspect_steps"] = deferred_steps
        if pre_note is not None:
            res["note"] = pre_note
        return res

    def trial(self, mid, move, args=None, budget_steps=6,
              examples=None):
        """FR-15 bounded probe through the served surface:
        apply (move,args) via the ONE move vocabulary, train at
        most budget_steps, measure, UNCONDITIONAL rollback.
        Requires `examples` (the lib data dialect)."""
        from reference_net.growth_store import trial as _trial
        from reference_net.method.gates import apply_move
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if examples is None:
            return {"refusal": "trial requires `examples` "
                    "(training rows) at the lib tier; the "
                    "measurement needs data to train on"}
        X, y = self._rows_xy(organ, meta, list(examples))
        # 72B D-2: whole-act budget BEFORE any mutation (the
        # probe would roll back, but a beyond-budget probe
        # must refuse like every other entrance — seam
        # parity). GROWTH moves only; any propose error here
        # falls through so the legacy path keeps its exact
        # loud-refusal text (review fix: the first cut called
        # propose unguarded and an unknown move CRASHED
        # instead of refusing).
        if move in ("deepen", "grow", "insert_layer",
                    "grow_site"):
            from reference_net.method.gates import \
                propose as _prop
            try:
                _pre = _prop(organ, move,
                             self._gpp_cap(organ, meta,
                                           self.policy(mid)),
                             **(args or {}))
            except (ValueError, AttributeError, KeyError,
                    TypeError):
                _pre = {}
            _gb = _pre.get("gates", {}).get("G-BUDGET")
            if _gb is not None and not _gb["met"]:
                self._log(mid, "trial_refused",
                          reason="params budget")
                return {"refusal": "params budget",
                        "cap": _gb["cap"],
                        "post": _gb["post"]}
        try:
            rep, texts = self._run_captured(
                lambda: _trial(organ,
                               lambda h: apply_move(h, move,
                                                    args),
                               X, y, budget_steps))
        except (ValueError, KeyError, TypeError,
                AttributeError) as e:
            self._log(mid, "trial_refused", reason=str(e))
            return {"refusal": f"trial {move!r}: {e}"}
        self._save_working(mid, organ, meta)
        self._log(mid, "trial", move=move,
                  realized_gain=rep.get("realized_gain"),
                  steps_run=rep.get("steps_run"))
        return self._attach(rep, texts)

    def describe(self, mid):
        """Anatomy report (READ-ONLY, JSON-safe)."""
        from reference_net.instrument import describe as _desc
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        rep = _desc(organ)
        self._log(mid, "describe")
        return rep

    def assess(self, mid):
        """Dynamics instrument panel (READ-ONLY, JSON-safe)."""
        from reference_net.method.gates import assess_growth
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        rep = assess_growth(organ)
        self._log(mid, "assess")
        return rep

    def add_feature(self, mid, name, default=0.0):
        """sigma: a NEW input feature enters with zero weights (exact
        preservation); participation is earned by training."""
        from core.plasticity.net_ops import add_feature_net
        from reference_net.net import Network
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if meta.get("data_form") == "sequence":
            return {"refusal": "sigma on sequence hosts is deferred; "
                               "use refound"}
        if not isinstance(organ, Network):
            self._log(mid, "add_feature_refused",
                      reason="host without sigma support")
            return {"refusal": "this host has no live add_feature yet; "
                               "use refound (rebuilds at the new schema)"}
        if name in meta["features"]:
            return {"refusal": f"feature '{name}' already exists"}
        out = add_feature_net(organ, default=default)
        meta["features"] = list(meta["features"]) + [name]
        self._save_working(mid, organ, meta)
        self._log(mid, "add_feature", name=name, default=default,
                  d_in=out["d_in"])
        return {**out, "features": meta["features"]}

    def refound(self, mid, rows, steps=4000, mode="fresh"):
        """Phi: rebuild the WORKING organ from accumulated experience
        rows (real data, never the old model's answers). Promotion
        stays with the normal commit gate — Phi never bypasses it."""
        import numpy as np
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "no working state"}
        if not rows:
            return {"refusal": "no experience rows to refound from"}
        feats = sorted({k for r in rows for k in r["input"]})
        X = np.array([[float(r["input"].get(f, 0.0)) for f in feats]
                      for r in rows])
        y = np.array([[float(r["target"])] for r in rows])
        from core.plasticity.refound import size_from_store
        cls = type(organ)                      # same substrate class
        cand = cls(len(feats), size_from_store(len(feats), len(rows)),
                   mode=getattr(organ, "mode", "numeric"),
                   vocab=getattr(organ, "vocab", None) or None,
                   seed=self.f.config.seed + 1)
        if mode == "shrink_perturb":
            import numpy as _np
            rng = _np.random.default_rng(self.f.config.seed + 2)
            h0 = min(organ.H, cand.H)
            d0 = min(organ.d_in, cand.d_in)
            cand.W1[:h0, :d0] = (0.4 * organ.W1[:h0, :d0]
                                 + rng.normal(0, 0.01, (h0, d0)))
        for _ in range(steps):
            cand.train_step(X, y)
        meta2 = dict(meta)
        meta2["features"] = feats
        meta2["metamorphosis"] = {"from_params": organ.n_params(),
                                  "to_params": cand.n_params(),
                                  "mode": mode, "rows": len(rows)}
        self._save_working(mid, cand, meta2)
        self._log(mid, "refound", mode=mode, rows=len(rows),
                  old_params=organ.n_params(),
                  new_params=cand.n_params())
        return {"candidate_params": cand.n_params(),
                "features": feats,
                "note": "candidate is in WORKING state; commit() gates "
                        "promotion against the incumbent as usual"}

    def commit(self, mid, note=""):
        pol = self.policy(mid)
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"refusal": "nothing to commit"}
        rn = pol["gate_recent_n"]
        w_score, n = self._holdout_score(mid, organ, meta, rn)
        if w_score is None:
            return {"refusal": "no holdout to gate on"}
        live = self.f.evaluate(mid, recent_n=rn)["metric"]
        if w_score <= live:
            self._log(mid, "commit_rejected", working=w_score, live=live)
            return {"promoted": False, "working": w_score, "live": live}
        version = self.reg.next_version(mid)
        out_dir = self.reg.weights_dir(mid, version)
        organ.save(out_dir)
        shape = {"features": meta["features"],
                 "data_form": meta.get("data_form", "vector"),
                 **organ.shape_record()}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "shape.json").write_text(json.dumps(shape))
        self.reg.add_version(mid, version, parent=meta["base_version"],
                             note=note or "session commit")
        self.reg.set_active(mid, version)
        self.reg.record_score(mid, version, w_score)
        self.f.model_manager.invalidate_cache(mid)
        meta["base_version"] = version
        self._save_working(mid, organ, meta)
        self._log(mid, "commit", version=version, score=w_score, live=live,
                  note=note)
        return {"promoted": True, "version": version, "score": w_score,
                "live_before": live}

    def reset(self, mid):
        wdir = self._wdir(mid)
        for f in ("msorgan.pkl", "meta.json"):
            p = wdir / f
            if p.exists():
                p.unlink()
        self._log(mid, "reset")
        return {"reset": True}

    # ---------- serving ----------
    def infer(self, mid, input_, working=False):
        if not working:
            return self.f.infer(mid, input_)      # committed version only
        organ, meta = self._load_working(mid)
        if organ is None:
            return {"output": None, "confidence": 0.0, "note": "untrained"}
        X = self._build_X(meta, [input_])
        if organ.mode in ("numeric", "numeric_dist"):
            return {"output": float(organ.predict(X)[0, 0]),
                    "confidence": None, "state": "working"}
        labels, conf = organ.predict_label(X)
        return {"output": labels[0], "confidence": round(float(conf[0]), 4),
                "state": "working"}
