"""System facade (IWP4): the ONE public entry — the Model Protocol.

Verb table (SYSTEM_DESIGN V.5; growth-control verbs 59B):
  mutating : create_model, study, practice_update, teach, grow, commit,
             reset, rollback, add_holdout, set_policy,
             deepen, remove_block, remove_grown, plan_run, trial
  read-only: infer, attempts, evaluate, trajectory, growth_report,
             check_drift, discoveries, get_versions, card, list_models,
             attribution, propose, plan_validate, describe, assess
Read-only verbs never change weights, structure, or lineage (tested).
`prune` is RESERVED (deferred; same contract as grow when it lands).
Errors: refusals are OUTCOMES (returned dicts), never exceptions.
"""
from __future__ import annotations

import json

from core._modules import generator  # noqa: F401
from generator.config import Config

from core.wiring import SysFactory
from core.lifecycle import Lifecycle
from core import teaching


class System:
    def __init__(self, config: Config | None = None):
        self.f = SysFactory(config)
        self.lc = Lifecycle(self.f)

    # ---------- mutating ----------
    def create_model(self, model_id, description="", holdout=None,
                     policy=None, substrate=None):
        """substrate: explicit AI choice; None -> transparent auto-default
        by detected data form (stated in the response)."""
        if holdout:
            holdout = self._normalize_holdout_rows(list(holdout))
        from core.substrates import get_substrate
        from core.substrates.forms import detect_form, FORM_DEFAULT
        policy = dict(policy or {})
        r = self._validate_policy_keys(policy)   # 105 FR-1/FR-4
        if r:
            return r
        note = None
        if substrate is not None:
            if get_substrate(substrate) is None:
                return {"refusal": f"unknown substrate: {substrate}"}
            policy["substrate"] = substrate
        elif "substrate" not in policy:
            form = detect_form(holdout or []) or "vector"
            default = FORM_DEFAULT.get(form)
            if default is None:
                return {"refusal": f"data form '{form}' has no registered "
                        "substrate yet"}
            policy["substrate"] = default
            note = (f"substrate auto-selected: '{default}' because the "
                    f"detected data form is '{form}'")
        if get_substrate(policy["substrate"]) is None:
            # review F1: the dict path must hit the same
            # registry check as the explicit argument
            return {"refusal":
                    f"unknown substrate: {policy['substrate']}"}
        nh = policy.get("numeric_head", "point")
        if nh not in ("point", "dist"):
            return {"refusal": f"unknown numeric_head: {nh!r}; "
                    "valid values: 'point' (default), 'dist' "
                    "(numeric-uncertainty head, GSM-I3)"}
        sp = policy.get("substrate_params")
        if sp is not None and not isinstance(sp, dict):
            # review F10: type refusal, never a character soup
            return {"refusal": "substrate_params must be a JSON "
                    f"object, got {type(sp).__name__}"}
        if sp:
            # S9 D-G1: loud validation against the CHOSEN substrate's
            # own constructor signature (refusal doctrine — unknown or
            # non-overridable keys are named, never silently dropped)
            import inspect
            cls = get_substrate(policy["substrate"])
            allowed = set(inspect.signature(cls.__init__).parameters) \
                - {"self", "d_in", "hidden", "mode", "vocab"}
            bad = sorted(set(sp) - allowed)
            if bad:
                birth = sorted(set(bad)
                               & {"d_in", "hidden", "mode", "vocab"})
                msg = (f"substrate_params keys not accepted by "
                       f"substrate '{policy['substrate']}': {bad}; "
                       f"allowed: {sorted(allowed)}")
                if birth:
                    msg += (f"; {birth} are birth-derived from the "
                            "data's own shape and never overridable")
                return {"refusal": msg}
        gp = policy.get("growth_params")
        if gp:
            from reference_net.growthpolicy import (
                DEFAULT_GROWTH_POLICY, EXTENDED_GROWTH_KEYS)
            _rl_keys = {k for k in gp if k.startswith("rl.")
                        or k == "gate.eval_stream"}
            bad = sorted(set(gp) - set(DEFAULT_GROWTH_POLICY)
                         - EXTENDED_GROWTH_KEYS - _rl_keys)
            if bad:
                return {"refusal": "unknown growth_params keys: "
                        f"{bad}; valid keys are those of "
                        "DEFAULT_GROWTH_POLICY plus the "
                        "EXTENDED_GROWTH_KEYS registry (see "
                        "docs/PARAMETER_REFERENCE.md)"}
            if any(k.startswith("preference.") for k in gp):
                # 84 D-4: preference VALUES refused loudly HERE
                # (doc 83 SS4.5: never silent fallback)
                from reference_net.growthpolicy.preference import \
                    validate_preference_policy
                r = validate_preference_policy(gp)
                if r:
                    return r
            if any(k.startswith("rl.") or k == "gate.eval_stream"
                   for k in gp):
                # 84 D-P4: rl.*/gate.eval_stream VALUES refused
                # loudly HERE (both doors; FR-6, the preference
                # precedent). Unknown rl.* keys refuse too.
                from rl_trainer.defaults import validate_rl_policy
                r = validate_rl_policy(gp)
                if r:
                    return r
        spu = {k: v for k, v in policy.items()
               if k.startswith("spu_")}
        if spu:
            # 104 v1.4 door symmetry: the same engine validator
            # that rules at set_policy rules at birth too.
            # review F2: TypeError (non-numeric values) refuses
            # like ValueError — never an escaping exception
            from engine.spu.spu_policy import validate_spu_policy
            try:
                validate_spu_policy(spu)
            except (ValueError, TypeError) as e:
                msg = str(e)
                if not msg.startswith("spu"):
                    msg = f"spu policy refused: {msg}"
                return {"refusal": msg}
        att = {k: v for k, v in policy.items()
               if k.startswith("att_")}
        if att:
            from core.substrates.growable_attention import POLICY
            bad = sorted(set(att) - set(POLICY))
            if bad:
                return {"refusal": f"unknown att_* keys: {bad}; "
                        f"valid: {sorted(POLICY)}"}
        inn = {k: v for k, v in policy.items()
               if k.startswith("innovation_")}
        if inn:
            from core.selfassess import (V27_DEFAULTS,
                                         list_available)
            bad = sorted(set(inn) - set(V27_DEFAULTS))
            if bad:
                return {"refusal": "unknown innovation_* "
                        f"keys: {bad}; valid: "
                        f"{sorted(V27_DEFAULTS)}"}
            m = inn.get("innovation_method")
            if m is not None and m not in list_available():
                return {"refusal": f"unknown innovation_"
                        f"method {m!r}; available: "
                        f"{list_available()}"}
            # review F6: full VALUE validation (ranges/enums),
            # not just the method
            from core.selfassess import _validate_innovation
            try:
                _validate_innovation(inn)
            except ValueError as e:
                return {"refusal": str(e)}
        if holdout:
            # quarantine FIRST among state writes — but only
            # after every refusal check (105 FR-4: a refused
            # create leaves zero state)
            st = self.store(model_id)
            st.register_holdout(holdout)
            st.save()
        out = self.lc.create(model_id, description, holdout, policy)
        out["substrate"] = policy["substrate"]
        if note:
            out["auto_selection"] = note
        return out

    def study(self, model_id, examples, steps=None):
        rows = self._normalize_holdout_rows(list(examples))
        st = self.store(model_id)
        from core.plasticity.store import row_hash
        clean = [r for r in rows if row_hash(r) not in st._quarantine]
        if clean:
            st.add(clean, source="study")
            st.save()
        return self.lc.study(model_id, rows, steps)

    def practice_update(self, model_id, inputs, passed):
        return self.lc.practice_update(model_id, inputs, passed)

    def teach(self, model_id, examples, window=None, recent_n=None):
        """Phase-1 one-call convenience (gated, versioned, full-replay)."""
        examples = self._normalize_holdout_rows(list(examples))
        return self.f.teach(model_id, examples, window=window,
                            recent_n=recent_n)

    def grow(self, model_id, k_nodes=2, hidden=16, body_type=None):
        """Structural growth. body_type (S9.5): explicit body
        family for grown inner bodies (string; None -> the model's
        growth policy / module default 'reference')."""
        return self.lc.grow(model_id, k_nodes, hidden, body_type)

    def grow_attention(self, model_id, layer=0, tol=1.05):
        """One governed attention-growth event (attention-build
        S8; docs/attention-build ALGO P10): evidence -> suggest ->
        held-out gate -> accept/refuse. Refuses loudly unless the
        model's substrate is growable_attention. tol (S9 D-G3) is
        the held-out gate tolerance (default 1.05 = prior
        behavior). MCP/CLI inherit this verb through the existing
        plumbing."""
        return self.lc.grow_attention(model_id, layer, tol)

    def set_attention_selfproc(self, model_id, on, heads=None):
        """Switch the attention self-processing discipline on/off
        per model (S9.3 D-G2); heads (JSON array of h or [l,h])
        optionally restricts it to an allow-set. Refuses loudly on
        non-attention substrates. Warmup/age gates still apply."""
        return self.lc.set_attention_selfproc(model_id, on, heads)

    def commit(self, model_id, note=""):
        return self.lc.commit(model_id, note)

    def reset(self, model_id):
        return self.lc.reset(model_id)

    def rollback(self, model_id, to):
        # 96 E-2 (G-1, doc 83 M1): capture the CURRENT
        # preference table BEFORE the pointer moves; the
        # registered rollback_mode governs its fate — keep
        # (default): lessons survive the rollback; revert:
        # the with-model snapshot's table stands.
        pre = self._preference_open(model_id)
        out = self.f.rollback(model_id, to)
        self.lc.reset(model_id)              # working follows the pointer
        self.lc._log(model_id, "rollback", to=to)
        if isinstance(pre, dict) and "blob" in pre:
            cur_blob = pre["blob"]
            from reference_net.growthpolicy.preference import \
                apply_rollback
            mode = (pre.get("policy") or {}).get(
                "preference.rollback_mode", "keep")
            post = self._preference_open(model_id)
            if isinstance(post, dict) and "blob" in post \
                    and cur_blob is not None:
                final = apply_rollback(cur_blob, post["blob"],
                                       mode)
                if final is not None:
                    self._preference_write(
                        model_id, final,
                        [{"kind": "preference_reset",
                          "mode": "rollback",
                          "by": "policy",
                          "rollback_mode": mode, "to": to}])
        return out

    @staticmethod
    def _normalize_holdout_rows(rows):
        """Entry-layer tolerance (E2E report F2, corrected by
        code facts): the machinery's canonical row key is
        'target' EVERYWHERE (trainer, lifecycle). Accept
        'output' as an alias at the entry layer so operating AIs
        that guess 'output' never hit a KeyError; the machinery
        itself is untouched."""
        out = []
        for r in rows:
            if isinstance(r, dict) and "target" not in r                     and "output" in r:
                r = {**r, "target": r["output"]}
            out.append(r)
        return out

    def add_holdout(self, model_id, examples):
        examples = self._normalize_holdout_rows(list(examples))
        st = self.store(model_id)
        st.register_holdout(examples)          # quarantine FIRST
        st.save()
        return self.f.add_holdout(model_id, examples)

    # ---------- total plasticity (omega / sigma / Phi / memory) ----
    def store(self, model_id):
        """The model's experience store (episodic memory), quarantined
        from every holdout stream."""
        if not hasattr(self, "_stores"):
            self._stores = {}
        if model_id not in self._stores:
            from core.plasticity.store import ExperienceStore
            self._stores[model_id] = ExperienceStore(
                self.lc._mdir(model_id) / "experience",
                seed=self.f.config.seed)
        return self._stores[model_id]

    def widen(self, model_id, container="root", k=2):
        return self.lc.widen(model_id, container=container, k=k)

    def loop(self, model_id, container="root", m=None):
        """lambda — grow a governed directed cycle (iteration as a
        computational resource). Opt-in via loop_enabled."""
        return self.lc.loop(model_id, container=container, m=m)

    def remove_loop(self, model_id, container="root"):
        return self.lc.remove_loop(model_id, container=container)

    def add_feature(self, model_id, name, default=0.0):
        return self.lc.add_feature(model_id, name, default=default)

    # ---------- growth control (59B s2: the served depth/
    # removal/dry-run/plan/trial/inspection surface) ----------
    def deepen(self, model_id, m=None, position=None, recipe=None,
               recipe_params=None, zero_side=None, scope=None,
               force=False):
        """delta — one verb for every organ kind: hosts insert a
        whole layer (position None = END = global deepening);
        reference-network models append/insert a composition
        block; scope=[...] confines the block to a sub-scope
        (advisory warning attached in the response)."""
        return self.lc.deepen(model_id, m=m, position=position,
                              recipe=recipe,
                              recipe_params=recipe_params,
                              zero_side=zero_side, scope=scope,
                              force=force)

    def remove_block(self, model_id, k, force=False):
        return self.lc.remove_block(model_id, k, force=force)

    def remove_grown(self, model_id, key, force=False):
        return self.lc.remove_grown(model_id, key, force=force)

    def propose(self, model_id, move, args=None):
        """Dry-run (FR-14): gates, cost estimate, verdict — no
        mutation."""
        return self.lc.propose(model_id, move, args)

    def plan_validate(self, model_id, plan):
        return self.lc.plan_validate(model_id, plan)

    def plan_run(self, model_id, plan, examples=None,
                 steps_between=10):
        if examples is not None:
            examples = self._normalize_holdout_rows(list(examples))
        return self.lc.plan_run(model_id, plan, examples=examples,
                                steps_between=steps_between)

    def trial(self, model_id, move, args=None, budget_steps=6,
              examples=None):
        if examples is not None:
            examples = self._normalize_holdout_rows(list(examples))
        return self.lc.trial(model_id, move, args,
                             budget_steps=budget_steps,
                             examples=examples)

    def describe(self, model_id):
        return self.lc.describe(model_id)

    def assess(self, model_id):
        return self.lc.assess(model_id)

    def refound(self, model_id, steps=4000, mode="fresh"):
        rows = self.store(model_id).all_rows()
        return self.lc.refound(model_id, rows, steps=steps, mode=mode)

    def self_review(self, model_id, probe_inputs=None):
        """Read-only self-report (S9): retention/LP/saturation/store
        stats + the question list. Changes nothing."""
        from core.plasticity.self_review import self_review as _sr
        return _sr(self, model_id, probe_inputs=probe_inputs)

    def run_self(self, model_id, block_budget, suites=None,
                 allow_growth=False):
        """GRANTED self-study session (S9; consent rules C1-C6): the
        grant is this call; material comes only from the model's own
        store; every action is logged; commit() stays the gate."""
        from core.plasticity.run_self import run_self as _rs
        return _rs(self, model_id, block_budget, suites=suites,
                   allow_growth=allow_growth)

    # ===== growth-preference surface (84 D-4; doc 83 §4.7) =====
    # Two product verbs (inspect/reset) + the offline fold tool
    # + internal open/write plumbing. SMS reaches these ONLY
    # through this facade (the AST fence).

    def _pref_audit_path(self, model_id):
        return self.lc._wdir(model_id) / "preference_audit.jsonl"

    def _rl_audit_write(self, model_id, events):
        """96 E-7 (G-4): INTERNAL plumbing — append P-loop audit
        events (runner drain_audit output) as JSON lines to the
        model's rl_audit.jsonl, mirroring the preference tail
        (default=str keeps numpy payloads replay-readable)."""
        organ, _meta = self._pref_organ(model_id)
        if organ is None:
            return {"refusal": "unknown or untrained model "
                               f"{model_id!r}"}
        f = self.lc._wdir(model_id) / "rl_audit.jsonl"
        with f.open("a") as fh:
            for ev in (events or []):
                fh.write(json.dumps(ev, default=str) + "\n")
        return {"written": True, "n": len(events or [])}

    def _pref_policy_keys(self, model_id):
        gp = (self.lc.policy(model_id).get("growth_params")
              or {})
        keys = {k: v for k, v in gp.items()
                if k.startswith("preference.")}
        keys["seed"] = gp.get("seed", 0)
        return keys

    def _pref_organ(self, model_id):
        try:
            organ, meta = self.lc._load_working(model_id)
        except Exception:
            organ = meta = None
        return organ, meta

    def _preference_open(self, model_id):
        """INTERNAL plumbing: (blob, policy-keys) for the model,
        or a refusal dict."""
        organ, _meta = self._pref_organ(model_id)
        if organ is None:
            return {"refusal": "unknown or untrained model "
                               f"{model_id!r}"}
        from reference_net.growthpolicy import preference as prf
        return {"blob": prf.read_blob(organ),
                "policy": self._pref_policy_keys(model_id)}

    def _preference_write(self, model_id, snapshot,
                         audit_events=None):
        """INTERNAL plumbing: persist the blob with the working
        organ; append audit events to the model's tail."""
        organ, meta = self._pref_organ(model_id)
        if organ is None:
            return {"refusal": "unknown or untrained model "
                               f"{model_id!r}"}
        from reference_net.growthpolicy import preference as prf
        prf.attach_blob(organ, snapshot)
        self.lc._save_working(model_id, organ, meta)
        if audit_events:
            f = self._pref_audit_path(model_id)
            with f.open("a") as fh:
                for ev in audit_events:
                    fh.write(json.dumps(ev, default=str) + "\n")
        return {"written": True}

    def preference_inspect(self, model_id):
        """Growth-preference state dump (doc 83 §4.7):
        READ-ONLY; stats, echoes, prior fingerprint, quota,
        draws tail, health, policy echo, audit tail."""
        opened = self._preference_open(model_id)
        if "refusal" in opened:
            return opened
        from reference_net.growthpolicy import preference as prf
        part = prf.GrowthPreference(opened["policy"])
        if opened["blob"]:
            r = part.restore(opened["blob"])
            if "refusal" in r:
                return r
        ins = part.inspect()
        tail = []
        f = self._pref_audit_path(model_id)
        if f.exists():
            tail = [json.loads(ln) for ln in
                    f.read_text().splitlines()[-20:]]
        tail += list(part.audit_events)
        ins["audit_tail"] = tail[-40:]
        return ins

    def preference_reset(self, model_id):
        """Revert growth preference to INERT (doc 83 §4.7):
        removes every preference.* key (merge-None deletion),
        clears the learned table, audits. Consent gating
        (confirm) is the PRODUCT door's duty (SMS)."""
        organ, meta = self._pref_organ(model_id)
        if organ is None:
            return {"refusal": "unknown or untrained model "
                               f"{model_id!r}"}
        gp = (self.lc.policy(model_id).get("growth_params")
              or {})
        kills = {k: None for k in gp
                 if k.startswith("preference.")}
        if kills:
            self.set_policy(model_id, growth_params=kills)
        from reference_net.growthpolicy import preference as prf
        prf.attach_blob(organ, None)
        self.lc._save_working(model_id, organ, meta)
        import uuid
        audit_id = uuid.uuid4().hex[:12]
        f = self._pref_audit_path(model_id)
        with f.open("a") as fh:
            fh.write(json.dumps(
                {"kind": "preference_reset", "mode": "manual",
                 "by": "verb", "audit_id": audit_id}) + "\n")
        return {"reset": True, "audit_id": audit_id}

    def preference_prior_fold(self, ledgers, out=None):
        """Offline fleet-prior fold tool (doc 83 M8/§4.7)."""
        from reference_net.growthpolicy import preference as prf
        return prf.fold_prior(list(ledgers), out)

    def _validate_policy_keys(self, updates):
        """105 D-1..D-3: top-level model-policy name gate. A
        key passes iff it is a DEFAULT_POLICY key or carries a
        routed prefix (spu_/att_ — their family validators then
        rule on it). Refusal is total and names every offender;
        nothing is stored."""
        from core.lifecycle import (VALID_POLICY_KEYS,
                                    VALID_POLICY_PREFIXES)
        bad = sorted((k for k in updates
                      if not isinstance(k, str)
                      or (k not in VALID_POLICY_KEYS
                          and not k.startswith(
                              VALID_POLICY_PREFIXES))),
                     key=str)
        if bad:
            return {"refusal":
                    f"unknown model-policy keys: {bad}; valid "
                    "keys are DEFAULT_POLICY + spu_*/att_* "
                    "(see docs/PARAMETER_REFERENCE.md)"}
        return None

    def set_policy(self, model_id, **updates):
        """Update the model's policy. ROUTED FAMILY (S9.4 D-N2):
        spu_* keys are engine-validated and installed onto the
        organ at the next study/load (spu_enabled switches the
        SPU seam); substrate_params acts at birth only. Bad spu
        keys/values refuse loudly HERE (before storage)."""
        r = self._validate_policy_keys(updates)   # 105 FR-1
        if r:
            return r
        if not (self.lc._mdir(model_id) / "policy.json").exists():
            # review F7: unknown model is an OUTCOME, not a
            # FileNotFoundError from the storage layer
            return {"refusal": f"unknown model: {model_id!r}; "
                    "create_model first"}
        sub_new = updates.get("substrate")
        if sub_new is not None:
            from core.substrates import get_substrate
            if get_substrate(sub_new) is None:
                # review F1 (life-door sibling)
                return {"refusal":
                        f"unknown substrate: {sub_new}"}
        nh = updates.get("numeric_head")
        if nh is not None and nh not in ("point", "dist"):
            return {"refusal": f"unknown numeric_head: {nh!r}; "
                    "valid values: 'point' (default), 'dist' "
                    "(numeric-uncertainty head, GSM-I3)"}
        spu = {k: v for k, v in updates.items()
               if k.startswith("spu_")}
        if spu:
            from engine.spu.spu_policy import validate_spu_policy
            # review F5: cross-key constraints judge the update
            # MERGED OVER THE STORED spu policy, not defaults
            stored_spu = {k: v for k, v in
                          self.lc.policy(model_id).items()
                          if k.startswith("spu_")}
            try:
                validate_spu_policy({**stored_spu, **spu})
            except (ValueError, TypeError) as e:  # review F2
                msg = str(e)
                if not msg.startswith("spu"):
                    msg = f"spu policy refused: {msg}"
                return {"refusal": msg}
        gp = updates.get("growth_params")
        if gp:
            # S9.5 D-N1: unknown growth keys refused loudly HERE
            # (before storage), named against the module key set
            from reference_net.growthpolicy import (
                DEFAULT_GROWTH_POLICY, EXTENDED_GROWTH_KEYS)
            _rl_keys = {k for k in gp if k.startswith("rl.")
                        or k == "gate.eval_stream"}
            bad = sorted(set(gp) - set(DEFAULT_GROWTH_POLICY)
                         - EXTENDED_GROWTH_KEYS - _rl_keys)
            if bad:
                return {"refusal": "unknown growth_params keys: "
                        f"{bad}; valid keys are those of "
                        "DEFAULT_GROWTH_POLICY plus the "
                        "EXTENDED_GROWTH_KEYS registry (see "
                        "docs/PARAMETER_REFERENCE.md)"}
            if any(k.startswith("preference.") for k in gp):
                # 84 D-4: preference VALUES refused loudly HERE
                # (doc 83 SS4.5: never silent fallback)
                from reference_net.growthpolicy.preference import \
                    validate_preference_policy
                r = validate_preference_policy(gp)
                if r:
                    return r
            if any(k.startswith("rl.") or k == "gate.eval_stream"
                   for k in gp):
                # 84 D-P4: rl.*/gate.eval_stream VALUES refused
                # loudly HERE (both doors; FR-6, the preference
                # precedent). Unknown rl.* keys refuse too.
                from rl_trainer.defaults import validate_rl_policy
                r = validate_rl_policy(gp)
                if r:
                    return r
        att = {k: v for k, v in updates.items()
               if k.startswith("att_")}
        if att:
            # S9.5b D-G5: unknown att_* keys refused loudly, named
            from core.substrates.growable_attention import POLICY
            bad = sorted(set(att) - set(POLICY))
            if bad:
                return {"refusal": f"unknown att_* keys: {bad}; "
                        f"valid: {sorted(POLICY)}"}
        sp = updates.get("substrate_params")
        if sp is not None and not isinstance(sp, dict):
            # review F10: type refusal, never a character soup
            return {"refusal": "substrate_params must be a JSON "
                    f"object, got {type(sp).__name__}"}
        if sp:
            # 104 v1.4 door symmetry (round-2 addendum): the
            # same signature-filtered check the birth door has.
            # review F4: a substrate updated in THIS call wins
            # over the stored one; review F3: an unregistered
            # stored substrate refuses coherently
            import inspect
            from core.substrates import get_substrate
            sub = updates.get("substrate") \
                or self.lc.policy(model_id).get("substrate",
                                                "mlp")
            cls = get_substrate(sub)
            if cls is None:
                return {"refusal": "substrate_params cannot be "
                        f"checked: substrate '{sub}' stored in "
                        "this model's policy is not in the "
                        "current registry"}
            allowed = set(inspect.signature(
                cls.__init__).parameters) \
                - {"self", "d_in", "hidden", "mode", "vocab"}
            bad = sorted(set(sp) - allowed)
            if bad:
                return {"refusal": f"substrate_params keys not "
                        f"accepted by substrate '{sub}': {bad}; "
                        f"allowed: {sorted(allowed)}; note "
                        "substrate_params acts at birth only"}
        inn = {k: v for k, v in updates.items()
               if k.startswith("innovation_")}
        if inn:
            # 104 v1.4 door symmetry + review F6: the FULL
            # innovation validator (ranges/enums), same as the
            # birth door; keys are whitelist-guarded above
            from core.selfassess import _validate_innovation
            try:
                _validate_innovation(inn)
            except ValueError as e:
                return {"refusal": str(e)}
        return self.lc.set_policy(model_id, **updates)

    # ---------- read-only ----------
    def infer(self, model_id, input_, working=False, version=None):
        if version is not None:
            return self.f.infer(model_id, input_, version=version)
        return self.lc.infer(model_id, input_, working=working)

    def predict_dist(self, model_id, input_, working=False,
                     version=None):
        """GSM-I1 (DESIGN_SOFTMODELSYSTEM Part II): surface the
        ALREADY-COMPUTED output distribution that infer()
        truncates to a point/argmax. kind: numeric | categorical
        | none(untrained). The single verb the SMS generation
        plane keys on."""
        if working:
            return self.lc.predict_dist_working(model_id, input_)
        version = version or self.f.registry.active(model_id)
        return self.f.model_manager.predict_dist(
            model_id, version, input_)

    def attempts(self, model_id, inputs):
        return self.lc.attempts(model_id, inputs)

    def evaluate(self, model_id, suites=None, recent_n=None):
        if suites is not None:
            return self.lc.evaluate(model_id, suites)
        return self.f.evaluate(model_id, recent_n=recent_n)

    def trajectory(self, model_id, current_stage=None):
        return teaching.trajectory(self.lc, model_id, current_stage)

    def growth_report(self, model_id, top=6):
        return self.lc.growth_report(model_id, top)

    def check_drift(self, model_id, recent_n=None):
        return self.f.check_drift(model_id, recent_n=recent_n)

    def innovation_report(self, model_id):
        """Doc 29 3.4a: read-only innovation self-assessment
        report. Refusals (doctrine): unknown model; component
        not enabled for this model."""
        try:
            organ, _ = self.lc._load_working(model_id)
        except FileNotFoundError:
            organ = None
        if organ is None:
            return {"refusal": f"unknown model: {model_id}"}
        sa = getattr(organ, "_selfassess", None)
        if sa is None:
            return {"refusal": "innovation self-assessment "
                    "not enabled for this model; set "
                    "innovation_slice_mode in the policy"}
        return sa.innovation_report()

    def discoveries(self, model_id, version=None):
        return self.f.discoveries(model_id, version=version)

    def get_versions(self, model_id):
        return self.f.versions(model_id)

    def card(self, model_id):
        card = self.f.card(model_id)
        card["policy"] = self.lc.policy(model_id)
        card["events"] = len(self.lc.events(model_id))
        return card

    def list_models(self):
        return self.f.list_models()

    def attribution(self, model_id, suites):
        return teaching.attribution(self.lc, model_id, suites)

    # ---------- AI substrate selection (read-only) ----------
    def list_substrates(self):
        from core.substrates import GUIDANCE
        return {"substrates": GUIDANCE}

    def recommend_substrate(self, sample_examples):
        """Advisory only: detect form + probe relation strength; the AI
        decides. No state change."""
        from core.substrates.forms import (detect_form,
                                                   interaction_probe,
                                                   FORM_DEFAULT)
        form = detect_form(sample_examples)
        if form is None:
            return {"refusal": "could not detect a single data form"}
        if form != "vector":
            name = FORM_DEFAULT.get(form)
            return {"data_form": form,
                    "recommendations": [{"substrate": name or "none",
                                         "reason": f"data form is {form}"}]}
        gain = interaction_probe(sample_examples)
        recs = ([{"substrate": "transformer",
                  "reason": f"strong feature interactions (probe gain "
                            f"{gain:.2f}) favor attention"},
                 {"substrate": "mlp",
                  "reason": "cheaper fallback; watch for growth plateaus"}]
                if gain > 0.15 else
                [{"substrate": "mlp",
                  "reason": f"weak interactions (probe gain {gain:.2f}); "
                            "cheapest adequate body"},
                 {"substrate": "transformer",
                  "reason": "upgrade if mlp shows persistent plateaus"}])
        avail = [r for r in recs]
        return {"data_form": form, "interaction_gain": round(gain, 3),
                "recommendations": avail}

    # ---------- the automatic loop (teach -> grow -> gate, chained) ----------
    def run_course(self, model_id, curriculum, policy=None):
        from core.course import run_course
        return run_course(self.lc, model_id, curriculum, policy)
