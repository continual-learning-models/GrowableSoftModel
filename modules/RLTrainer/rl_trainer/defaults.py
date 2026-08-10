"""rl.* policy-key defaults (registered at the facade in
D-P4). Defaults are the BLUEPRINT constants — changing any
default is a design change, not a tuning knob."""

RL_DEFAULTS = {
    "rl.lr": 3e-4,
    "rl.gamma": 0.99,
    "rl.lam": 0.95,
    "rl.clip": 0.2,
    "rl.vcoef": 0.5,
    "rl.ent_coef": 0.0,
    "rl.n_epochs": 10,
    "rl.batch_size": 64,
    "rl.max_grad_norm": 0.5,
    "rl.adam_eps": 1e-5,
    "rl.target_kl": None,
    "rl.trainer": "ppo",
    "rl.horizon": 256,
    "rl.eval_episode_budget": 8,
    "rl.eval_window": 8,
    "rl.eval_tol": 1.0,
    "rl.kl_ref_coef": 0.0,
    "rl.regime": "auto",
    "rl.interleave": None,
}

# gate evaluation-stream selector (doc 89 FR-4.1; doc 86 §3.1)
GATE_DEFAULTS = {
    "gate.eval_stream": "labeled_slice",
}

_RL_ENUMS = {"rl.trainer": ("ppo", "grpo"),
             "rl.regime": ("auto", "teach", "rl"),
             "gate.eval_stream": ("labeled_slice",
                                  "eval_episodes")}


def validate_rl_policy(gp):
    """Loud, key-naming validation of rl.* / gate.eval_stream
    values (FR-6; the preference-door precedent: refusal dict
    or None, never silent fallback)."""
    known = set(RL_DEFAULTS) | set(GATE_DEFAULTS)
    for k in sorted(gp):
        if not (k.startswith("rl.") or k in GATE_DEFAULTS):
            continue
        if k not in known:
            return {"refusal": f"unknown key {k!r}; registered "
                    f"rl keys: {sorted(RL_DEFAULTS)}"}
        v = gp[k]
        if v is None:
            # merge-deletion sentinel (72B D-1 [RV F6]):
            # {key: None} REMOVES the key — never typed (R-E2)
            continue
        if k in _RL_ENUMS:
            if v not in _RL_ENUMS[k]:
                return {"refusal": f"{k}={v!r} invalid; valid "
                        f"values: {list(_RL_ENUMS[k])}"}
            continue
        if k == "rl.target_kl":
            if v is not None and not (isinstance(v, (int, float))
                                      and v > 0):
                return {"refusal": f"{k}={v!r} invalid; None or "
                        "a positive number"}
            continue
        if k == "rl.interleave":
            if v is not None and not (
                    isinstance(v, (list, tuple)) and len(v) == 2
                    and all(isinstance(x, int) and x >= 1
                            for x in v)):
                return {"refusal": f"{k}={v!r} invalid; None or "
                        "[teach_count, rl_count] of ints >= 1"}
            continue
        if not isinstance(v, (int, float)) or                 isinstance(v, bool):
            return {"refusal": f"{k}={v!r} invalid; a number "
                    "is required"}
        import math as _math
        if not _math.isfinite(v):
            # O-1: inf would pass any open range bound —
            # non-finite refuses on every numeric key
            return {"refusal": f"{k}={v!r} invalid; a FINITE "
                    "number is required"}
        if k in ("rl.n_epochs", "rl.batch_size", "rl.horizon",
                 "rl.eval_episode_budget", "rl.eval_window") \
                and not isinstance(v, int):
            # R-R3: int-semantic keys refuse non-integers —
            # silent int() truncation is a silent fallback
            return {"refusal": f"{k}={v!r} invalid; an "
                    "INTEGER is required"}
        lo_ok = {"rl.lr": v > 0, "rl.gamma": 0 < v <= 1,
                 "rl.lam": 0 <= v <= 1, "rl.clip": v > 0,
                 "rl.vcoef": v >= 0, "rl.ent_coef": v >= 0,
                 "rl.n_epochs": v >= 1, "rl.batch_size": v >= 1,
                 "rl.max_grad_norm": v > 0, "rl.adam_eps": v > 0,
                 "rl.horizon": v >= 2,
                 "rl.eval_episode_budget": v >= 1,
                 "rl.eval_window": v >= 1,
                 "rl.eval_tol": v > 0,
                 "rl.kl_ref_coef": v >= 0}[k]
        if not lo_ok:
            return {"refusal": f"{k}={v!r} out of range (see "
                    "rl_trainer.defaults registry)"}
    return None
