"""SPU policy: defaults and load-time validation (DESIGN §10).

Separate from the released DEFAULT_GROWTH_POLICY by architecture
ruling — the released policy module is not touched. Validation
follows the certificate-triple discipline: an inconsistent policy
is refused loudly at load, never silently accepted.
"""

DEFAULT_SPU_POLICY = {
    "spu_enabled": False,          # v0 default off (minimal core)
    "spu_scope": "newborn_inner_nets",  # paper vocabulary: newborn
                                   # inner networks (within the
                                   # newborn window); or
                                   # "all_inner_nets" = any age
    "spu_warmup_steps": 100,       # enrollment age: no self-processing
                                   # before this (rough shape first --
                                   # a newborn's weights are still
                                   # violently forming)
    "spu_newborn_steps": 300,      # retirement age: self-processing
                                   # window is [warmup, newborn)
    "spu_S_max": 4,                # max self-processing count per event (owner: e.g. 4 or 8; user-modifiable)
    "spu_K": 4,                    # perturbation copies
    "spu_p_mask": 0.10,            # hidden-unit drop probability
                                   # (owner rulings 07-09/10:
                                   # 0.25 on small subnets is
                                   # amputation; the mechanism
                                   # is PROBABILISTIC and
                                   # self-governing — NO node
                                   # floor exists by design)
    "spu_eta": 0.01,               # plain-SGD local step size
    "spu_clip": 0.05,              # update-norm cap, fraction of ||theta||
    "spu_tau_rel": 0.01,           # RELATIVE convergence tolerance
                                   # (numerical-analysis practice):
                                   # converged when J_inv <=
                                   # tau_rel * s_entry^2, i.e. the
                                   # masked-copy disagreement is
                                   # below this fraction of the
                                   # unit's own output variance
    "spu_gamma": 1.0,              # discrimination weight
    "spu_rho_floor": 0.5,          # relative spread floor
    "spu_n_min": 8,                # minimum batch rows to run
    "spu_every": 1,                # process every m-th evolution step
}

_SCOPES = ("newborn_inner_nets", "all_inner_nets")


def validate_spu_policy(policy=None):
    """Merge with defaults and validate; raise ValueError naming
    every offending key (refused loudly, never silently)."""
    merged = dict(DEFAULT_SPU_POLICY)
    if policy:
        unknown = sorted(set(policy) - set(DEFAULT_SPU_POLICY)
                         - {"spu_objective"})    # B1 optional key
        if unknown:
            raise ValueError(f"unknown spu policy keys: {unknown}")
        merged.update(policy)
    bad = []
    if not isinstance(merged["spu_enabled"], bool):
        bad.append("spu_enabled must be bool")
    if merged["spu_scope"] not in _SCOPES:
        bad.append(f"spu_scope must be one of {_SCOPES}")
    if not (isinstance(merged["spu_newborn_steps"], int)
            and merged["spu_newborn_steps"] >= 1):
        bad.append("spu_newborn_steps must be int >= 1")
    if not (isinstance(merged["spu_warmup_steps"], int)
            and merged["spu_warmup_steps"] >= 0):
        bad.append("spu_warmup_steps must be int >= 0")
    elif merged["spu_warmup_steps"] >= merged["spu_newborn_steps"]:
        bad.append("spu_warmup_steps must be < spu_newborn_steps "
                   "(the window [warmup, newborn) must be nonempty)")
    if not (isinstance(merged["spu_S_max"], int) and merged["spu_S_max"] >= 1):
        bad.append("spu_S_max must be int >= 1")
    if not (isinstance(merged["spu_K"], int) and merged["spu_K"] >= 2):
        bad.append("spu_K must be int >= 2")
    if not (0.0 < merged["spu_p_mask"] < 1.0):
        bad.append("spu_p_mask must be in (0, 1)")
    if not (merged["spu_eta"] > 0):
        bad.append("spu_eta must be > 0")
    if not (merged["spu_clip"] > 0):
        bad.append("spu_clip must be > 0")
    if not (merged["spu_tau_rel"] >= 0):
        bad.append("spu_tau_rel must be >= 0")
    if not (merged["spu_gamma"] >= 0):
        bad.append("spu_gamma must be >= 0")
    if not (0.0 < merged["spu_rho_floor"] < 1.0):
        bad.append("spu_rho_floor must be in (0, 1)")
    if not (isinstance(merged["spu_n_min"], int) and merged["spu_n_min"] >= 1):
        bad.append("spu_n_min must be int >= 1")
    if not (isinstance(merged["spu_every"], int) and merged["spu_every"] >= 1):
        bad.append("spu_every must be int >= 1")

    # B1 (attention-build S7): OPTIONAL objective selector — NOT in
    # DEFAULT_SPU_POLICY (its length is pinned by an existing test);
    # spu_loop reads policy.get("spu_objective", "mask").
    if "spu_objective" in merged and \
            merged["spu_objective"] not in ("mask", "analytic"):
        bad.append("spu_objective must be 'mask' or 'analytic'")
    if bad:
        raise ValueError("spu policy refused: " + "; ".join(bad))
    return merged
