"""SPU bounded local solver (DESIGN_SPU §4; DEV_PLAN S2).

self_process(unit, X_unit, policy, step_index) runs the juvenile
inner network's local solving loop: plain SGD on J_local over the unit's
own (W1, b1, W2, c) — NEVER touching the Adam moments, blocks,
inner nets, or instrumentation — with a hard step budget, a
convergence stop, and a per-step update-norm cap. Skip logic
lives HERE (single authority). Deterministic: masks come from a
generator seeded by (unit seed, root step index).
"""
import numpy as np

from .spu_objective import (batch_std, draw_masks, forward_chain,
                            jinv_and_grads, jlocal_and_grads)

SKIP_BODY_TYPE = "body_type_unsupported"
SKIP_HOSTS_COMPOSITE = "hosts_composite_nodes"
SKIP_HAS_LOOP = "loop_block_unsupported"
SKIP_ROOT = "root"
SKIP_UNFIT = "scalers_unfit"
SKIP_MATURED = "matured"
SKIP_WARMUP = "warming_up"
SKIP_DISABLED = "disabled"
SKIP_SMALL_BATCH = "batch_below_n_min"
SKIP_EVERY = "not_this_step"


def skip_reason(unit, n, policy, step_index, is_root=False):
    """The single eligibility authority. The candidate unit is an
    INNER NETWORK (a node grown into a subnetwork by rho — the
    paper's composite-node body); it qualifies while it is a
    NEWBORN INNER NETWORK (paper vocabulary) (the network grows for life, so newborn
    subnets keep arriving; each self-processes during its own
    newborn window, then retires — a population that follows the
    growth frontier): hosts no composite nodes of its own (it is
    the refinement frontier — deepened units are ELIGIBLE since
    P2-S2: the chain-aware objective processes them), scalers
    fit, within the newborn window. Atomic nodes are not
    objects and never reach this check. Returns a reason string,
    or None if the loop should run."""
    if not policy["spu_enabled"]:
        return SKIP_DISABLED
    if step_index % policy["spu_every"] != 0:
        return SKIP_EVERY
    if is_root:
        return SKIP_ROOT
    if not hasattr(unit, "W1") and \
            getattr(unit, "BODY_TYPE", None) != "attention":
        # unknown body types are disclosed-skipped (T6 realized the
        # attention spu_step; anything else waits for its own).
        return SKIP_BODY_TYPE
    if getattr(unit, "loop_block", None) is not None:
        # lambda-carrying units: stage-2 realization is a recorded
        # future item (forward-under-mask through K11, backward
        # through K12 — the delta-chain precedent); until then the
        # skip is DISCLOSED, the third use of this staging pattern
        return SKIP_HAS_LOOP
    port = getattr(unit, "_port_site", None)
    if unit.inner or (port is not None and port.bodies):
        # composite via the legacy dict OR the fullwidth port
        # (Growth Interface Reform): either way this unit is no
        # longer the refinement frontier
        return SKIP_HOSTS_COMPOSITE
    if unit._x_mu is None:
        return SKIP_UNFIT
    if policy["spu_scope"] == "newborn_inner_nets":
        if unit._step_count < policy["spu_warmup_steps"]:
            return SKIP_WARMUP
        if unit._step_count >= policy["spu_newborn_steps"]:
            return SKIP_MATURED
    if n < policy["spu_n_min"]:
        return SKIP_SMALL_BATCH
    return None


def self_process(unit, X_unit, policy, step_index):
    """Run the bounded loop on one eligible juvenile inner
    network. X_unit is the batch as this unit receives it, NOT yet
    passed through the unit's own scaler (this function applies
    it, mirroring the released predict path). Returns the event
    dict."""
    if getattr(unit, "BODY_TYPE", None) == "attention":
        from .spu_attention import self_process_attention  # lazy
        return self_process_attention(unit, X_unit, policy,
                                      step_index)
    Xs = unit._bk.standardize(X_unit, unit._x_mu, unit._x_sd)
    blocks = unit.blocks
    theta = [unit.W1, unit.b1, unit.W2, unit.c]
    for blk in blocks:
        theta += [blk["Bin"], blk["bb"], blk["Bout"]]
    names = ["W1", "b1", "W2", "c"]
    rng = np.random.default_rng([int(unit._seed_counter),
                                 int(step_index)])
    _, _, _, _, z0 = unit._bk.forward_chain(
        unit.W1, unit.b1, unit.W2, unit.c, Xs, blocks)
    s_entry = unit._bk.batch_std(z0)
    j_before = None
    steps = clips = disc_hits = 0
    converged = False
    for _ in range(policy["spu_S_max"]):
        masks = draw_masks(rng, policy["spu_K"], unit.H,
                           policy["spu_p_mask"])
        if policy.get("spu_objective", "mask") == "analytic" \
                and not blocks:
            # B1: exact-expectation objective (leaf-only); Ji is
            # the J_inv-ANALOG J~ = (1-1/K)*J_an, so the stop test
            # below carries over unchanged.
            J, Ji, Jd, g = unit._bk.jlocal_an_and_grads(
                unit.W1, unit.b1, unit.W2, unit.c, Xs,
                policy["spu_p_mask"], policy["spu_K"], s_entry,
                policy["spu_gamma"], policy["spu_rho_floor"])
        else:
            J, Ji, Jd, g = unit._bk.jlocal_and_grads(
                unit.W1, unit.b1, unit.W2, unit.c, Xs, masks,
                s_entry, policy["spu_gamma"],
                policy["spu_rho_floor"], blocks=blocks)
        if j_before is None:
            j_before = Ji
        if Ji <= policy["spu_tau_rel"] * (s_entry ** 2):
            converged = True
            break
        if Jd > 0:
            disc_hits += 1
        gnorm = float(np.sqrt(
            sum(float((g[k] ** 2).sum()) for k in names)
            + sum(float((gb[key] ** 2).sum())
                  for gb in g["blocks"] for key in gb)))
        tnorm = float(np.sqrt(sum(float((p ** 2).sum())
                                  for p in theta)))
        eta = policy["spu_eta"]
        if gnorm * eta > policy["spu_clip"] * max(tnorm, 1e-12):
            eta = policy["spu_clip"] * max(tnorm, 1e-12) / gnorm
            clips += 1
        unit.W1 -= eta * g["W1"]
        unit.b1 -= eta * g["b1"]
        unit.W2 -= eta * g["W2"]
        unit.c -= eta * g["c"]
        for blk, gb in zip(blocks, g["blocks"]):
            blk["Bin"] -= eta * gb["Bin"]
            blk["bb"] -= eta * gb["bb"]
            blk["Bout"] -= eta * gb["Bout"]
        steps += 1
    masks = draw_masks(rng, policy["spu_K"], unit.H,
                       policy["spu_p_mask"])
    j_after, _ = unit._bk.jinv_and_grads(
        unit.W1, unit.b1, unit.W2, unit.c, Xs, masks,
        blocks=blocks)
    _, _, _, _, z0_after = unit._bk.forward_chain(
        unit.W1, unit.b1, unit.W2, unit.c, Xs, blocks)
    event_extra = {"blocks": len(blocks)} if blocks else {}
    return {**event_extra,
            "steps": steps, "converged": converged,
            "s_after": unit._bk.batch_std(z0_after),
            "j_inv_before": j_before if j_before is not None
            else j_after,
            "j_inv_after": j_after, "s_entry": s_entry,
            "disc_hits": disc_hits, "clips": clips,
            "policy_snapshot": {k: policy[k] for k in
                                ("spu_S_max", "spu_K", "spu_eta",
                                 "spu_clip", "spu_tau_rel")},
            "skip": None}
