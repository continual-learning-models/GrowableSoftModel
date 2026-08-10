"""SPU realization for attention inner bodies
(DESIGN_GROW_BODY_TYPE 7 stage 2 = T6).

One contract, per-type realization: the OBJECTIVE layer is
unchanged by design (J_inv and the J_disc hinge read only output
vectors); what forks is WHERE the perturbation applies (whole
TOKEN representations, p_mask per token) and HOW the local
gradient is computed (dJ/dz seeded analytically, then
backpropagated through the body's own FD-verified backward).
Discipline carried over verbatim: bounded steps, relative
convergence tolerance, update-norm cap over the FULL parameter
vector, plain SGD only, Adam moments untouched, deterministic
masks seeded by (unit seed, holder step). Exact-zero facts:
d(bout) is EXACTLY zero for both terms (shared readout bias
cancels in every J_inv copy; the batch std is shift-invariant) —
zeroed explicitly, mirroring the reference grad_c convention.
"""
import numpy as np

from .spu_objective import batch_std, draw_masks


def _z_rows(unit, Xs, masks):
    """Masked-copy outputs z_i (K, n) with per-copy caches."""
    rows, packs = [], []
    for mi in masks:
        raw, Pool, caches = unit._forward(Xs, cache=True,
                                          token_mask=mi)
        rows.append(raw[:, 0])
        packs.append((Pool, caches, mi))
    return unit._bk.stack(rows), packs


def _accumulate(unit, dz, pack, G):
    Pool, caches, mi = pack
    g = unit._grads_from_draw(dz.reshape(-1, 1), Pool, caches,
                              token_mask=mi)
    for k in G:
        G[k] += g[k]


def jlocal_attention(unit, Xs, masks, s_entry, policy):
    """J_local = J_inv + gamma * J_disc on an attention body,
    with hand-seeded gradients for every parameter."""
    n = len(Xs)
    K = len(masks)
    z, packs = _z_rows(unit, Xs, masks)
    zbar = z.mean(axis=0)
    diff = z - zbar
    Ji = float((diff ** 2).sum() / (K * n))
    G = {k: unit._bk.zeros_like(v)
         for k, v in unit.P.items()}
    for i in range(K):
        _accumulate(unit, (2.0 / (K * n)) * diff[i], packs[i], G)
    Jd = 0.0
    raw0, Pool0, caches0 = unit._forward(Xs, cache=True)
    z0 = raw0[:, 0]
    s = unit._bk.batch_std(z0)
    gap = policy["spu_rho_floor"] * s_entry - s
    if gap > 0.0:
        Jd = float(gap ** 2)
        dz0 = (-2.0 * gap) * (z0 - z0.mean()) / (n * s)
        gd = unit._grads_from_draw(
            (policy["spu_gamma"] * dz0).reshape(-1, 1), Pool0,
            caches0)
        for k in G:
            G[k] += gd[k]
    G["bout"][:] = 0.0                    # exact zero (both terms)
    return Ji + policy["spu_gamma"] * Jd, Ji, Jd, G


def self_process_attention(unit, X_unit, policy, step_index):
    """The bounded local loop on one eligible attention body —
    the reference loop's structure verbatim, attention forward."""
    Xs = unit._std_x(X_unit)
    rng = np.random.default_rng([int(unit._seed_counter),
                                 int(step_index)])
    raw0 = unit._forward(Xs)
    s_entry = unit._bk.batch_std(raw0[:, 0])
    j_before = None
    steps = clips = disc_hits = 0
    converged = False
    names = sorted(unit.P)
    for _ in range(policy["spu_S_max"]):
        masks = draw_masks(rng, policy["spu_K"], unit.d_in,
                           policy["spu_p_mask"])
        J, Ji, Jd, G = jlocal_attention(unit, Xs, masks, s_entry,
                                        policy)
        if j_before is None:
            j_before = Ji
        if Ji <= policy["spu_tau_rel"] * (s_entry ** 2):
            converged = True
            break
        if Jd > 0:
            disc_hits += 1
        gnorm = float(np.sqrt(sum(float((G[k] ** 2).sum())
                                  for k in names)))
        tnorm = float(np.sqrt(sum(float((unit.P[k] ** 2).sum())
                                  for k in names)))
        eta = policy["spu_eta"]
        if gnorm * eta > policy["spu_clip"] * max(tnorm, 1e-12):
            eta = policy["spu_clip"] * max(tnorm, 1e-12) / gnorm
            clips += 1
        for k in names:                    # plain SGD; Adam untouched
            unit.P[k] = unit.P[k] - eta * G[k]
        steps += 1
    masks = draw_masks(rng, policy["spu_K"], unit.d_in,
                       policy["spu_p_mask"])
    z, _ = _z_rows(unit, Xs, masks)
    diff = z - z.mean(axis=0)
    j_after = float((diff ** 2).sum() / (len(masks) * len(Xs)))
    raw_after = unit._forward(Xs)
    return {"body_type": "attention",
            "steps": steps, "converged": converged,
            "s_after": unit._bk.batch_std(raw_after[:, 0]),
            "j_inv_before": j_before if j_before is not None
            else j_after,
            "j_inv_after": j_after, "s_entry": s_entry,
            "disc_hits": disc_hits, "clips": clips,
            "policy_snapshot": {k: policy[k] for k in
                                ("spu_S_max", "spu_K", "spu_eta",
                                 "spu_clip", "spu_tau_rel")},
            "skip": None}
