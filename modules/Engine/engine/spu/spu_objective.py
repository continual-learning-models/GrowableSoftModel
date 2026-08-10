"""SPU objective: masks, J_inv, J_disc, and hand-derived
gradients (DESIGN_SPU v1.3 §3; DEV_PLAN S1).

Pure functions — no Network coupling, no mutation, no global RNG.
X is the leaf's ALREADY-STANDARDIZED input (the caller mirrors
the released input path). Reuses the released gelu/gelu_d
(imported, not modified).

Derivative facts used (hand derivation, FD-verified in tests):
- dJ_inv/dz_i = (2/(K n)) (z_i - z_bar): the mean-coupling term
  vanishes because sum_i (z_i - z_bar) = 0.
- grad_c(J_inv) = 0 exactly (all copies share c).
- std is shift-invariant, so grad_c(J_disc) = 0 exactly.
- At an exact constant output (s = 0) J_disc is flat: the term
  PREVENTS drifting into collapse (gradient pushes back while
  s > 0); it does not rescue from an exact constant — the gate
  is the system-level line (DESIGN §5).
"""
import numpy as np

from ..primitives import gelu, gelu_d

_EPS = 1e-12


def draw_masks(rng, K, H, p_mask):
    """(K, H) 0/1 float masks; each hidden unit dropped
    independently with probability p_mask; same mask for every
    batch row of one copy. Deterministic under the caller's rng.

    REJECTION SAMPLING of the truncated Bernoulli (owner ruling
    2026-07-10): an all-masked draw destroys the computation
    rather than perturbing it (the copy degenerates to its bias
    constant and poisons the consistency mean), so the target
    law is conditioned on >=1 surviving node; redraw is the
    textbook-exact sampler for it, keeps K fixed (no downstream
    empty-set corner), and expects ~1.0 draws at sane policies.
    The 100-try fuse only makes an unbounded loop structurally
    impossible; its biased fallback (keep one uniformly RANDOM
    node) fires with probability (p^H)^100 — below
    hardware-error scale even at pathological policies."""
    m = (rng.random((K, H)) >= p_mask).astype(float)
    for k in range(K):
        tries = 0
        while m[k].sum() == 0.0:
            m[k] = (rng.random(H) >= p_mask).astype(float)
            tries += 1
            if tries >= 100:
                m[k, int(rng.integers(H))] = 1.0
                break
    return m


def forward_parts(W1, b1, W2, c, X):
    """Shared forward pieces: pre-activations A (n,H), hidden h
    (n,H), unperturbed scalar output z0 (n,)."""
    A = X @ W1.T + b1
    h = gelu(A)
    z0 = h @ W2[0] + c[0]
    return A, h, z0


def chain_apply(blocks, H0):
    """Apply the unit's own composition blocks to a hidden batch
    (the released block math, replicated as a pure function):
    H_k = H_{k-1} + gelu(H_{k-1} Bin^T + bb) Bout^T. Returns the
    final hidden batch and per-block caches for the backward."""
    H, caches = H0, []
    for blk in blocks:
        P = H @ blk["Bin"].T + blk["bb"]
        G = gelu(P)
        caches.append((H, P, G))
        H = H + G @ blk["Bout"].T
    return H, caches


def chain_backward(blocks, caches, dH_out):
    """Backward through the block chain. Returns the gradient at
    the chain input and one grad dict per block."""
    gblocks = [None] * len(blocks)
    dH = dH_out
    for k in range(len(blocks) - 1, -1, -1):
        Hk, P, G = caches[k]
        blk = blocks[k]
        dBout = dH.T @ G
        dP = (dH @ blk["Bout"]) * gelu_d(P)
        gblocks[k] = {"Bin": dP.T @ Hk, "bb": dP.sum(axis=0),
                      "Bout": dBout}
        dH = dH + dP @ blk["Bin"]
    return dH, gblocks


def forward_chain(W1, b1, W2, c, X, blocks=()):
    """Chain-aware unperturbed forward: A, base hidden h, chain
    output HL, scalar output z0. With no blocks this executes the
    exact Stage-A operations (HL is h itself)."""
    A = X @ W1.T + b1
    h = gelu(A)
    HL, caches = chain_apply(blocks, h)
    z0 = HL @ W2[0] + c[0]
    return A, h, HL, caches, z0


def _zero_block_grads(blocks):
    return [{"Bin": np.zeros_like(b["Bin"]),
             "bb": np.zeros_like(b["bb"]),
             "Bout": np.zeros_like(b["Bout"])} for b in blocks]


def batch_std(z0):
    """Population std with an epsilon guard (n=1 -> 0.0)."""
    return float(np.sqrt(np.mean((z0 - z0.mean()) ** 2) + _EPS))


def jinv_and_grads(W1, b1, W2, c, X, masks, blocks=()):
    """J_inv and hand gradients w.r.t. (W1, b1, W2, c) and, when
    the unit carries composition blocks, each block's parameters
    (grads["blocks"], one dict per block).

    z_i = chain(m_i o h) W2' + c ;
    J_inv = (1/(K n)) sum_i ||z_i - z_bar||^2.
    The blocks=() path executes the Stage-A code verbatim
    (bitwise regression-guarded)."""
    if blocks:
        return _jinv_chain(W1, b1, W2, c, X, masks, blocks)
    n = len(X)
    K = len(masks)
    A, h, _ = forward_parts(W1, b1, W2, c, X)
    w_eff = masks * W2[0]                      # (K,H) effective weights
    z = h @ w_eff.T + c[0]                     # (K written as columns) -> (n,K)
    z = z.T                                    # (K,n)
    zbar = z.mean(axis=0)                      # (n,)
    diff = z - zbar                            # (K,n)
    J = float((diff ** 2).sum() / (K * n))
    G = (2.0 / (K * n)) * diff                 # dJ/dz_i rows (K,n)
    # W2: dJ/dW2[k] = sum_i m_i[k] * (G_i . h[:,k])
    gW2 = ((G @ h) * masks).sum(axis=0)[None, :]        # (1,H)
    gc = np.zeros(1)                                     # exact zero
    # h: dJ/dh[r,k] = sum_i G_i[r] * m_i[k] * W2[k]
    dh = G.T @ w_eff                                     # (n,H)
    dA = dh * gelu_d(A)
    gW1 = dA.T @ X                                       # (H,d)
    gb1 = dA.sum(axis=0)
    return J, {"W1": gW1, "b1": gb1, "W2": gW2, "c": gc,
               "blocks": []}


def _jinv_chain(W1, b1, W2, c, X, masks, blocks):
    """Chain branch of jinv_and_grads: each masked copy passes
    through the block chain separately; gradients flow back
    through the chain into every parameter (grad_c stays exactly
    zero — c is shared by all copies and cancels in diff)."""
    n = len(X)
    K = len(masks)
    A = X @ W1.T + b1
    h = gelu(A)
    z_rows, cache_list = [], []
    for i in range(K):
        HL, caches = chain_apply(blocks, masks[i][None, :] * h)
        z_rows.append(HL @ W2[0] + c[0])
        cache_list.append((HL, caches))
    z = np.stack(z_rows)                       # (K,n)
    zbar = z.mean(axis=0)
    diff = z - zbar
    J = float((diff ** 2).sum() / (K * n))
    G = (2.0 / (K * n)) * diff                 # dJ/dz_i (K,n)
    gW2 = np.zeros_like(W2)
    gblocks = _zero_block_grads(blocks)
    dh = np.zeros_like(h)
    for i in range(K):
        HL, caches = cache_list[i]
        gW2 += (G[i] @ HL)[None, :]
        dHL = np.outer(G[i], W2[0])
        dH0, gb = chain_backward(blocks, caches, dHL)
        dh += dH0 * masks[i][None, :]
        for k in range(len(blocks)):
            for key in gb[k]:
                gblocks[k][key] += gb[k][key]
    dA = dh * gelu_d(A)
    return J, {"W1": dA.T @ X, "b1": dA.sum(axis=0), "W2": gW2,
               "c": np.zeros(1), "blocks": gblocks}


def jdisc_and_grads(W1, b1, W2, c, X, s_entry, rho_floor,
                    blocks=()):
    """J_disc = max(0, rho_floor*s_entry - s)^2 and gradients.

    s = batch std of the chain-aware z0; shift-invariance makes
    grad_c exactly 0; inactive hinge (or degenerate s_entry) ->
    zero everywhere. The blocks=() path executes the Stage-A code
    verbatim."""
    zeros = {"W1": np.zeros_like(W1), "b1": np.zeros_like(b1),
             "W2": np.zeros_like(W2), "c": np.zeros_like(c),
             "blocks": _zero_block_grads(blocks)}
    n = len(X)
    if blocks:
        A, h, HL, caches, z0 = forward_chain(W1, b1, W2, c, X,
                                             blocks)
    else:
        A, h, z0 = forward_parts(W1, b1, W2, c, X)
        HL = h
    s = batch_std(z0)
    gap = rho_floor * s_entry - s
    if gap <= 0.0:
        return 0.0, zeros
    J = float(gap ** 2)
    # dJ/ds = -2*gap ; ds/dz0 = (z0 - mean)/(n*s)
    dz0 = (-2.0 * gap) * (z0 - z0.mean()) / (n * s)      # (n,)
    gW2 = (dz0 @ HL)[None, :]
    gc = np.zeros(1)                                     # exact zero
    gblocks = _zero_block_grads(blocks)
    if blocks:
        dHL = np.outer(dz0, W2[0])
        dh, gblocks = chain_backward(blocks, caches, dHL)
    else:
        dh = dz0[:, None] * W2[0]
    dA = dh * gelu_d(A)
    return J, {"W1": dA.T @ X, "b1": dA.sum(axis=0),
               "W2": gW2, "c": gc, "blocks": gblocks}


def jlocal_and_grads(W1, b1, W2, c, X, masks, s_entry, gamma,
                     rho_floor, blocks=()):
    """J_local = J_inv + gamma * J_disc, combined gradients
    (incl. per-block when the unit carries composition blocks)."""
    Ji, gi = jinv_and_grads(W1, b1, W2, c, X, masks, blocks=blocks)
    Jd, gd = jdisc_and_grads(W1, b1, W2, c, X, s_entry, rho_floor,
                             blocks=blocks)
    grads = {k: gi[k] + gamma * gd[k] for k in gi
             if k != "blocks"}
    grads["blocks"] = [
        {key: gi["blocks"][k][key] + gamma * gd["blocks"][k][key]
         for key in gi["blocks"][k]}
        for k in range(len(blocks))]
    return Ji + gamma * Jd, Ji, Jd, grads


def jan_and_grads(W1, b1, W2, c, X, p_mask, K):
    """B1 (attention-build S7): the EXACT closed form of the mask
    objective — E[J_inv] under the truncated Bernoulli mask law
    (drop probability p_mask, conditioned on >=1 survivor; the law
    draw_masks samples). Certified against this module's own
    draw_masks + jinv_and_grads (4000 MC sessions, |z| = 0.73;
    docs/attention-build certifications S3). Leaf-only: the caller
    enforces the leaf gate exactly as the mask path does.

    Returns (J_tilde, grads) where J_tilde = (1 - 1/K) * J_an is
    session-comparable: every existing threshold (spu_tau_rel on
    Ji) carries over unchanged. grads is jinv-SHAPED (incl.
    "blocks": [], empty by the leaf definition) so jlocal-style
    combiners iterate it safely; grad c = 0 exactly (variance is
    shift-invariant, matching jinv_and_grads).
    """
    n = len(X)
    A, h, _ = forward_parts(W1, b1, W2, c, X)
    H = h.shape[1]
    C = h * W2[0]                              # loadings (n, H)
    S1 = C.sum(axis=1)
    S2 = (C ** 2).sum(axis=1)
    q = p_mask ** H
    a = 1.0 / (1.0 - q)
    kp = 1.0 - p_mask
    V = a * p_mask * kp * S2 + kp ** 2 * (a - a * a) * S1 ** 2
    scale = 1.0 - 1.0 / K
    Jt = float(scale * V.mean())
    dV_dC = (scale / n) * (2 * a * p_mask * kp * C
                           + 2 * kp ** 2 * (a - a * a) * S1[:, None])
    gW2 = (dV_dC * h).sum(axis=0)[None, :]
    dh = dV_dC * W2[0]
    dA = dh * gelu_d(A)
    return Jt, {"W1": dA.T @ X, "b1": dA.sum(axis=0),
                "W2": gW2, "c": np.zeros(1), "blocks": []}


def jlocal_an_and_grads(W1, b1, W2, c, X, p_mask, K, s_entry,
                        gamma, rho_floor):
    """B1 combiner: J_local = J_tilde + gamma * J_disc (the
    jlocal_and_grads pattern; J_disc reused unchanged as the
    mandatory anti-collapse member). Leaf-only (no blocks)."""
    Ja, ga = jan_and_grads(W1, b1, W2, c, X, p_mask, K)
    Jd, gd = jdisc_and_grads(W1, b1, W2, c, X, s_entry, rho_floor)
    grads = {k: ga[k] + gamma * gd[k] for k in ga if k != "blocks"}
    grads["blocks"] = []
    return Ja + gamma * Jd, Ja, Jd, grads
