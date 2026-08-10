"""L0-adjacent trainer math (doc 86 §5 normative forms;
referee boxes TB-P01..P03/P06). Pure numpy on plain arrays —
no object knowledge (layering law)."""
import numpy as np


def gae(rews, vals, last_v, dones, gamma, lam):
    """Generalized advantage estimation, normative recursion
    (doc 86 §5): acc_t = delta_t + gamma*lam*nonterm*acc_{t+1},
    delta_t = r_t + gamma*V(s_{t+1})*nonterm - V(s_t).
    Verified vs SB3 RolloutBuffer.compute_returns_and_advantage
    (blueprint A1, 3e-6 at float32) and vs the explicit
    (gamma*lam)^k truncated sums (TB-P01 dual form)."""
    rews = np.asarray(rews, dtype=float)
    vals = np.asarray(vals, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    T = len(rews)
    adv = np.empty(T)
    acc = 0.0
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else vals[t + 1]
        nonterm = 0.0 if dones[t] else 1.0
        delta = rews[t] + gamma * nv * nonterm - vals[t]
        acc = delta + gamma * lam * nonterm * acc
        adv[t] = acc
    return adv


def clipped_surrogate(logp_new, logp_old, adv, clip):
    """PPO clipped-surrogate loss (mean over samples):
    -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))."""
    ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    un = ratio * np.asarray(adv)
    cl = np.clip(ratio, 1.0 - clip, 1.0 + clip) * np.asarray(adv)
    return float(-np.mean(np.minimum(un, cl)))


def value_loss(vpred, vtarget):
    """Value MSE (mean squared error)."""
    d = np.asarray(vpred) - np.asarray(vtarget)
    return float(np.mean(d * d))


def entropy(probs):
    """Mean categorical entropy of prob rows: -sum p ln p.
    R-R1 guard (G-8 boundary class): the p=0 term's math
    limit is 0 and p*log(p+1e-300) delivers it exactly."""
    p = np.asarray(probs, dtype=float)
    return float(np.mean(-np.sum(p * np.log(p + 1e-300),
                                 axis=1)))


def group_advantages(returns, eps=1e-9):
    """GRPO group-relative standardization (doc 86 §5):
    (R - mean)/std over the COMPLETE-episode group.
    ZERO-VARIANCE EDGE (C-1 registered): sd <= eps =>
    advantages EXACTLY zero (no update signal, no div/0)."""
    r = np.asarray(returns, dtype=float)
    mu, sd = r.mean(), r.std()
    if sd <= eps:
        return np.zeros_like(r)
    return (r - mu) / sd


def entropy_grad_logits(p):
    """dH/dz for p = softmax(z) rows (plan 95 Formula A,
    verified 5 checks x 5 real-number runs vs sympy / torch
    autograd / mpmath-50 FD / SB3 source / Jacobian route):
      dH/dz_k = -p_k (ln p_k + H)."""
    p = np.asarray(p, dtype=float)
    # G-8 guard (in-file 1e-300 precedent): softmax can
    # underflow to exact 0; the math limit p->0 of
    # -p(ln p + H) is 0, and p*log(p+1e-300) delivers it
    # exactly since p==0 annihilates the factor.
    lp = np.log(p + 1e-300)
    H = -np.sum(p * lp, axis=1, keepdims=True)
    return -p * (lp + H)


def kl_grad_logits(p, q):
    """dKL/dz for p = softmax(z) rows against a FIXED
    reference distribution q (plan 95 Formula B, verified
    5 checks x 5 real-number runs):
      KL = sum p (ln p - ln q)
      dKL/dz_k = p_k [ (ln p_k - ln q_k) - KL ]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # G-8 guard: rows with underflowed-to-0 entries stay
    # finite; the p_k->0 limit of p_k[(ln p_k - ln q_k) - KL]
    # is 0 and the guarded form delivers it exactly (p_k==0
    # annihilates its factor).
    d = np.log(p + 1e-300) - np.log(q + 1e-300)
    kl = np.sum(p * d, axis=1, keepdims=True)
    return p * (d - kl)
