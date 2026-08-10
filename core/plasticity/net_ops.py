"""omega — the widening operator, Network level (Stage 3a).

Complement to the reference-net refinement operator rho (node -> inner
network): rho refines where difficulty is LOCAL; omega widens where
saturation is GLOBAL. Because transformer inner bodies ARE reference-net
Networks, this one function gives outward growth at EVERY scale of
both hosts.

Contract (the axiom, mechanized):
- function preservation is EXACT: new units enter with ZERO output
  weights (an initial value, never a mask);
- every new parameter is trainable from step one (optimizer slots
  extended; nothing excluded);
- EMA instability stats are warm-started at the network's current
  row-mean so new units compete fairly in growth rankings (kills the
  cold-start bias W2b);
- new units are ordinary growth sites afterwards (rho can refine them
  — the operators compose).
"""
from __future__ import annotations

import numpy as np

from core._modules import reference_net  # noqa: F401
from reference_net.net import Network


def widen_net(net: Network, k: int = 1) -> dict:
    """Append k hidden units to this network (any scale). Exact
    function preservation; returns a small report."""
    if k < 1:
        raise ValueError("k must be >= 1")
    d_in, H0 = net.d_in, net.H
    net._seed_counter += 1
    rng = np.random.default_rng(net._seed_counter)

    new_W1 = rng.normal(0, np.sqrt(2.0 / d_in), (k, d_in))
    net.W1 = np.vstack([net.W1, new_W1])
    net.b1 = np.concatenate([net.b1, np.zeros(k)])
    n_out = net.W2.shape[0]         # 1 numeric, n_classes, 2 dist
    net.W2 = np.hstack([net.W2, np.zeros((n_out, k))])  # ZERO -> preserved
    net.H = H0 + k

    # optimizer slots: extend with zeros (params stay fully trainable;
    # zero moments simply mean "no history yet")
    opt = net.opt
    opt.m[0] = np.vstack([opt.m[0], np.zeros((k, d_in))])
    opt.v[0] = np.vstack([opt.v[0], np.zeros((k, d_in))])
    opt.m[1] = np.concatenate([opt.m[1], np.zeros(k)])
    opt.v[1] = np.concatenate([opt.v[1], np.zeros(k)])
    opt.m[2] = np.hstack([opt.m[2], np.zeros((n_out, k))])
    opt.v[2] = np.hstack([opt.v[2], np.zeros((n_out, k))])

    # EMA warm-start at row-mean: new units enter rankings mid-pack
    mean_dw = net._ema_dw.mean(axis=0, keepdims=True)
    mean_adw = net._ema_adw.mean(axis=0, keepdims=True)
    net._ema_dw = np.vstack([net._ema_dw, np.repeat(mean_dw, k, 0)])
    net._ema_adw = np.vstack([net._ema_adw, np.repeat(mean_adw, k, 0)])

    # Growth Interface Reform: the attach site is k channels wider
    # -> every assembly A_g gains ZERO columns (exact preservation;
    # optimizer/instability slots extend with zeros likewise)
    port = getattr(net, "_port_site", None)
    if port is not None:
        port.C = net.H
        for slot in port.bodies:
            kw = np.asarray(slot["A"]).shape[0]
            for key in ("A", "mA", "vA", "edw", "eadw"):
                pad = np.zeros((kw, k)) if key != "eadw" \
                    else np.zeros((kw, k)) + 1e-12
                slot[key] = np.hstack(
                    [np.asarray(slot[key]), pad])

    # omega-on-deepened/looped-scope integration (60D D-8):
    # every composition block is ZERO-EXTENDED to the new width
    # — Bin (m x H) gains k ZERO columns, Bout (H x m) gains k
    # ZERO rows, bb unchanged; the loop block identically
    # (L_in/L_out). The new trunk units feed ZERO into and
    # receive ZERO from every block, so the served function is
    # BITWISE unchanged at the extension; the new stripes are
    # trainable from step one (optimizer slots extend zeroed —
    # the widen precedent for new tails). Param/opt layout:
    # [W1, b1, W2, c] + 3 per block + 3 loop (train_step).
    def _ext(idx_in, idx_out, m_rows):
        for slots in (opt.m, opt.v):
            if idx_out < len(slots):
                slots[idx_in] = np.hstack(
                    [np.asarray(slots[idx_in]),
                     np.zeros((m_rows, k))])
                slots[idx_out] = np.vstack(
                    [np.asarray(slots[idx_out]),
                     np.zeros((k, m_rows))])
    for j, blk in enumerate(net.blocks):
        m_b = int(np.asarray(blk["bb"]).size)
        blk["Bin"] = np.hstack([np.asarray(blk["Bin"]),
                                np.zeros((m_b, k))])
        blk["Bout"] = np.vstack([np.asarray(blk["Bout"]),
                                 np.zeros((k, m_b))])
        _ext(4 + 3 * j, 4 + 3 * j + 2, m_b)
    lb = getattr(net, "loop_block", None)
    if lb is not None:
        m_l = int(np.asarray(lb["b_l"]).size)
        lb["L_in"] = np.hstack([np.asarray(lb["L_in"]),
                                np.zeros((m_l, k))])
        lb["L_out"] = np.vstack([np.asarray(lb["L_out"]),
                                 np.zeros((k, m_l))])
        base = 4 + 3 * len(net.blocks)
        _ext(base, base + 2, m_l)

    return {"widened": k, "H": net.H, "new_units": list(range(H0, net.H)),
            "params": net.n_params()}


def widen_at(net: Network, path: str, k: int = 1) -> dict:
    """Widen the network at `path` ('root' or 'j/k/...' inner hops) —
    outward growth exists at every scale, as the axiom requires."""
    owner = net
    if path not in ("", "root"):
        for hop in path.split("/"):
            owner = owner.grown_body(int(hop))
            if owner is None:
                raise KeyError(int(hop))
    out = widen_net(owner, k)
    out["path"] = path or "root"
    return out


def add_feature_net(net: Network, default: float = 0.0) -> dict:
    """sigma — input-schema growth (Stage 4): a new input feature
    enters with ZERO weights (initial value, never a mask), so the
    function is preserved EXACTLY for ANY values in the new column;
    participation is earned by training. Applied recursively — every
    inner network shares d_in. Standardization vectors extend with
    (mu=default, sd=1): the constant-backfill divide-by-zero guard."""
    def _grow(n: Network):
        n.W1 = np.hstack([n.W1, np.zeros((n.H, 1))])
        n._ema_dw = np.hstack([n._ema_dw, np.zeros((n.H, 1))])
        n._ema_adw = np.hstack([n._ema_adw,
                                np.zeros((n.H, 1)) + 1e-12])
        n.opt.m[0] = np.hstack([n.opt.m[0], np.zeros((n.H, 1))])
        n.opt.v[0] = np.hstack([n.opt.v[0], np.zeros((n.H, 1))])
        if n._x_mu is not None:
            n._x_mu = np.concatenate([n._x_mu, [float(default)]])
            n._x_sd = np.concatenate([n._x_sd, [1.0]])
        n.d_in += 1
        for inner in n.inner.values():
            _grow(inner)
        # Growth Interface Reform: port bodies share this scope's
        # input schema (X_b = the scope's standardized input) —
        # same zero-weight column + identity-scaler extension
        port = getattr(n, "_port_site", None)
        if port is not None:
            for slot in port.bodies:
                _grow(slot["body"])
    _grow(net)
    return {"d_in": net.d_in, "default": float(default),
            "params": net.n_params()}
