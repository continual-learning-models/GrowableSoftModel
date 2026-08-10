"""ENGINE math primitives — the pure functions every layer
shares (0_ARCHITECTURE section 5). Public: gelu, gelu_d, ln_fwd,
ln_bwd (the LayerNorm pair, public here because they are used
across packages; section 4 forbids cross-package underscore
imports)."""
import numpy as np


def gelu(a):
    return 0.5 * a * (1.0 + np.tanh(0.7978845608 * (a + 0.044715 * a ** 3)))



def gelu_d(a):
    t = np.tanh(0.7978845608 * (a + 0.044715 * a ** 3))
    return 0.5 * (1.0 + t) + 0.5 * a * (1.0 - t ** 2) * 0.7978845608 * (1.0 + 3 * 0.044715 * a ** 2)



def ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    sd = np.sqrt(var + 1e-5)
    xhat = (x - mu) / sd
    return xhat * g + b, (xhat, sd)



def ln_bwd(dy, cache, g):
    xhat, sd = cache
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / sd
    return dx, dg, db


