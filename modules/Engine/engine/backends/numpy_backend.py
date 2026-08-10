"""NumpyBackend — the judge (DESIGN_BACKEND v2.2).

Every kernel body here is the released implementation MOVED
VERBATIM (pure code motion; B1). No algebraic rewrites: the
default world must remain bitwise-identical, guarded by the
golden fixtures, the ARCH_equiv gate, and the full battery.
gelu/gelu_d live here now; net.py re-exports them so every
existing importer is untouched.
"""
import numpy as np

from ..primitives import (gelu, gelu_d,   # noqa: F401
    ln_fwd, ln_bwd)

_ln_fwd = ln_fwd
_ln_bwd = ln_bwd

class NumpyBackend:
    name = "numpy"
    device = "cpu"
    dtype = "float64"

    # ---------- K9: edges ----------
    @staticmethod
    def ingest(x):
        # identity for float arrays (no copy — bitwise judge);
        # normalizes lists/ints like the released asarray idiom
        return np.asarray(x, float)

    @staticmethod
    def to_numpy(x):
        return x

    @staticmethod
    def copy(x):
        return x.copy()

    @staticmethod
    def numel(x):
        return x.size

    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def stack(rows):
        return np.stack(rows)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def rowmax_keep(x):
        return x.max(-1, keepdims=True)

    @staticmethod
    def rowsum_keep(x):
        return x.sum(-1, keepdims=True)

    @staticmethod
    def einsum(spec, *ops):
        return np.einsum(spec, *ops)

    @staticmethod
    def perm4(M, p):
        return M.transpose(p)

    @staticmethod
    def repeat_tokens(dPool, Fn):
        return np.repeat(dPool[:, None, :], Fn, axis=1) / Fn

    @staticmethod
    def ln_fwd(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        sd = np.sqrt(var + 1e-5)
        xhat = (x - mu) / sd
        return xhat * g + b, (xhat, sd)

    @staticmethod
    def ln_bwd(dy, cache, g):
        xhat, sd = cache
        dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
        db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dxhat = dy * g
        dx = (dxhat - dxhat.mean(-1, keepdims=True)
              - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / sd
        return dx, dg, db

    @staticmethod
    def gelu(a):
        return gelu(a)

    @staticmethod
    def gelu_d(a):
        return gelu_d(a)

    # ---------- K1 ----------
    @staticmethod
    def dense_forward(W1, b1, X):
        A = X @ W1.T + b1
        return A, gelu(A)

    # ---------- K2 (verbatim from Network._apply_blocks) ----------
    @staticmethod
    def block_chain_forward(blocks, H0, cache=False):
        Hk = H0
        caches = []
        for blk in blocks:
            if blk["Bin"].shape[1] != Hk.shape[1]:
                raise ValueError(
                    "composition block width mismatch: scope was widened "
                    "after deepen; omega-on-deepened-scope integration is "
                    "a next-round item (DESIGN_DEEPEN, out of scope)")
            Z = Hk @ blk["Bin"].T + blk["bb"]
            G = gelu(Z)
            Hn = Hk + G @ blk["Bout"].T
            if cache:
                caches.append((Hk, Z, G))
            Hk = Hn
        if cache:
            return Hk, caches
        return Hk

    # ---------- K3 (verbatim tail of Network._grads) ----------
    def scope_backward(self, W1, b1, W2, c, blocks, Xs, ys, A, H0,
                       loop_block=None, loop_tol=1e-6,
                       loop_K_max=32):
        n = len(Xs)
        if blocks:
            HL, caches = self.block_chain_forward(blocks, H0,
                                                  cache=True)
        else:
            HL, caches = H0, []
        # lambda tail (K11): the head reads z* in place of HL;
        # loop_block=None keeps the historical path byte-identical
        if loop_block is not None:
            z_star, loop_k, zs = self.loop_forward(
                HL, loop_block["L_in"], loop_block["b_l"],
                loop_block["L_out"], loop_tol, loop_K_max)
        else:
            z_star, loop_k, zs = HL, None, None
        pred = z_star @ W2.T + c
        err = pred - ys
        mse = float(np.mean(err ** 2))
        dH = err @ W2
        if loop_block is not None:
            # K12: unrolled BPTT back to HL; per-sample dH stays
            # unscaled (the /n convention applies at param grads)
            dH, lgL_in, lgb_l, lgL_out = self.loop_backward(
                zs, loop_block["L_in"], loop_block["b_l"],
                loop_block["L_out"], dH)
        bgrads = [None] * len(blocks)
        for k in range(len(blocks) - 1, -1, -1):
            blk = blocks[k]
            Hprev, Z, G = caches[k]
            dBout = dH.T @ G / n
            dG = dH @ blk["Bout"]
            dZ = dG * gelu_d(Z)
            dBin = dZ.T @ Hprev / n
            dbb = dZ.mean(0)
            dH = dH + dZ @ blk["Bin"]
            bgrads[k] = (dBin, dbb, dBout)
        dA = dH * gelu_d(A)
        gW1 = dA.T @ Xs / n
        gb1 = dA.mean(0)
        gW2 = err.T @ z_star / n
        gc = err.mean(0)
        grads = [gW1, gb1, gW2, gc]
        for gBin, gbb, gBout in bgrads:
            grads += [gBin, gbb, gBout]
        if loop_block is not None:
            grads += [lgL_in / n, lgb_l / n, lgL_out / n]
        return grads, {"dH0": dH, "mse": mse, "loop_k": loop_k}

    # ---------- K7 (verbatim from _Adam.step) ----------
    @staticmethod
    def adam_step(m, v, t, lr, params, grads):
        t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9 ** t)
            vh = v[i] / (1 - 0.999 ** t)
            out.append(p - lr * mh / (np.sqrt(vh) + 1e-8))
        return out, t

    @staticmethod
    def sgd_step(params, grads, lr):
        return [pv - lr * gv for pv, gv in zip(params, grads)]

    # (verbatim from AttentionBody.train_step optimizer loop)
    @staticmethod
    def adam_step_dict(P, G, adam, t, lr):
        t += 1
        for k in P:
            m, v = adam[k]
            m[:] = 0.9 * m + 0.1 * G[k]
            v[:] = 0.999 * v + 0.001 * G[k] ** 2
            mh = m / (1 - 0.9 ** t)
            vh = v / (1 - 0.999 ** t)
            P[k] = P[k] - lr * mh / (np.sqrt(vh) + 1e-8)
        return t

    # (verbatim from the transformer host's _step — NOTE the
    # multiplication ASSOCIATION 0.001 * G * G differs from the
    # attention body's 0.001 * G**2 by one ULP; each host keeps
    # its own exact expression — bitwise doctrine)
    @staticmethod
    def adam_step_dict_mul(P, G, adam, t, lr):
        t += 1
        for k in P:
            m, v = adam[k]
            m[:] = 0.9 * m + 0.1 * G[k]
            v[:] = 0.999 * v + 0.001 * G[k] * G[k]
            mh = m / (1 - 0.9 ** t)
            vh = v / (1 - 0.999 ** t)
            P[k] = P[k] - lr * mh / (np.sqrt(vh) + 1e-8)
        return t

    @staticmethod
    def sgd_step_dict(P, G, lr):
        for k in P:
            P[k] = P[k] - lr * G[k]

    # ---------- K8 ----------
    @staticmethod
    def readout(HL, W2, c):
        return HL @ W2.T + c

    # ---------- K10 (verbatim from Network.train_step) ----------
    @staticmethod
    def fit_x_stats(X):
        return X.mean(0), X.std(0) + 1e-8

    @staticmethod
    def y_stats(y):
        return float(y.mean()), float(y.std()) + 1e-8

    @staticmethod
    def standardize(X, mu, sd):
        return (X - mu) / sd

    @staticmethod
    def refresh_readout(W2, c, y_mu, y_sd, mu_n, sd_n):
        return (W2 * (y_sd / sd_n),
                (c * y_sd + y_mu - mu_n) / sd_n)

    # ---------- K4 (installed in B1 batch 2: attention body) ----
    # att_forward / att_backward are appended by the batch-2
    # motion (kept in one class; see below in this file).

    # ---------- K4 (verbatim from AttentionBody, B1 batch 2) --
    def att_forward(self, P, Xs, d_model, L, n_heads, m_ffn,
                    cache=False, token_mask=None):
        n = len(Xs)
        d, hh = d_model, n_heads
        dh = d // hh
        T = Xs[:, :, None] * P["Wv"][None] + P["Bf"][None]
        if token_mask is not None:
            # SPU perturbation site (DESIGN_GROW_BODY_TYPE 7 stage
            # 2): drop whole TOKEN representations — the same
            # partial-internal-information semantics as masking
            # hidden nodes on the reference body.
            T = T * token_mask[None, :, None]
        caches = [Xs]
        for l in range(L):
            Tn, c1 = _ln_fwd(T, P[f"g1_{l}"], P[f"b1n_{l}"])
            Q = Tn @ P[f"Wq_{l}"]
            K = Tn @ P[f"Wk_{l}"]
            V = Tn @ P[f"Wk2_{l}"]

            def heads_(M):
                return M.reshape(n, -1, hh, dh).transpose(0, 2, 1, 3)
            Qh, Kh, Vh = heads_(Q), heads_(K), heads_(V)
            S = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(dh)
            A = np.exp(S - S.max(-1, keepdims=True))
            A = A / A.sum(-1, keepdims=True)
            O = (A @ Vh).transpose(0, 2, 1, 3).reshape(n, -1, d) \
                @ P[f"Wo_{l}"]
            T1 = T + O
            Tn2, c2 = _ln_fwd(T1, P[f"g2_{l}"],
                              P[f"b2n_{l}"])
            pre = Tn2 @ P[f"W1_{l}"] + P[f"b1_{l}"]
            Hf = gelu(pre)
            F_out = Hf @ P[f"W2_{l}"] + P[f"b2_{l}"]
            T2 = T1 + F_out
            if cache:
                caches.append((T, Tn, c1, Q, K, V, A, Vh, T1, Tn2,
                               c2, pre, Hf))
            T = T2
        Pool = T.mean(axis=1)
        raw = Pool @ P["Wout"] + P["bout"]
        if cache:
            return raw, Pool, caches
        return raw


    def att_backward(self, P, dlog, Pool, caches, d_model, L,
                     n_heads, m_ffn, token_mask=None):
        """Backward from a seeded d(objective)/d(raw) — shared by
        the training loss and the SPU local objective (which seeds
        dJ/dz analytically). Same operations, same order as the
        original inline backward (bitwise regression-guarded)."""
        n = len(dlog)
        d, hh = d_model, n_heads
        dh = d // hh
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["Wout"] = Pool.T @ dlog
        G["bout"] = dlog.sum(0)
        dPool = dlog @ P["Wout"].T
        Fn = caches[0].shape[1]
        dT = np.repeat(dPool[:, None, :], Fn, axis=1) / Fn
        for l in range(L - 1, -1, -1):
            (T, Tn, c1, Q, K, V, A, Vh, T1, Tn2, c2,
             pre, Hf) = caches[1 + l]
            dF = dT
            G[f"W2_{l}"] += Hf.reshape(-1, m_ffn).T \
                @ dF.reshape(-1, d)
            G[f"b2_{l}"] += dF.sum((0, 1))
            dH = dF @ P[f"W2_{l}"].T
            dpre = dH * gelu_d(pre)
            G[f"W1_{l}"] += Tn2.reshape(-1, d).T \
                @ dpre.reshape(-1, m_ffn)
            G[f"b1_{l}"] += dpre.sum((0, 1))
            dTn2 = dpre @ P[f"W1_{l}"].T
            dT1, dg2, db2 = _ln_bwd(dTn2, c2, P[f"g2_{l}"])
            G[f"g2_{l}"] += dg2
            G[f"b2n_{l}"] += db2
            dT1 = dT1 + dT                        # residual skip
            dO = dT1
            Ocat = (A @ Vh).transpose(0, 2, 1, 3).reshape(n, -1, d)
            G[f"Wo_{l}"] += Ocat.reshape(-1, d).T @ dO.reshape(-1, d)
            dOcat = dO @ P[f"Wo_{l}"].T
            dOh = dOcat.reshape(n, -1, hh, dh).transpose(0, 2, 1, 3)
            dA = dOh @ Vh.transpose(0, 1, 3, 2)
            dVh = A.transpose(0, 1, 3, 2) @ dOh
            dS = A * (dA - (dA * A).sum(-1, keepdims=True))
            dS = dS / np.sqrt(dh)
            Qh = Q.reshape(n, -1, hh, dh).transpose(0, 2, 1, 3)
            Kh = K.reshape(n, -1, hh, dh).transpose(0, 2, 1, 3)
            dQh = dS @ Kh
            dKh = dS.transpose(0, 1, 3, 2) @ Qh

            def unheads(M):
                return M.transpose(0, 2, 1, 3).reshape(n, -1, d)
            dQ, dK, dV = unheads(dQh), unheads(dKh), unheads(dVh)
            G[f"Wq_{l}"] += Tn.reshape(-1, d).T @ dQ.reshape(-1, d)
            G[f"Wk_{l}"] += Tn.reshape(-1, d).T @ dK.reshape(-1, d)
            G[f"Wk2_{l}"] += Tn.reshape(-1, d).T @ dV.reshape(-1, d)
            dTn = (dQ @ P[f"Wq_{l}"].T
                   + dK @ P[f"Wk_{l}"].T
                   + dV @ P[f"Wk2_{l}"].T)
            dT0, dg1, db1 = _ln_bwd(dTn, c1, P[f"g1_{l}"])
            G[f"g1_{l}"] += dg1
            G[f"b1n_{l}"] += db1
            dT = dT0 + dT1                        # residual skip
        Xs_ = caches[0]
        if token_mask is not None:
            dT = dT * token_mask[None, :, None]
        G["Wv"] += (Xs_[:, :, None] * dT).sum(0)
        G["Bf"] += dT.sum(0)
        return G


    # ---------- K14/K15: categorical-head kernels (GSM-I2;
    # numpy bodies are VERBATIM relocations of the MSOrgan
    # categorical math — the judge stays bitwise) ----------
    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def cat_forward(self, W2, c, Hact):
        return self.softmax(Hact @ W2.T + c)

    @staticmethod
    def cat_ce(probs, onehot):
        # ce = -mean(log p_true); expressed via onehot so the
        # same formula ports to torch without fancy indexing
        return float(-np.mean(
            np.log((probs * onehot).sum(axis=1) + 1e-12)))

    @staticmethod
    def cat_backward(W1, b1, W2, c, Xs, A, Hact, probs, onehot):
        n = len(Xs)
        err = probs - onehot
        dH = err @ W2
        dA = dH * gelu_d(A)
        gW1 = dA.T @ Xs / n
        gb1 = dA.mean(0)
        gW2 = err.T @ Hact / n
        gc = err.mean(0)
        return [gW1, gb1, gW2, gc], dH

    # ---------- K16: numeric-uncertainty head kernels (GSM-I3;
    # heteroscedastic Gaussian NLL, DEV_PLAN_GSM_I3 3.2 — NEW
    # code, the FD net is the truth standard; K14 convention:
    # per-sample err, /n at the leaves, dH undivided) ----------
    NLL_CLAMP = 10.0

    def nll_forward(self, W2, c, Hact, nll_clamp=None):
        # nll_clamp: per-call override (param-interface batch S6,
        # docs/system/22 item 28); None -> class default. The shared
        # backend singleton is never mutated.
        clamp = self.NLL_CLAMP if nll_clamp is None else nll_clamp
        z = Hact @ W2.T + c
        mu = z[:, 0]
        v = np.clip(z[:, 1], -clamp, clamp)
        return mu, v

    @staticmethod
    def nll_loss(t, mu, v):
        return float(np.mean(
            0.5 * (v + (t - mu) ** 2 * np.exp(-v))))

    def nll_backward(self, W1, b1, W2, c, Xs, A, Hact, mu, v, t,
                     nll_clamp=None):
        clamp = self.NLL_CLAMP if nll_clamp is None else nll_clamp
        n = len(Xs)
        r = t - mu
        e = np.exp(-v)
        err = np.stack([-r * e,
                        0.5 * (1.0 - r * r * e)], axis=1)
        # masked subgradient where the clamp is active
        err[:, 1] = np.where(np.abs(v) >= clamp,
                             0.0, err[:, 1])
        dH = err @ W2
        dA = dH * gelu_d(A)
        gW1 = dA.T @ Xs / n
        gb1 = dA.mean(0)
        gW2 = err.T @ Hact / n
        gc = err.mean(0)
        return [gW1, gb1, gW2, gc], dH

    # ---------- K11/K12: loop-operator kernels (delegation — the
    # judge reference bodies live in engine.loop_ops) ----------
    @staticmethod
    def loop_forward(H_L, L_in, b_l, L_out, tol, K_max):
        from ..loop_ops import loop_forward
        return loop_forward(H_L, L_in, b_l, L_out, tol, K_max)

    @staticmethod
    def loop_backward(zs, L_in, b_l, L_out, dz_star):
        from ..loop_ops import loop_backward
        return loop_backward(zs, L_in, b_l, L_out, dz_star)

    # ---------- K5/K6: SPU objective kernels (delegation — the
    # released functions ARE already pure kernels; lazy imports
    # keep the module graph acyclic) ----------
    @staticmethod
    def forward_parts(*a, **k):
        from ..spu.spu_objective import forward_parts as f
        return f(*a, **k)

    @staticmethod
    def forward_chain(*a, **k):
        from ..spu.spu_objective import forward_chain as f
        return f(*a, **k)

    @staticmethod
    def batch_std(*a, **k):
        from ..spu.spu_objective import batch_std as f
        return f(*a, **k)

    @staticmethod
    def jinv_and_grads(*a, **k):
        from ..spu.spu_objective import jinv_and_grads as f
        return f(*a, **k)

    @staticmethod
    def jdisc_and_grads(*a, **k):
        from ..spu.spu_objective import jdisc_and_grads as f
        return f(*a, **k)

    @staticmethod
    def jlocal_and_grads(*a, **k):
        from ..spu.spu_objective import jlocal_and_grads as f
        return f(*a, **k)

    @staticmethod
    def jlocal_an_and_grads(*a, **k):
        from ..spu.spu_objective import jlocal_an_and_grads as f
        return f(*a, **k)
