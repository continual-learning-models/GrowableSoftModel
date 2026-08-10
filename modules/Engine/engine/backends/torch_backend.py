"""TorchBackend — GPU/CPU tensor backend (DESIGN_BACKEND v2.2).

Implements the K1-K10 kernel contract with torch operations
mirroring the judge's formulas one-to-one (torch semantics
differences handled explicitly: population std via
unbiased=False, permute for 4-d transposes). Qualified by the
BACKEND CONFORMANCE KIT, not by bitwise equality: tolerances per
tests/backend_kit/spec.md; INTEGER run structure must match the
judge exactly because all randomness stays numpy-sourced.
Devices: cpu / cuda / mps (float32 on mps — no float64 there);
the device string is passed through openly (the xla door).
"""
import numpy as np

try:
    import torch
except ImportError:                                   # pragma: no cover
    torch = None

_DTYPES = {"float32": "float32", "float64": "float64"}


def _t_gelu(a):
    return 0.5 * a * (1.0 + torch.tanh(
        0.7978845608 * (a + 0.044715 * a ** 3)))


def _t_gelu_d(a):
    t = torch.tanh(0.7978845608 * (a + 0.044715 * a ** 3))
    return (0.5 * (1.0 + t) + 0.5 * a * (1.0 - t ** 2)
            * 0.7978845608 * (1.0 + 3 * 0.044715 * a ** 2))


def _t_ln_fwd(x, g, b):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    sd = torch.sqrt(var + 1e-5)
    xhat = (x - mu) / sd
    return xhat * g + b, (xhat, sd)


def _t_ln_bwd(dy, cache, g):
    xhat, sd = cache
    dims = tuple(range(dy.dim() - 1))
    dg = (dy * xhat).sum(dim=dims)
    db = dy.sum(dim=dims)
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(-1, keepdim=True)
          - xhat * (dxhat * xhat).mean(-1, keepdim=True)) / sd
    return dx, dg, db


class TorchBackend:
    name = "torch"

    def __init__(self, device=None, dtype=None):
        if torch is None:
            raise ImportError("compute_backend='torch' requires "
                              "PyTorch (pip install torch)")
        self.device = device or "cpu"
        # 61C: device NAME validated at construction (torch's
        # own validator — the authoritative legal list; loud,
        # named, typed per X2; never accept-then-crash)
        try:
            torch.device(self.device)
        except RuntimeError as e:
            raise ValueError(
                f"unknown compute_device {self.device!r}: {e}")
        dt = dtype or "float32"
        if dt not in _DTYPES:
            raise ValueError(f"unknown compute_dtype {dt!r}; "
                             f"valid: {sorted(_DTYPES)}")
        if self.device == "mps" and dt == "float64":
            raise ValueError("mps has no float64; use "
                             "compute_dtype='float32'")
        self.dtype = dt
        self._dt = getattr(torch, dt)

    # ---------- K9: edges ----------
    def ingest(self, x):
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=self._dt)
        return torch.as_tensor(np.asarray(x, dtype=np.float64),
                               dtype=self._dt, device=self.device)

    @staticmethod
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    @staticmethod
    def copy(x):
        return x.clone()

    @staticmethod
    def numel(x):
        return int(x.numel()) if torch.is_tensor(x) else x.size

    def zeros(self, shape):
        return torch.zeros(shape, dtype=self._dt,
                           device=self.device)

    @staticmethod
    def zeros_like(x):
        return torch.zeros_like(x)

    @staticmethod
    def abs(x):
        return torch.abs(x)

    @staticmethod
    def stack(rows):
        return torch.stack(rows)

    @staticmethod
    def exp(x):
        return torch.exp(x)

    @staticmethod
    def rowmax_keep(x):
        return x.max(-1, keepdim=True).values

    @staticmethod
    def rowsum_keep(x):
        return x.sum(-1, keepdim=True)

    @staticmethod
    def einsum(spec, *ops):
        return torch.einsum(spec, *ops)

    @staticmethod
    def perm4(M, p):
        return M.permute(p)

    @staticmethod
    def repeat_tokens(dPool, Fn):
        return dPool[:, None, :].repeat(1, Fn, 1) / Fn

    @staticmethod
    def ln_fwd(x, g, b):
        return _t_ln_fwd(x, g, b)

    @staticmethod
    def ln_bwd(dy, cache, g):
        return _t_ln_bwd(dy, cache, g)

    @staticmethod
    def gelu(a):
        return _t_gelu(a)

    @staticmethod
    def gelu_d(a):
        return _t_gelu_d(a)

    # ---------- K1 ----------
    @staticmethod
    def dense_forward(W1, b1, X):
        A = X @ W1.T + b1
        return A, _t_gelu(A)

    # ---------- K2 ----------
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
            G = _t_gelu(Z)
            Hn = Hk + G @ blk["Bout"].T
            if cache:
                caches.append((Hk, Z, G))
            Hk = Hn
        if cache:
            return Hk, caches
        return Hk

    # ---------- K3 ----------
    def scope_backward(self, W1, b1, W2, c, blocks, Xs, ys, A, H0,
                       loop_block=None, loop_tol=1e-6,
                       loop_K_max=32):
        n = len(Xs)
        if blocks:
            HL, caches = self.block_chain_forward(blocks, H0,
                                                  cache=True)
        else:
            HL, caches = H0, []
        if loop_block is not None:
            z_star, loop_k, zs = self.loop_forward(
                HL, loop_block["L_in"], loop_block["b_l"],
                loop_block["L_out"], loop_tol, loop_K_max)
        else:
            z_star, loop_k, zs = HL, None, None
        pred = z_star @ W2.T + c
        err = pred - ys
        mse = float(torch.mean(err ** 2))
        dH = err @ W2
        if loop_block is not None:
            dH, lgL_in, lgb_l, lgL_out = self.loop_backward(
                zs, loop_block["L_in"], loop_block["b_l"],
                loop_block["L_out"], dH)
        bgrads = [None] * len(blocks)
        for k in range(len(blocks) - 1, -1, -1):
            blk = blocks[k]
            Hprev, Z, G = caches[k]
            dBout = dH.T @ G / n
            dG = dH @ blk["Bout"]
            dZ = dG * _t_gelu_d(Z)
            dBin = dZ.T @ Hprev / n
            dbb = dZ.mean(0)
            dH = dH + dZ @ blk["Bin"]
            bgrads[k] = (dBin, dbb, dBout)
        dA = dH * _t_gelu_d(A)
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

    # ---------- K14/K15: categorical-head kernels (torch
    # ports of the judge bodies; same formulas) ----------
    @staticmethod
    def softmax(z):
        z = z - z.max(dim=1, keepdim=True).values
        e = torch.exp(z)
        return e / e.sum(dim=1, keepdim=True)

    def cat_forward(self, W2, c, Hact):
        return self.softmax(Hact @ W2.T + c)

    @staticmethod
    def cat_ce(probs, onehot):
        return float(-torch.mean(
            torch.log((probs * onehot).sum(dim=1) + 1e-12)))

    @staticmethod
    def cat_backward(W1, b1, W2, c, Xs, A, Hact, probs, onehot):
        n = len(Xs)
        err = probs - onehot
        dH = err @ W2
        dA = dH * _t_gelu_d(A)
        gW1 = dA.T @ Xs / n
        gb1 = dA.mean(0)
        gW2 = err.T @ Hact / n
        gc = err.mean(0)
        return [gW1, gb1, gW2, gc], dH

    # ---------- K16: numeric-uncertainty head kernels (torch
    # ports of the judge bodies; same formulas) ----------
    NLL_CLAMP = 10.0

    def nll_forward(self, W2, c, Hact, nll_clamp=None):
        clamp = self.NLL_CLAMP if nll_clamp is None else nll_clamp
        z = Hact @ W2.T + c
        mu = z[:, 0]
        v = torch.clamp(z[:, 1], -clamp, clamp)
        return mu, v

    @staticmethod
    def nll_loss(t, mu, v):
        return float(torch.mean(
            0.5 * (v + (t - mu) ** 2 * torch.exp(-v))))

    def nll_backward(self, W1, b1, W2, c, Xs, A, Hact, mu, v, t,
                     nll_clamp=None):
        clamp = self.NLL_CLAMP if nll_clamp is None else nll_clamp
        n = len(Xs)
        r = t - mu
        e = torch.exp(-v)
        err = torch.stack([-r * e,
                           0.5 * (1.0 - r * r * e)], dim=1)
        # masked subgradient where the clamp is active
        err[:, 1] = torch.where(torch.abs(v) >= clamp,
                                torch.zeros_like(v), err[:, 1])
        dH = err @ W2
        dA = dH * _t_gelu_d(A)
        gW1 = dA.T @ Xs / n
        gb1 = dA.mean(0)
        gW2 = err.T @ Hact / n
        gc = err.mean(0)
        return [gW1, gb1, gW2, gc], dH

    # ---------- K11/K12: loop-operator kernels (torch ports of
    # the loop_ops judge bodies; same formulas) ----------
    @staticmethod
    def loop_forward(H_L, L_in, b_l, L_out, tol, K_max):
        z = H_L
        zs = [z]
        k_used = 0
        for _ in range(int(K_max)):
            z_next = H_L + _t_gelu(z @ L_in.T + b_l) @ L_out.T
            k_used += 1
            # one host sync per iteration (disclosed: the stop
            # test reads a scalar off the device)
            delta = float((z_next - z).abs().max().item())
            z = z_next
            zs.append(z)
            if delta < tol:
                break
        return z, k_used, zs

    @staticmethod
    def loop_backward(zs, L_in, b_l, L_out, dz_star):
        k_hat = len(zs) - 1
        gL_in = torch.zeros_like(L_in)
        gb_l = torch.zeros_like(b_l)
        gL_out = torch.zeros_like(L_out)
        dH_L = torch.zeros_like(zs[0])
        r = dz_star
        for k in range(k_hat - 1, -1, -1):
            u = zs[k] @ L_in.T + b_l          # recomputed
            g = _t_gelu(u)
            gL_out = gL_out + r.T @ g
            s = (r @ L_out) * _t_gelu_d(u)
            gL_in = gL_in + s.T @ zs[k]
            gb_l = gb_l + s.sum(0)
            dH_L = dH_L + r
            r = s @ L_in
        dH_L = dH_L + r
        return dH_L, gL_in, gb_l, gL_out

    # ---------- K7 ----------
    @staticmethod
    def adam_step(m, v, t, lr, params, grads):
        t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9 ** t)
            vh = v[i] / (1 - 0.999 ** t)
            out.append(p - lr * mh / (torch.sqrt(vh) + 1e-8))
        return out, t

    @staticmethod
    def sgd_step(params, grads, lr):
        return [pv - lr * gv for pv, gv in zip(params, grads)]

    @staticmethod
    def adam_step_dict(P, G, adam, t, lr):
        t += 1
        for k in P:
            m, v = adam[k]
            m[:] = 0.9 * m + 0.1 * G[k]
            v[:] = 0.999 * v + 0.001 * G[k] ** 2
            mh = m / (1 - 0.9 ** t)
            vh = v / (1 - 0.999 ** t)
            P[k] = P[k] - lr * mh / (torch.sqrt(vh) + 1e-8)
        return t

    @staticmethod
    def adam_step_dict_mul(P, G, adam, t, lr):
        t += 1
        for k in P:
            m, v = adam[k]
            m[:] = 0.9 * m + 0.1 * G[k]
            v[:] = 0.999 * v + 0.001 * G[k] * G[k]
            mh = m / (1 - 0.9 ** t)
            vh = v / (1 - 0.999 ** t)
            P[k] = P[k] - lr * mh / (torch.sqrt(vh) + 1e-8)
        return t

    @staticmethod
    def sgd_step_dict(P, G, lr):
        for k in P:
            P[k] = P[k] - lr * G[k]

    # ---------- K8 ----------
    @staticmethod
    def readout(HL, W2, c):
        return HL @ W2.T + c

    # ---------- K10 ----------
    @staticmethod
    def fit_x_stats(X):
        return X.mean(0), X.std(0, unbiased=False) + 1e-8

    @staticmethod
    def y_stats(y):
        return (float(y.mean()),
                float(torch.std(y, unbiased=False)) + 1e-8)

    @staticmethod
    def standardize(X, mu, sd):
        return (X - mu) / sd

    @staticmethod
    def refresh_readout(W2, c, y_mu, y_sd, mu_n, sd_n):
        return (W2 * (y_sd / sd_n),
                (c * y_sd + y_mu - mu_n) / sd_n)

    # ---------- K4 ----------
    def att_forward(self, P, Xs, d_model, L, n_heads, m_ffn,
                    cache=False, token_mask=None):
        n = len(Xs)
        d, hh = d_model, n_heads
        dh = d // hh
        T = Xs[:, :, None] * P["Wv"][None] + P["Bf"][None]
        if token_mask is not None:
            tm = self.ingest(token_mask)
            T = T * tm[None, :, None]
        caches = [Xs]
        for l in range(L):
            Tn, c1 = _t_ln_fwd(T, P[f"g1_{l}"], P[f"b1n_{l}"])
            Q = Tn @ P[f"Wq_{l}"]
            K = Tn @ P[f"Wk_{l}"]
            V = Tn @ P[f"Wk2_{l}"]

            def heads_(M):
                return M.reshape(n, -1, hh, dh).permute(0, 2, 1, 3)
            Qh, Kh, Vh = heads_(Q), heads_(K), heads_(V)
            S = Qh @ Kh.permute(0, 1, 3, 2) / (dh ** 0.5)
            A = torch.exp(S - S.max(-1, keepdim=True).values)
            A = A / A.sum(-1, keepdim=True)
            O = (A @ Vh).permute(0, 2, 1, 3).reshape(n, -1, d) \
                @ P[f"Wo_{l}"]
            T1 = T + O
            Tn2, c2 = _t_ln_fwd(T1, P[f"g2_{l}"], P[f"b2n_{l}"])
            pre = Tn2 @ P[f"W1_{l}"] + P[f"b1_{l}"]
            Hf = _t_gelu(pre)
            F_out = Hf @ P[f"W2_{l}"] + P[f"b2_{l}"]
            T2 = T1 + F_out
            if cache:
                caches.append((T, Tn, c1, Q, K, V, A, Vh, T1, Tn2,
                               c2, pre, Hf))
            T = T2
        Pool = T.mean(dim=1)
        raw = Pool @ P["Wout"] + P["bout"]
        if cache:
            return raw, Pool, caches
        return raw

    def att_backward(self, P, dlog, Pool, caches, d_model, L,
                     n_heads, m_ffn, token_mask=None):
        n = len(dlog)
        d, hh = d_model, n_heads
        dh = d // hh
        G = {k: torch.zeros_like(v) for k, v in P.items()}
        G["Wout"] = Pool.T @ dlog
        G["bout"] = dlog.sum(0)
        dPool = dlog @ P["Wout"].T
        Fn = caches[0].shape[1]
        dT = dPool[:, None, :].repeat(1, Fn, 1) / Fn
        for l in range(L - 1, -1, -1):
            (T, Tn, c1, Q, K, V, A, Vh, T1, Tn2, c2,
             pre, Hf) = caches[1 + l]
            dF = dT
            G[f"W2_{l}"] += Hf.reshape(-1, m_ffn).T \
                @ dF.reshape(-1, d)
            G[f"b2_{l}"] += dF.sum((0, 1))
            dH = dF @ P[f"W2_{l}"].T
            dpre = dH * _t_gelu_d(pre)
            G[f"W1_{l}"] += Tn2.reshape(-1, d).T \
                @ dpre.reshape(-1, m_ffn)
            G[f"b1_{l}"] += dpre.sum((0, 1))
            dTn2 = dpre @ P[f"W1_{l}"].T
            dT1, dg2, db2 = _t_ln_bwd(dTn2, c2, P[f"g2_{l}"])
            G[f"g2_{l}"] += dg2
            G[f"b2n_{l}"] += db2
            dT1 = dT1 + dT
            dO = dT1
            Ocat = (A @ Vh).permute(0, 2, 1, 3).reshape(n, -1, d)
            G[f"Wo_{l}"] += Ocat.reshape(-1, d).T \
                @ dO.reshape(-1, d)
            dOcat = dO @ P[f"Wo_{l}"].T
            dOh = dOcat.reshape(n, -1, hh, dh).permute(0, 2, 1, 3)
            dA = dOh @ Vh.permute(0, 1, 3, 2)
            dVh = A.permute(0, 1, 3, 2) @ dOh
            dS = A * (dA - (dA * A).sum(-1, keepdim=True))
            dS = dS / (dh ** 0.5)
            Qh = Q.reshape(n, -1, hh, dh).permute(0, 2, 1, 3)
            Kh = K.reshape(n, -1, hh, dh).permute(0, 2, 1, 3)
            dQh = dS @ Kh
            dKh = dS.permute(0, 1, 3, 2) @ Qh

            def unheads(M):
                return M.permute(0, 2, 1, 3).reshape(n, -1, d)
            dQ, dK, dV = unheads(dQh), unheads(dKh), unheads(dVh)
            G[f"Wq_{l}"] += Tn.reshape(-1, d).T @ dQ.reshape(-1, d)
            G[f"Wk_{l}"] += Tn.reshape(-1, d).T @ dK.reshape(-1, d)
            G[f"Wk2_{l}"] += Tn.reshape(-1, d).T \
                @ dV.reshape(-1, d)
            dTn = (dQ @ P[f"Wq_{l}"].T + dK @ P[f"Wk_{l}"].T
                   + dV @ P[f"Wk2_{l}"].T)
            dT0, dg1, db1 = _t_ln_bwd(dTn, c1, P[f"g1_{l}"])
            G[f"g1_{l}"] += dg1
            G[f"b1n_{l}"] += db1
            dT = dT0 + dT1
        Xs_ = caches[0]
        if token_mask is not None:
            tm = self.ingest(token_mask)
            dT = dT * tm[None, :, None]
        G["Wv"] += (Xs_[:, :, None] * dT).sum(0)
        G["Bf"] += dT.sum(0)
        return G

    # ---------- K5/K6: SPU objective kernels (torch ports of the
    # released pure functions; masks arrive numpy — ingested) ----
    @staticmethod
    def batch_std(z0):
        return float(torch.sqrt(
            torch.mean((z0 - z0.mean()) ** 2) + 1e-12))

    @staticmethod
    def forward_parts(W1, b1, W2, c, X):
        A = X @ W1.T + b1
        h = _t_gelu(A)
        z0 = h @ W2[0] + c[0]
        return A, h, z0

    def forward_chain(self, W1, b1, W2, c, X, blocks=()):
        A = X @ W1.T + b1
        h = _t_gelu(A)
        HL, caches = self._chain_apply(blocks, h)
        z0 = HL @ W2[0] + c[0]
        return A, h, HL, caches, z0

    @staticmethod
    def _chain_apply(blocks, H0):
        H, caches = H0, []
        for blk in blocks:
            Pz = H @ blk["Bin"].T + blk["bb"]
            Gz = _t_gelu(Pz)
            caches.append((H, Pz, Gz))
            H = H + Gz @ blk["Bout"].T
        return H, caches

    @staticmethod
    def _chain_backward(blocks, caches, dH_out):
        gblocks = [None] * len(blocks)
        dH = dH_out
        for k in range(len(blocks) - 1, -1, -1):
            Hk, Pz, Gz = caches[k]
            blk = blocks[k]
            dBout = dH.T @ Gz
            dP = (dH @ blk["Bout"]) * _t_gelu_d(Pz)
            gblocks[k] = {"Bin": dP.T @ Hk, "bb": dP.sum(dim=0),
                          "Bout": dBout}
            dH = dH + dP @ blk["Bin"]
        return dH, gblocks

    def jinv_and_grads(self, W1, b1, W2, c, X, masks, blocks=()):
        masks = self.ingest(masks)
        n = len(X)
        K = len(masks)
        if blocks:
            return self._jinv_chain(W1, b1, W2, c, X, masks,
                                    blocks)
        A, h, _ = self.forward_parts(W1, b1, W2, c, X)
        w_eff = masks * W2[0]
        z = (h @ w_eff.T + c[0]).T
        zbar = z.mean(dim=0)
        diff = z - zbar
        J = float((diff ** 2).sum() / (K * n))
        Gd = (2.0 / (K * n)) * diff
        gW2 = ((Gd @ h) * masks).sum(dim=0)[None, :]
        gc = torch.zeros_like(c)
        dh = Gd.T @ w_eff
        dA = dh * _t_gelu_d(A)
        return J, {"W1": dA.T @ X, "b1": dA.sum(dim=0),
                   "W2": gW2, "c": gc, "blocks": []}

    def _jinv_chain(self, W1, b1, W2, c, X, masks, blocks):
        # BATCHED over the K masked copies (B3): one (K, n, H)
        # tensor pass through the chain instead of a Python loop —
        # a torch-internal reorder, inside the kit's tolerance
        # boundary (DESIGN 6b); integer structure unaffected.
        n = len(X)
        K = len(masks)
        A = X @ W1.T + b1
        h = _t_gelu(A)
        Hm = masks[:, None, :] * h[None]           # (K, n, H)
        caches = []
        for blk in blocks:
            Z = Hm @ blk["Bin"].T + blk["bb"]
            Gz = _t_gelu(Z)
            caches.append((Hm, Z, Gz))
            Hm = Hm + Gz @ blk["Bout"].T
        z = Hm @ W2[0] + c[0]                      # (K, n)
        zbar = z.mean(dim=0)
        diff = z - zbar
        J = float((diff ** 2).sum() / (K * n))
        Gd = (2.0 / (K * n)) * diff
        gW2 = torch.einsum("kn,knh->h", Gd, Hm)[None, :]
        dH = Gd[:, :, None] * W2[0]                # (K, n, H)
        gblocks = [None] * len(blocks)
        for k in range(len(blocks) - 1, -1, -1):
            Hk, Z, Gz = caches[k]
            blk = blocks[k]
            dBout = torch.einsum("knh,knm->hm", dH, Gz)
            dP = (dH @ blk["Bout"]) * _t_gelu_d(Z)
            gblocks[k] = {"Bin": torch.einsum("knm,knh->mh", dP,
                                              Hk),
                          "bb": dP.sum(dim=(0, 1)),
                          "Bout": dBout}
            dH = dH + dP @ blk["Bin"]
        dh = torch.einsum("knh,kh->nh", dH, masks)
        dA = dh * _t_gelu_d(A)
        return J, {"W1": dA.T @ X, "b1": dA.sum(dim=0), "W2": gW2,
                   "c": torch.zeros_like(c), "blocks": gblocks}

    def jdisc_and_grads(self, W1, b1, W2, c, X, s_entry, rho_floor,
                        blocks=()):
        zeros = {"W1": torch.zeros_like(W1),
                 "b1": torch.zeros_like(b1),
                 "W2": torch.zeros_like(W2),
                 "c": torch.zeros_like(c),
                 "blocks": [{k2: torch.zeros_like(v2)
                             for k2, v2 in b2.items()}
                            for b2 in blocks]}
        n = len(X)
        if blocks:
            A, h, HL, caches, z0 = self.forward_chain(
                W1, b1, W2, c, X, blocks)
        else:
            A, h, z0 = self.forward_parts(W1, b1, W2, c, X)
            HL = h
        s = self.batch_std(z0)
        gap = rho_floor * s_entry - s
        if gap <= 0.0:
            return 0.0, zeros
        J = float(gap ** 2)
        dz0 = (-2.0 * gap) * (z0 - z0.mean()) / (n * s)
        gW2 = (dz0 @ HL)[None, :]
        gblocks = zeros["blocks"]
        if blocks:
            dHL = torch.outer(dz0, W2[0])
            dh, gblocks = self._chain_backward(blocks, caches, dHL)
        else:
            dh = dz0[:, None] * W2[0]
        dA = dh * _t_gelu_d(A)
        return J, {"W1": dA.T @ X, "b1": dA.sum(dim=0),
                   "W2": gW2, "c": torch.zeros_like(c),
                   "blocks": gblocks}

    def jlocal_an_and_grads(self, W1, b1, W2, c, X, p_mask, K,
                            s_entry, gamma, rho_floor):
        """B1 delegator: the analytic objective runs in judge
        (numpy) form — leaf-only, no torch port needed in v1 (the
        judge is the bitwise reference; the SPU loop already
        round-trips through numpy for masks/std)."""
        import numpy as _np
        from ..spu.spu_objective import jlocal_an_and_grads as f
        to_np = lambda t: (t.detach().cpu().numpy()
                           if hasattr(t, "detach")
                           else _np.asarray(t))
        J, Ja, Jd, g = f(to_np(W1), to_np(b1), to_np(W2),
                         to_np(c), to_np(X), p_mask, K,
                         float(s_entry), gamma, rho_floor)
        g = {k: (self.ingest(v) if k != "blocks" else v)
             for k, v in g.items()}
        return J, Ja, Jd, g

    def jlocal_and_grads(self, W1, b1, W2, c, X, masks, s_entry,
                         gamma, rho_floor, blocks=()):
        Ji, gi = self.jinv_and_grads(W1, b1, W2, c, X, masks,
                                     blocks=blocks)
        Jd, gd = self.jdisc_and_grads(W1, b1, W2, c, X, s_entry,
                                      rho_floor, blocks=blocks)
        grads = {k: gi[k] + gamma * gd[k] for k in gi
                 if k != "blocks"}
        grads["blocks"] = [
            {key: gi["blocks"][k][key] + gamma * gd["blocks"][k][key]
             for key in gi["blocks"][k]}
            for k in range(len(blocks))]
        return Ji + gamma * Jd, Ji, Jd, grads
