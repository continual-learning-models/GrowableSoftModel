"""Loop-operator kernels (K11/K12) and governance helpers.

DESIGN_LOOP_V2 v2.3 SS3-5: the judge (numpy) reference bodies,
single-sourced. NumpyBackend delegates here lazily (the SPU
kernel pattern); TorchBackend ports the same formulas at L2.
"""
import numpy as np

from .backends.numpy_backend import gelu, gelu_d

# max |gelu'(x)| of the RELEASED kernel, measured at L0 over
# [-10, 10] x 20M points: 1.128993069 (attained near x=1.4185).
# Rounded UP at the 6th decimal so the bound stays a bound.
C_G = 1.128994


def _cg_derivation_guard():
    # Import-time derivation check (param-interface batch S6,
    # docs/system/22 section 2.8): C_G must remain a valid upper
    # bound of |gelu'(x)|, tight to ~1e-5 — an activation change
    # cannot silently invalidate the contraction certificate.
    import numpy as np
    from engine.primitives import gelu_d
    x = np.linspace(-10.0, 10.0, 200_001)
    m = float(np.max(np.abs(gelu_d(x))))
    assert m <= C_G < m + 1e-4, (m, C_G)


_cg_derivation_guard()

_LOOP_KEYS = ("loop_enabled", "loop_m", "loop_K_max",
              "loop_tol", "loop_rho_max")


def validate_loop_policy(pol):
    """Loud validation at loop() application (and on overrides).
    Raises ValueError; never adjusts silently."""
    unknown = [k for k in pol
               if k.startswith("loop_") and k not in _LOOP_KEYS]
    if unknown:
        raise ValueError(
            f"unknown loop policy key(s) {sorted(unknown)}; "
            f"valid: {list(_LOOP_KEYS)}")
    if not pol.get("loop_enabled", False):
        raise ValueError(
            "loop_enabled is False: the loop operator is an "
            "opt-in option; set loop_enabled=True to grow cycles")
    m = pol.get("loop_m", 8)
    K = pol.get("loop_K_max", 32)
    tol = pol.get("loop_tol", 1e-6)
    rho = pol.get("loop_rho_max", 0.6)
    if not (isinstance(m, (int, np.integer)) and 1 <= m <= 256):
        raise ValueError(f"loop_m must be an int in [1, 256], got {m!r}")
    if not (isinstance(K, (int, np.integer)) and 1 <= K <= 256):
        raise ValueError(
            f"loop_K_max must be an int in [1, 256], got {K!r}")
    if not (0.0 < float(tol) <= 1e-2):
        raise ValueError(f"loop_tol must be in (0, 1e-2], got {tol!r}")
    if not (0.0 < float(rho) < 1.0):
        raise ValueError(
            f"loop_rho_max must be in (0, 1), got {rho!r}")
    if rho ** K > tol:
        raise ValueError(
            f"infeasible certificate triple: loop_rho_max^loop_K_max "
            f"= {rho ** K:.3e} > loop_tol {tol:.1e}; tighten rho_max, "
            f"raise K_max, or loosen tol")
    return {"loop_m": int(m), "loop_K_max": int(K),
            "loop_tol": float(tol), "loop_rho_max": float(rho)}


def loop_forward(H_L, L_in, b_l, L_out, tol, K_max):
    """K11 — Picard relaxation of  z = H_L + gelu(z Lin^T + b) Lout^T.

    Returns (z_star, k_used, zs). zs holds the z-iterates ONLY
    (the memory pin: u is recomputed in the backward). Hitting
    K_max returns the last iterate — bounded compute is the
    contract, not an error.
    """
    z = H_L
    zs = [z]
    k_used = 0
    for _ in range(int(K_max)):
        z_next = H_L + gelu(z @ L_in.T + b_l) @ L_out.T
        k_used += 1
        delta = float(np.max(np.abs(z_next - z)))
        z = z_next
        zs.append(z)
        if delta < tol:
            break
    return z, k_used, zs


def loop_backward(zs, L_in, b_l, L_out, dz_star):
    """K12 — unrolled BPTT through the EXECUTED iterates.

    The stop index is a constant of the backward (standard
    unrolled convention; exact a.e.). Recursion per DESIGN SS3;
    contraction makes the r-products decay.
    """
    k_hat = len(zs) - 1
    gL_in = np.zeros_like(L_in)
    gb_l = np.zeros_like(b_l)
    gL_out = np.zeros_like(L_out)
    dH_L = np.zeros_like(zs[0])
    r = dz_star
    for k in range(k_hat - 1, -1, -1):
        u = zs[k] @ L_in.T + b_l          # recomputed (memory pin)
        g = gelu(u)
        gL_out += r.T @ g
        s = (r @ L_out) * gelu_d(u)
        gL_in += s.T @ zs[k]
        gb_l += s.sum(axis=0)
        dH_L += r                          # the skip path each step
        r = s @ L_in
    dH_L += r                              # the z0 = H_L seed
    return dH_L, gL_in, gb_l, gL_out


def loop_rho_hat(L_in, L_out, c_g=C_G):
    """Contraction bound  rho_hat = c_g * s_max(L_in) * s_max(L_out).
    Host-side numpy SVD (matrices are m x H — tiny); the torch
    backend reaches this via its to_numpy edge (judge-single-
    sourced norm code)."""
    s_in = float(np.linalg.svd(L_in, compute_uv=False)[0])
    s_out = float(np.linalg.svd(L_out, compute_uv=False)[0])
    return float(c_g * s_in * s_out)
