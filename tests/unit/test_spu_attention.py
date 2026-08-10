"""T6 tests: SPU realization for attention bodies."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "modules" / "Engine"))
sys.path.insert(0, str(REPO / "modules" / "ReferenceNet"))
from reference_net.attention_body import AttentionBody       # noqa: E402
from engine.spu.spu_attention import (                # noqa: E402
    jlocal_attention, self_process_attention)
from engine.spu.spu_loop import skip_reason           # noqa: E402
from engine.spu.spu_objective import (                # noqa: E402
    batch_std, draw_masks)
from engine.spu.spu_policy import validate_spu_policy  # noqa: E402


def make_unit(seed=3, d_in=4, trained=25):
    b = AttentionBody(d_in, seed=seed, d_model=4, n_heads=2, ffn=3)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (24, d_in))
    y = np.sin(X[:, :1])
    for _ in range(trained):
        b.train_step(X, y)
    return b, X


def pol(**kw):
    base = {"spu_enabled": True, "spu_warmup_steps": 0}
    base.update(kw)
    return validate_spu_policy(base)


def test_fd_jlocal_attention_every_param():
    unit, X = make_unit()
    Xs = unit._std_x(X)
    p = pol(spu_rho_floor=0.5, spu_gamma=1.0)
    rng = np.random.default_rng(5)
    masks = draw_masks(rng, 3, unit.d_in, 0.25)
    raw0 = unit._forward(Xs)
    s_entry = batch_std(raw0[:, 0]) * 4.0        # hinge ACTIVE
    _, _, Jd, _ = jlocal_attention(unit, Xs, masks, s_entry, p)
    assert Jd > 0                                 # disc term live
    eps, tol = 1e-6, 5e-5
    _, _, _, G = jlocal_attention(unit, Xs, masks, s_entry, p)
    for k, arr in unit.P.items():
        if k == "bout":
            continue                              # exact-zero, below
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            keep = arr[idx]
            arr[idx] = keep + eps
            Jp, _, _, _ = jlocal_attention(unit, Xs, masks,
                                           s_entry, p)
            arr[idx] = keep - eps
            Jm, _, _, _ = jlocal_attention(unit, Xs, masks,
                                           s_entry, p)
            arr[idx] = keep
            fd = (Jp - Jm) / (2 * eps)
            assert abs(fd - G[k][idx]) < tol * max(1.0, abs(fd)), \
                (k, idx, fd, G[k][idx])


def test_bout_grad_exactly_zero():
    unit, X = make_unit()
    Xs = unit._std_x(X)
    p = pol()
    masks = draw_masks(np.random.default_rng(5), 4, unit.d_in, 0.25)
    raw0 = unit._forward(Xs)
    _, _, _, G = jlocal_attention(unit, Xs, masks,
                                  batch_std(raw0[:, 0]) * 4.0, p)
    assert np.all(G["bout"] == 0.0)


def test_budget_cap_and_event_schema():
    unit, X = make_unit()
    p = pol(spu_S_max=3)
    ev = self_process_attention(unit, X, p, 0)
    assert ev["steps"] <= 3
    assert ev["body_type"] == "attention"
    for key in ("converged", "s_entry", "s_after", "j_inv_before",
                "j_inv_after", "disc_hits", "clips",
                "policy_snapshot"):
        assert key in ev
    assert ev["skip"] is None


def test_determinism_bit_replay():
    outs = []
    for _ in range(2):
        unit, X = make_unit(seed=11)
        self_process_attention(unit, X, pol(), 5)
        outs.append({k: v.copy() for k, v in unit.P.items()})
    assert all(np.array_equal(outs[0][k], outs[1][k])
               for k in outs[0])


def test_adam_moments_untouched():
    unit, X = make_unit()
    m_snap = {k: mv[0].copy() for k, mv in unit._adam.items()}
    t_snap = unit._t
    self_process_attention(unit, X, pol(), 0)
    assert unit._t == t_snap
    assert all(np.array_equal(m_snap[k], unit._adam[k][0])
               for k in m_snap)


def test_relative_convergence_stops_loop():
    unit, X = make_unit()
    ev = self_process_attention(unit, X, pol(spu_tau_rel=1e6), 0)
    assert ev["converged"] and ev["steps"] == 0


def test_eligibility_accepts_attention_rejects_unknown():
    unit, X = make_unit()
    unit._step_count = 50
    p = pol(spu_warmup_steps=0, spu_newborn_steps=100)
    assert skip_reason(unit, len(X), p, 0) is None

    class Dummy:
        BODY_TYPE = "mystery"
        inner, blocks = {}, []
        _x_mu = 0.0
        _step_count = 50
    assert skip_reason(Dummy(), 64, p, 0) == "body_type_unsupported"


def test_update_norm_cap_engages():
    unit, X = make_unit(trained=2)
    ev = self_process_attention(
        unit, X, pol(spu_eta=100.0, spu_p_mask=0.25), 0)
    # p_mask pinned: this test asserts EVERY step clips at an
    # absurd eta — it needs the stated perturbation strength
    assert ev["clips"] == ev["steps"] > 0
