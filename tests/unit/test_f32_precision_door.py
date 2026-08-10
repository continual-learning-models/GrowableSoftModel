"""float32 precision door + the Apple-GPU third-party referee
(owner ruling 2026-07-25, G-MPS I-D: computation must be
ACCURATE; f32's error limit is stated to the user, who makes
the explicit choice — strategy 2. Boxes written FIRST, strict
TDD; the door boxes are the RED carriers).

Attribution evidence behind the numbers (I-D investigation,
third-party referee doctrine): our forward vs PyTorch
OFFICIAL forward on the SAME Apple GPU = 4.8e-7 (f32 machine
precision — our kernels are faithful); official-only f32 vs
f64 = 3.3-3.8e-7 per forward (cpu and mps ALIKE — the
deviation is f32 physics, not Apple's); identical-state
stepping mps vs f64 = 1e-8..2.6e-7/step, mps == cpu-f32 on
the worst tensor delta; full-script deviation (peak 4.038e-3,
deterministic 5/5) is accumulated f32 STATE drift expressed
through the sharp post-growth re-learning phase, decaying to
2e-4 by step 99. float64 does not exist on Apple GPUs (Metal
hardware limitation; PyTorch MPS errors on float64)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet",
           REPO / "tests" / "unit"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from engine.backends import (set_compute_policy,     # noqa: E402
                             resolve_backend, COMPUTE_POLICY)

try:
    import torch
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    torch = None
    HAS_MPS = False


def _restore():
    set_compute_policy(compute_backend="numpy",
                       compute_device="cpu",
                       compute_dtype="float64")


# ---------------- the door (strategy 2) ----------------

def test_door_f32_refused_without_acknowledgment():
    """Selecting float32 compute WITHOUT the explicit
    acknowledgment refuses loudly, NAMES the flag, STATES
    the error limit (the user must choose knowingly), and
    leaves the active policy unchanged."""
    before = dict(COMPUTE_POLICY)
    with pytest.raises(ValueError) as ei:
        set_compute_policy(compute_backend="torch",
                           compute_device="cpu",
                           compute_dtype="float32")
    msg = str(ei.value)
    assert "acknowledge_f32_precision" in msg
    assert "7 significant digits" in msg
    assert "4e-3" in msg                  # the measured limit
    assert "float64" in msg               # the certified set
    assert dict(COMPUTE_POLICY) == before  # untouched
    _restore()


def test_door_f32_proceeds_with_acknowledgment():
    """With acknowledge_f32_precision=True the same call
    proceeds (the user's explicit, informed choice) and the
    applied policy records the acknowledgment."""
    try:
        pol = set_compute_policy(
            compute_backend="torch", compute_device="cpu",
            compute_dtype="float32",
            acknowledge_f32_precision=True)
        assert pol["compute_dtype"] == "float32"
        assert pol.get("f32_precision_acknowledged") is True
    finally:
        _restore()


def test_door_f64_needs_no_flag():
    """The certified-precision path is untouched: float64
    selection needs no flag (bitwise-identical behavior for
    every existing f64 caller)."""
    try:
        pol = set_compute_policy(compute_backend="torch",
                                 compute_device="cpu",
                                 compute_dtype="float64")
        assert pol["compute_dtype"] == "float64"
        assert "f32_precision_acknowledged" not in pol
    finally:
        _restore()


def test_door_expert_tier_resolve_backend_ungated():
    """TIER LAW (58 E6-1 precedent): resolve_backend is the
    L1 expert surface — direct construction stays ungated
    (tests/experts); the door lives at the POLICY tier."""
    bk = resolve_backend("torch", device="cpu",
                         dtype="float32")
    assert bk is not None                  # no refusal


# ------------- Apple-GPU third-party referee -------------

@pytest.mark.skipif(not HAS_MPS, reason="no mps device")
def test_mps_referee_ours_vs_torch_official():
    """PERMANENT third-party referee ON APPLE GPU (owner
    order: verify with the authoritative library ON the
    hardware): our GA forward vs the 60C official twin
    (F.layer_norm/softmax/gelu-tanh primitives), same state,
    same mps device, f32. Line 2e-6 — one order above the
    measured 4.8e-7 (f32 ulp scale at output ~1; derivation
    in the I-D record); a breach means OUR kernels drifted
    from the official math, not a precision artifact."""
    from test_ga_backend_parity import mk_f1, f1_data
    from test_referee_torch_official import twin_ga_forward
    bk = resolve_backend("torch", device="mps",
                         dtype="float32")
    m = mk_f1(bk)
    data = f1_data()
    for i in range(25):
        m.train_step(*data[i % 3])
    Xp = np.random.default_rng(9).normal(size=(8, 6))
    Xs = m._stdx(Xp)
    ours = np.asarray(m._bk.to_numpy(m._forward(Xs)),
                      dtype=np.float64)
    P = {k: torch.tensor(np.asarray(m._bk.to_numpy(v),
                                    dtype=np.float64),
                         dtype=torch.float32, device="mps")
         for k, v in m.P.items()}
    heads = [[{nm: torch.tensor(
        np.asarray(m._bk.to_numpy(getattr(HS, nm)),
                   dtype=np.float64),
        dtype=torch.float32, device="mps")
        for nm in ("Wq", "Wk", "Wv", "Wo")}
        for HS in layer] for layer in m.heads]
    Xt = torch.tensor(np.asarray(m._bk.to_numpy(Xs),
                                 dtype=np.float64),
                      dtype=torch.float32, device="mps")
    twin = twin_ga_forward(P, heads, m.L,
                           Xt).cpu().double().numpy()
    assert np.max(np.abs(ours - twin)) < 2e-6


# ------------- f32 trajectory envelope canary -------------

@pytest.mark.skipif(not HAS_MPS, reason="no mps device")
def test_mps_f32_trajectory_envelope_record():
    """HARDWARE/PRECISION RECORD, NOT an accuracy pass
    (owner ruling: accuracy = f64 only). Asserts (i) the
    full growth script on mps-f32 is DETERMINISTIC (two
    runs, identical checkpoint errors — the I-D 5/5
    finding), and (ii) the peak deviation from the f64
    judge keeps >= 40% headroom under the envelope constant
    (6e-3) — the canary fires on WORSENING (a kernel
    regression or real defect) long before the row."""
    from test_ga_backend_parity import (mk_f1, f1_data,
                                        _growth_script,
                                        F32_TRAJECTORY_ENVELOPE)
    data = f1_data()

    def run():
        bk = resolve_backend("torch", device="mps",
                             dtype="float32")
        lj = _growth_script(mk_f1(), data)
        lt = _growth_script(mk_f1(bk), data)
        return [abs(a - b) / max(1.0, abs(a))
                for a, b in zip(lj[::10] + lj[-1:],
                                lt[::10] + lt[-1:])]

    e1, e2 = run(), run()
    assert e1 == e2                       # deterministic
    peak = max(e1)
    assert peak * 1.4 <= F32_TRAJECTORY_ENVELOPE, peak


# ------- device-name validation (61C; whitelist at the door) ----

def test_tdev1_bogus_device_refused_at_construction():
    """61C T-dev-1 (RED carrier): a mistyped torch device name
    refuses LOUDLY at construction (torch.device is the
    authoritative validator), never accept-then-crash. The
    message names the bad value and carries torch's own
    legal-list text."""
    with pytest.raises(ValueError) as ei:
        resolve_backend("torch", device="nosuchdev")
    msg = str(ei.value)
    assert "nosuchdev" in msg                # named
    assert "cpu" in msg                      # torch legal list


def test_tdev2_policy_door_atomic_on_bogus_device():
    """61C T-dev-2 (RED carrier): through set_compute_policy
    (f64 — the f32 door is not in play), the bogus name
    refuses and the policy dict is BITWISE untouched (the
    refusal writes NOTHING); a numpy restore still works.
    RED today: no exception AND the dict is corrupted."""
    _restore()
    pre = dict(COMPUTE_POLICY)
    with pytest.raises(ValueError):
        set_compute_policy("torch", "nosuchdev", "float64")
    assert dict(COMPUTE_POLICY) == pre       # untouched
    _restore()                               # still operable


def test_tdev3_whitelist_unbroken_and_door_precedence():
    """61C T-dev-3 (behavior pins, green before and after):
    legal names construct ('cpu' always; 'mps' when
    available); and the f32 PRECISION DOOR fires BEFORE
    device resolution — with f32 and no flag the refusal is
    the door's, not the device check's (precedence pinned)."""
    bk = resolve_backend("torch", device="cpu",
                         dtype="float64")
    assert bk is not None
    if HAS_MPS:
        assert resolve_backend("torch", device="mps",
                               dtype="float32") is not None
    _restore()
    with pytest.raises(ValueError) as ei:
        set_compute_policy("torch", "nosuchdev", "float32")
    assert "acknowledge_f32_precision" in str(ei.value)
    _restore()
