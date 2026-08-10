"""Backend registry (DESIGN_BACKEND v2.2 §1) — additive-only.

A backend implements the K1-K10 kernel contract
(BACKEND_B0_CENSUS.md). "numpy" is the DEFAULT and the JUDGE:
its kernels are the released code bodies moved verbatim, so the
default world is bitwise-identical to the pre-backend tree.
Adding a backend = one new module + one registry line + a
conformance-kit run (tests/backend_kit/spec.md); existing files
are never edited for a new backend.
"""
from .numpy_backend import NumpyBackend
from .torch_backend import TorchBackend

BACKEND_REGISTRY = {"numpy": NumpyBackend,
                    "torch": TorchBackend}

_default = None


def get_default_backend():
    global _default
    if _default is None:
        _default = NumpyBackend()
    return _default


def resolve_backend(name=None, device=None, dtype=None):
    """Resolve a backend instance. name=None/"numpy" -> the
    default judge; unknown names refused loudly. device/dtype are
    passed through to non-numpy backends (numpy is float64/cpu by
    definition — the judge does not fork)."""
    if name in (None, "numpy"):
        return get_default_backend()
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"unknown compute_backend {name!r}; "
                         f"registered: {sorted(BACKEND_REGISTRY)}")
    return BACKEND_REGISTRY[name](device=device, dtype=dtype)


# ---------------- user surface (B5) ----------------
# P8 discipline: the COMPUTE POLICY is user-selected, never
# self-switched. Models constructed WITHOUT an explicit backend
# consult it; the default is the judge (numpy/cpu/float64).
COMPUTE_POLICY = {"compute_backend": "numpy",
                  "compute_device": "cpu",
                  "compute_dtype": None}      # backend default

_active = None


_F32_PRECISION_NOTICE = (
    "float32 compute selected WITHOUT precision acknowledgment. "
    "float32 carries ~7 significant digits per operation; the "
    "measured training-trajectory deviation from the certified "
    "float64 judge reaches 4e-3 (growth script, deterministic, "
    "doc 61 I-D investigation 2026-07-25 — the deviation is "
    "float32 physics, identical on cpu-f32 and Apple-GPU f32; "
    "our kernels match the PyTorch-official referee at machine "
    "precision on the same device). Accuracy-CERTIFIED compute "
    "is float64 only (numpy judge; torch-cpu-f64 at 1e-13). "
    "Apple GPUs have no float64 hardware (Metal limitation). "
    "To proceed WITH this stated error limit — your explicit "
    "choice (owner doctrine: the system states the limit, the "
    "user decides) — pass acknowledge_f32_precision=True.")


def set_compute_policy(compute_backend=None, compute_device=None,
                       compute_dtype=None,
                       acknowledge_f32_precision=False):
    """SYSTEM INTERFACE: select the compute backend/device/dtype
    for subsequently constructed models. Violations refused
    loudly; returns the applied policy. PRECISION DOOR (owner
    ruling 2026-07-25, doc 61 I-D): float32 compute is NOT
    accuracy-certified — selecting it requires the explicit
    acknowledge_f32_precision=True (the refusal text states
    the measured error limit; the user chooses knowingly).
    float64 selection is untouched. TIER LAW: resolve_backend
    (L1 expert construction) stays ungated."""
    global _active
    pol = dict(COMPUTE_POLICY)
    if compute_backend is not None:
        if compute_backend not in BACKEND_REGISTRY:
            raise ValueError(
                f"unknown compute_backend {compute_backend!r}; "
                f"registered: {sorted(BACKEND_REGISTRY)}")
        pol["compute_backend"] = compute_backend
    if compute_device is not None:
        pol["compute_device"] = compute_device
    if compute_dtype is not None:
        pol["compute_dtype"] = compute_dtype
    pol.pop("f32_precision_acknowledged", None)
    # the door judges the EFFECTIVE precision: the numpy
    # backend always computes float64 (a stale dtype carried
    # by a dtype=None restore call is inert there)
    if (pol.get("compute_backend") != "numpy"
            and str(pol.get("compute_dtype"))
            in ("float32", "float16", "bfloat16")):
        if not acknowledge_f32_precision:
            raise ValueError(_F32_PRECISION_NOTICE)
        pol["f32_precision_acknowledged"] = True
    # constructing the backend validates device/dtype loudly
    _active = resolve_backend(pol["compute_backend"],
                              device=pol["compute_device"],
                              dtype=pol["compute_dtype"])
    # wholesale replace (same dict object): update() alone
    # would strand a stale f32_precision_acknowledged marker
    # after a switch back to float64
    COMPUTE_POLICY.clear()
    COMPUTE_POLICY.update(pol)
    return dict(COMPUTE_POLICY)


def current_backend():
    """The backend models get when constructed without one."""
    return _active if _active is not None \
        else get_default_backend()
