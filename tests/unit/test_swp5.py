"""SWP5 unit tests (SUT5.1-5.4): the sequence substrate — causality,
temporal learning, kit, and drift end-to-end on a regime switch (the
seismic-stress use-case shape)."""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core._modules import generator  # noqa: F401
from generator.config import Config
from core.substrates.sequence import SequenceSubstrate
from core.facade import System

RNG = np.random.default_rng(0)


def _fit(m, X, y, epochs=200, seed=0):
    r = np.random.default_rng(seed)
    for _ in range(epochs):
        order = r.permutation(len(X))
        for i in range(0, len(X), 32):
            m.train_step(X[order[i:i + 32]], y[order[i:i + 32]])


def test_sut5_1_causality_no_future_leakage():
    X = RNG.uniform(-1, 1, (40, 8, 2))
    y = X[:, -1, :1].copy()
    m = SequenceSubstrate(2, 8, mode="numeric", seed=1)
    _fit(m, X, y, epochs=30)
    # truncated prefixes must be unaffected by later-step perturbations
    X2 = X.copy()
    X2[:, 6, :] += 5.0
    assert np.array_equal(m.predict(X[:, :5, :]), m.predict(X2[:, :5, :]))


def test_sut5_2_learns_temporal_law():
    X = RNG.uniform(-1, 1, (300, 8, 2))
    law = lambda Z: (0.6 * Z[:, -1, 0] + 0.3 * Z[:, -2, 0]
                     + 0.5 * Z[:, -1, 1] * Z[:, -2, 1]).reshape(-1, 1)
    m = SequenceSubstrate(2, 8, mode="numeric", seed=2)
    _fit(m, X, law(X))
    Xh = RNG.uniform(-1, 1, (60, 8, 2))
    mse = float(np.mean((m.predict(Xh) - law(Xh)) ** 2))
    assert mse < 0.05, mse
    # growth preservation on the causal host
    before = m.predict(Xh).copy()
    m.grow_site(m.growth_sites()[0][0])
    assert np.allclose(m.predict(Xh), before)


def test_sut5_3_kit():
    sys.path.insert(0, str(ROOT / "tests" / "substrate_kit"))
    from kit import run_kit
    out = run_kit("sequence")
    assert out.get("pass"), out


def test_sut5_4_drift_e2e_regime_switch():
    """M2 on temporal data: regime A learned and committed -> reality
    switches to regime B -> drift fires -> windowed re-teach via session
    -> committed against the fresh reality."""
    tmp = tempfile.mkdtemp()
    try:
        s = System(Config.from_env(backend="mlp", models_root=Path(tmp)))
        law_a = lambda Z: (0.8 * Z[:, -1, 0] + 0.2 * Z[:, -2, 0]
                           ).reshape(-1, 1)
        law_b = lambda Z: (-0.8 * Z[:, -1, 0] + 0.5 * Z[:, -1, 1]
                           ).reshape(-1, 1)   # regime switch

        def rows(n, law, seed):
            r = np.random.default_rng(seed)
            X = r.uniform(-1, 1, (n, 8, 2))
            return [{"input": x.tolist(), "target": str(float(v))}
                    for x, v in zip(X, law(X)[:, 0])]

        out = s.create_model("seis", holdout=rows(50, law_a, 1))
        assert out["substrate"] == "sequence"      # auto by data form
        for _ in range(3):
            s.study("seis", rows(250, law_a, 2), steps=150)
        r1 = s.commit("seis")
        assert r1["promoted"], r1
        # committed serving works on sequences
        probe = rows(1, law_a, 3)[0]
        o = s.infer("seis", probe["input"])
        assert abs(o["output"] - float(probe["target"])) <= 0.5
        # reality switches: fresh labeled regime-B reality
        s.add_holdout("seis", rows(40, law_b, 4))
        d = s.check_drift("seis", recent_n=40)
        assert d["drifted"] and d["needs_reteach"], d
        # windowed re-teach in session, gated commit on recent reality
        s.set_policy("seis", gate_recent_n=40)
        for _ in range(3):
            s.study("seis", rows(250, law_b, 5), steps=150)
        r2 = s.commit("seis", note="regime B adaptation")
        assert r2["promoted"], r2
        d2 = s.check_drift("seis", recent_n=40)
        assert not d2["drifted"], d2
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_sut5_1_causality_no_future_leakage()
    print("   SUT5.1 causality OK")
    test_sut5_2_learns_temporal_law()
    print("   SUT5.2 temporal law + growth preservation OK")
    test_sut5_3_kit()
    print("   SUT5.3 kit PASS")
    test_sut5_4_drift_e2e_regime_switch()
    print("   SUT5.4 regime-switch drift e2e OK")
    print("swp5 tests passed")
