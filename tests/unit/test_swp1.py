"""SWP1 unit tests (SUT1.1-1.4): registry, heads component, contract
adapters, self-describing artifacts. (SUT1.5 full regression runs as the
existing suites, unchanged.)"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.substrates import (REGISTRY, GUIDANCE, get_substrate,
                                     load_artifact)
from core.substrates.base import Substrate, CONTRACT_V
from core.substrates import heads


def test_sut1_1_registry():
    assert "mlp" in REGISTRY and "mlp" in GUIDANCE
    cls = get_substrate("mlp")
    assert issubclass(cls, Substrate) and cls.DATA_FORM == "vector"
    assert get_substrate("nonexistent") is None       # refusal upstream


def test_sut1_2_heads_component():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, (16, 3))
    p = heads.softmax(logits)
    assert np.allclose(p.sum(axis=1), 1.0) and (p > 0).all()
    labels = rng.integers(0, 3, 16)
    loss, grad = heads.ce_loss_and_grad(logits, labels)
    assert loss > 0 and grad.shape == logits.shape
    # gradient sanity: finite-difference on one logit
    eps = 1e-5
    l2 = logits.copy()
    l2[0, 1] += eps
    loss2, _ = heads.ce_loss_and_grad(l2, labels)
    fd = (loss2 - loss) / eps
    assert abs(fd - grad[0, 1] / 16) < 1e-3           # mean-reduced grad
    # vocab growth epsilon
    W2, c = rng.normal(0, 1, (2, 8)), np.zeros(2)
    before = heads.softmax(rng.normal(0, 1, (32, 8)) @ W2.T + c)
    W2g, cg = heads.grow_vocab(W2, c, 8)
    after = heads.softmax(rng.normal(0, 1, (32, 8)) @ W2.T + c)  # same seed data unused; direct check:
    X = rng.normal(0, 1, (32, 8))
    p_old = heads.softmax(X @ W2.T + c)
    p_new = heads.softmax(X @ W2g.T + cg)
    assert np.abs(p_new[:, :2] - p_old).max() < 1e-3
    assert p_new[:, 2].max() < 1e-3


def test_sut1_3_contract_adapters_match_and_preserve():
    cls = get_substrate("mlp")
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 2, (64, 3))
    # numeric
    m = cls(3, 8, mode="numeric", seed=2)
    y = (2 * X[:, 0] - X[:, 2]).reshape(-1, 1)
    for _ in range(100):
        m.train_step(X, y)
    sites = m.growth_sites()
    assert len(sites) == 8
    assert sites == sorted(sites, key=lambda t: -t[1])     # ranked
    before = m.predict(X).copy()
    m.grow_site(sites[0][0])
    assert np.allclose(m.predict(X), before)               # preserved
    # composite parent now excluded; inner sites appear
    sites2 = m.growth_sites()
    assert len(sites2) == 7 + 16                           # 7 atomic + inner 16
    # deep site path parsing: grow an inner-network node
    inner_site = next(sp for sp, _ in sites2 if "/" not in sp
                      and not sp.startswith("root"))
    m.grow_site(inner_site)
    assert m.depth() == 3
    assert np.allclose(m.predict(X), before)               # still preserved
    # categorical
    c = cls(3, 8, mode="categorical", vocab=["A", "B"], seed=3)
    lab = np.where(X[:, 0] > 1, "A", "B")
    for _ in range(100):
        c.train_step(X, lab)
    pb = c.predict_proba(X).copy()
    c.grow_site(c.growth_sites()[0][0])
    assert np.allclose(c.predict_proba(X), pb)


def test_sut1_4_artifact_self_description():
    cls = get_substrate("mlp")
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 2, (32, 3))
    m = cls(3, 8, mode="numeric", seed=5)
    m.train_step(X, (X[:, 0]).reshape(-1, 1))
    tmp = tempfile.mkdtemp()
    try:
        m.save(tmp)
        meta = json.loads((Path(tmp) / "substrate.json").read_text())
        assert meta == {"substrate": "mlp", "contract": CONTRACT_V}
        m2 = load_artifact(tmp)
        assert np.allclose(m2.predict(X), m.predict(X))
        # legacy artifact (no substrate.json) defaults to mlp
        (Path(tmp) / "substrate.json").unlink()
        m3 = load_artifact(tmp)
        assert np.allclose(m3.predict(X), m.predict(X))
        assert m.shape_record()["substrate"] == "mlp"
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_sut1_1_registry()
    test_sut1_2_heads_component()
    test_sut1_3_contract_adapters_match_and_preserve()
    test_sut1_4_artifact_self_description()
    print("swp1 tests passed")
