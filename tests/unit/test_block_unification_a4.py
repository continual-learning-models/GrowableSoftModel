"""A4 boxes (plan 53 v1.2 step A4; 54 v1.10 section 3).

 V-A4-1 block unification — deepened/looped scopes carry
        contract-conformant structures; OLD dict-block
        artifacts wrap VALUE-PRESERVING on load and serve
        bitwise (FX-1 replay is the standing gate); the
        ON-DISK format stays plain dicts (an artifact written
        now loads in pre-adjustment code).
 V-A4-2 standalone-vs-kernel — Block.predict equals the fused
        block_chain_forward contribution BITWISE on identical
        inputs; Block.train_from_grad's exact chain rule is
        machine-checked by CENTRAL DIFFERENCES (the kernel
        train path trains jointly through the scope optimizer
        and is adjudicated by the replay gates, not here —
        documented deviation, 52 SR-14).
"""
import pickle   # safe: locally produced write-once fixtures
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "modules" / "Engine",
           REPO / "modules" / "ReferenceNet"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from engine.backends import get_default_backend             # noqa: E402
from reference_net.bodies import Block, LoopBlock           # noqa: E402
from reference_net.foundation.contract import conforms      # noqa: E402
from reference_net.net import Network                       # noqa: E402

FIX = REPO / "tests" / "unit" / "fixtures" / \
    "adjustment_baseline"


def _data(seed=101, n=16, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] * X[:, 1] + 0.5 * X[:, 2]).reshape(-1, 1)
    return X, y


def test_a4_deepened_scope_carries_conformant_blocks():
    net = Network(3, 8, lr=1e-2, seed=7)
    net.deepen()
    from reference_net.growthpolicy import DEFAULT_GROWTH_POLICY
    net._growth_policy = dict(DEFAULT_GROWTH_POLICY,
                              loop_enabled=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.loop(4)
    assert isinstance(net.blocks[0], Block)
    assert isinstance(net.loop_block, LoopBlock)
    assert conforms(net.blocks[0]) == []
    assert conforms(net.loop_block) == []
    assert net.blocks[0].n_params() == 8 * 8 + 8 + 8 * 8
    assert net.blocks[0]["Bin"].shape == (8, 8)   # mapping keys
    assert net.gain_ledger[-1]["event"] == "loop"
    assert net.gain_ledger[-1]["specs"]["structure"]["kind"] \
        == "loop"


def test_a4_old_dict_artifact_wraps_value_preserving():
    net = pickle.loads((FIX / "fx1_artifact.pkl").read_bytes())
    assert all(isinstance(b, Block) for b in net.blocks)
    rec = np.load(FIX / "fx1_records.npz")
    out = np.asarray(net._bk.to_numpy(net.predict(rec["Xe"])),
                     dtype=np.float64)
    assert np.array_equal(out, rec["final_pred"])   # bitwise


def test_a4_on_disk_format_stays_plain_dicts():
    net = Network(3, 8, lr=1e-2, seed=7)
    net.deepen()
    st = net.__getstate__()
    assert type(st["blocks"][0]) is dict
    assert set(st["blocks"][0]) == {"Bin", "bb", "Bout"}
    net2 = pickle.loads(pickle.dumps(net))
    assert isinstance(net2.blocks[0], Block)
    X, _ = _data()
    assert np.array_equal(
        np.asarray(net._bk.to_numpy(net.predict(X))),
        np.asarray(net2._bk.to_numpy(net2.predict(X))))


def test_a4_standalone_predict_equals_kernel_bitwise():
    bk = get_default_backend()
    rng = np.random.default_rng(5)
    H = rng.normal(size=(12, 8))
    blk = Block(bk.ingest(rng.normal(size=(6, 8))),
                bk.ingest(rng.normal(size=6)),
                bk.ingest(rng.normal(size=(8, 6))), bk=bk)
    kernel_out = bk.block_chain_forward([blk], bk.ingest(H))
    standalone = H + np.asarray(bk.to_numpy(
        blk.predict(bk.ingest(H))))
    assert np.array_equal(np.asarray(bk.to_numpy(kernel_out)),
                          standalone)


def test_a4_standalone_train_exact_by_central_differences():
    bk = get_default_backend()
    rng = np.random.default_rng(9)
    H = rng.normal(size=(5, 4))
    dU = rng.normal(size=(5, 4))
    Bin = rng.normal(size=(3, 4)); bb = rng.normal(size=3)
    Bout = rng.normal(size=(4, 3))
    blk = Block(bk.ingest(Bin.copy()), bk.ingest(bb.copy()),
                bk.ingest(Bout.copy()), bk=bk)
    lr = 0.5
    blk.train_from_grad(H, dU, sgd_lr=lr)
    # loss surrogate L = sum(dU * contribution): dL/dtheta must
    # match the applied update / lr (exact chain rule); check a
    # sample of coordinates by central differences at 1e-7
    def contrib(Bi, b_, Bo):
        Z = H @ Bi.T + b_
        G = np.asarray(bk.to_numpy(bk.gelu(bk.ingest(Z))))
        return float(np.sum(dU * (G @ Bo.T)))
    eps = 1e-6
    for (i, j) in ((0, 0), (2, 3), (1, 1)):
        Bp = Bin.copy(); Bp[i, j] += eps
        Bm = Bin.copy(); Bm[i, j] -= eps
        g = (contrib(Bp, bb, Bout) - contrib(Bm, bb, Bout)) \
            / (2 * eps)
        applied = (Bin[i, j]
                   - np.asarray(bk.to_numpy(blk.Bin))[i, j]) / lr
        assert abs(g - applied) <= 1e-7 * max(1.0, abs(g))
    for (i, j) in ((0, 0), (3, 2)):
        Bp = Bout.copy(); Bp[i, j] += eps
        Bm = Bout.copy(); Bm[i, j] -= eps
        g = (contrib(Bin, bb, Bp) - contrib(Bin, bb, Bm)) \
            / (2 * eps)
        applied = (Bout[i, j]
                   - np.asarray(bk.to_numpy(blk.Bout))[i, j]) / lr
        assert abs(g - applied) <= 1e-7 * max(1.0, abs(g))
