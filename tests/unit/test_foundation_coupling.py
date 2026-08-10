"""A1 boxes (plan 53 v1.2 step A1; 54 v1.10 section 3).

 V-A1-1 Coupling unit — three-link chain: the VM-3(a) 2x3 hand
        literals (derivation in test_growth_port.py, reused
        verbatim as the anchor), a fresh SPAN-masked hand case,
        SGD and Adam step against the textbook formulas written
        here (no production import for the reference side);
        span default = structural no-op (object identity).
 V-A1-2 current-artifact migration — the FX-1 growthport-v1
        artifact (grown x2 + deepened, serialized as HISTORICAL
        slot dicts) loads into Coupling-backed sites and SERVES
        BITWISE against its recorded outputs.
 V-A1-3 multi-body site — FX-2 (ONE shared site, TWO bodies)
        replayed through real train steps, losses and predicts
        BITWISE against the pre-A1 records (the site-shared
        Adam counter box, 52 SR-20/SR-24).
"""
import pickle   # safe: locally produced write-once fixtures
                # only (suite convention, e.g. T-11 boxes)
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "modules" / "Engine",
          REPO / "modules" / "ReferenceNet"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from engine.backends import get_default_backend            # noqa: E402
from reference_net.foundation.coupling import Coupling     # noqa: E402
from reference_net.growth_port import PortSite             # noqa: E402

FIX = REPO / "tests" / "unit" / "fixtures" / \
    "adjustment_baseline"


class _Body:
    """Minimal contract-conformant stub with a FIXED output."""
    def __init__(self, u):
        self._u = np.asarray(u, dtype=np.float64)
        self.out_width = self._u.shape[1]
        self.last_dU = None

    def predict(self, X):
        return self._u

    def n_params(self):
        return 0

    def train_from_grad(self, X, dU, sgd_lr=None):
        self.last_dU = np.asarray(dU)


# ---------------- V-A1-1 ----------------

def test_a1_hand_literals_vm3a():
    """u = [[1,2],[0,3]], A = [[1,0,2],[0,1,1]],
       dH = [[1,0,1],[2,1,0]]  (test_growth_port.py T-3 anchor):
       dA = u^T dH = [[1,0,1],[8,3,2]]
       dU = dH A^T = [[3,1],[2,1]]"""
    bk = get_default_backend()
    u = np.array([[1.0, 2.0], [0.0, 3.0]])
    A = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    dH = np.array([[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    c = Coupling(_Body(u), bk.ingest(A.copy()), bk=bk)
    dA, dU = c.gradients(dH, u)
    assert np.array_equal(np.asarray(dA),
                          [[1.0, 0.0, 1.0], [8.0, 3.0, 2.0]])
    assert np.array_equal(np.asarray(dU),
                          [[3.0, 1.0], [2.0, 1.0]])


def test_a1_sgd_step_textbook():
    bk = get_default_backend()
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    dA = np.array([[0.5, -1.0], [2.0, 0.0]])
    c = Coupling(_Body(np.zeros((1, 2))), bk.ingest(A.copy()),
                 bk=bk)
    c.step(dA, lr=None, sgd_lr=0.1, t=0)
    assert np.array_equal(np.asarray(c.A), A - 0.1 * dA)


def test_a1_adam_step_textbook():
    """Reference side written from the textbook equations here
    (mul-associated family form; no production import)."""
    bk = get_default_backend()
    A0 = np.array([[1.0, -2.0]])
    dA = np.array([[0.3, 0.7]])
    c = Coupling(_Body(np.zeros((1, 1))), bk.ingest(A0.copy()),
                 bk=bk)
    m = np.zeros_like(A0); v = np.zeros_like(A0)
    A_ref = A0.copy()
    for t in (1, 2, 3):
        c.step(dA, lr=0.01, sgd_lr=None, t=t)
        m = 0.9 * m + 0.1 * dA
        v = 0.999 * v + 0.001 * dA * dA
        upd = 0.01 * (m / (1 - 0.9 ** t)) \
            / ((v / (1 - 0.999 ** t)) ** 0.5 + 1e-8)
        A_ref = A_ref - upd
    assert np.array_equal(np.asarray(c.A), A_ref)


def test_a1_span_masked_hand_case():
    """span_in = [0] of a 2-wide source; span_out = [1,2] of a
    3-wide target. u = [[1],[2]] (col 0 of [[1,5],[2,6]]),
    A = [[10, 20]]:
      contribution = u @ A = [[10,20],[20,40]]
      dH (full, 3 cols) restricted to cols [1,2] -> dHs;
      dA = u^T dHs ; dU = dHs A^T   (hand-checked below)."""
    bk = get_default_backend()
    full_u = np.array([[1.0, 5.0], [2.0, 6.0]])
    A = np.array([[10.0, 20.0]])
    c = Coupling(_Body(full_u), bk.ingest(A.copy()), bk=bk,
                 span_in=np.array([0]),
                 span_out=np.array([1, 2]))
    u = c.source(None)
    assert np.array_equal(u, [[1.0], [2.0]])
    assert np.array_equal(np.asarray(c.contribution(u)),
                          [[10.0, 20.0], [20.0, 40.0]])
    dH = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]])
    dA, dU = c.gradients(dH, u)
    # dHs = [[2,3],[1,-1]]; dA = [[1],[2]]^T dHs = [[4,1]]
    # dU_restricted = dHs @ A^T = [[80],[-10]]; scattered back
    # to the body's own 2-wide output coordinates (col 0 =
    # span_in, col 1 unused -> exact zero):
    assert np.array_equal(np.asarray(dA), [[4.0, 1.0]])
    assert np.array_equal(np.asarray(dU),
                          [[80.0, 0.0], [-10.0, 0.0]])


def test_a1_span_default_structural_noop():
    bk = get_default_backend()
    body = _Body(np.array([[1.0, 2.0]]))
    c = Coupling(body, bk.ingest(np.zeros((2, 3))), bk=bk)
    u = c.source(None)
    assert u is body._u          # no masking arithmetic at all


def test_a1_mapping_shim_read_write():
    bk = get_default_backend()
    c = Coupling(_Body(np.zeros((1, 1))),
                 bk.ingest(np.zeros((1, 2))), key=7, bk=bk)
    assert c.get("key") == 7 and c["A"].shape == (1, 2)
    assert "edw" in c and c.get("missing") is None
    c["key"] = 9
    assert c.key == 9


# ---------------- V-A1-2 ----------------

def test_a1_fx1_artifact_loads_and_serves_bitwise():
    net = pickle.loads((FIX / "fx1_artifact.pkl").read_bytes())
    assert all(isinstance(s, Coupling)
               for s in net._port_site.bodies)
    rec = np.load(FIX / "fx1_records.npz")
    out = np.asarray(net._bk.to_numpy(net.predict(rec["Xe"])),
                     dtype=np.float64)
    assert np.array_equal(out, rec["final_pred"])


# ---------------- V-A1-3 ----------------

def test_a1_fx2_multibody_replay_bitwise():
    net = pickle.loads((FIX / "fx2_pre_train.pkl").read_bytes())
    site = net._port_site
    assert isinstance(site, PortSite) and len(site.bodies) == 2
    rec = np.load(FIX / "fx2_records.npz")
    X, y, Xe = rec["X"], rec["y"], rec["Xe"]
    for i in range(rec["losses"].shape[0]):
        loss = float(net.train_step(X, y))
        assert loss == float(rec["losses"][i]), f"step {i}"
        out = np.asarray(net._bk.to_numpy(net.predict(Xe)),
                         dtype=np.float64)
        assert np.array_equal(out, rec["preds"][i]), f"step {i}"
