"""Stage-1 unit tests: metric correctness on synthetic organs with
hand-computed expected values."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.plasticity.metrics import (   # noqa: E402
    scale_snapshot, locus, saturation, saturation_report, inversion,
    lp_curve, retention_trend)
from core._modules import reference_net      # noqa: F401, E402
from reference_net.net import Network        # noqa: E402


def _tiny(seed=7):
    return Network(d_in=2, hidden=3, seed=seed)


def test_locus_flat_network_hand_computed():
    net = _tiny()
    s0 = scale_snapshot(net)
    assert list(s0.keys()) == [0]
    norm0 = float(np.linalg.norm(s0[0]))
    net.W1 = net.W1 + 0.5           # move scale 0 only, known amount
    s1 = scale_snapshot(net)
    expect = 0.5 * np.sqrt(net.W1.size) / norm0
    assert abs(locus(s0, s1)[0] - expect) < 1e-9


def test_locus_isolates_the_moving_scale():
    net = _tiny()
    net.grow(0, hidden=2)
    s0 = scale_snapshot(net)
    b0 = net.grown_body(0)
    b0.W1 = b0.W1 + 1.0                         # move ONLY scale 1
    s1 = scale_snapshot(net)
    d = locus(s0, s1)
    assert d[0] == 0.0 and d[1] > 0.0


def test_locus_reports_newborn_scale():
    net = _tiny()
    s0 = scale_snapshot(net)
    net.grow(1, hidden=2)
    d = locus(s0, scale_snapshot(net))
    assert 1 in d and d[1] > 0.0    # grown-since-snap0 scale visible


def test_saturation_hand_computed():
    # uniform vector: cv=0 -> s = mean
    assert abs(saturation(np.array([0.8, 0.8, 0.8])) - 0.8) < 1e-12
    # dispersed vector, hand-computed: mean=0.5, std=0.5 -> cv~1 -> ~0
    assert saturation(np.array([1.0, 0.0])) < 1e-9
    # uniform beats dispersed at equal mean (the widen-vs-deepen core)
    assert (saturation(np.array([0.5, 0.5, 0.5]))
            > saturation(np.array([1.0, 0.4, 0.1])))
    assert saturation(np.zeros(3)) == 0.0


def test_saturation_report_walks_all_networks():
    net = _tiny()
    net.grow(0, hidden=2)
    rep = saturation_report(net)
    assert set(rep) == {"root", "root/0"}
    assert all(0.0 <= v <= 1.0 for v in rep.values())


def test_inversion_zero_for_silent_inner_and_grows_with_inner_output():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 2))
    y = (X[:, :1] * 2.0) + 1.0
    net = _tiny()
    for _ in range(30):
        net.train_step(X, y)
    base = inversion(net, X)
    assert base == 0.0 or base < 1e-9          # no inner nets yet
    net.grow(0, hidden=2)
    assert inversion(net, X) < 1e-9            # zero_out inner: silent
    # force the inner net to speak with a known scale (fullwidth:
    # the ASSEMBLY carries the coupling, so open it too)
    inner = net.grown_body(0)
    inner._x_mu, inner._x_sd = np.zeros(2), np.ones(2)
    inner._y_mu, inner._y_sd = 0.0, 1.0
    inner.W2 = np.ones_like(inner.W2)
    net._port_site.bodies[0]["A"][:] = 1.0
    r1 = inversion(net, X)
    assert r1 > 0.0
    inner.W2 = inner.W2 * 3.0                  # louder inner -> larger R
    assert inversion(net, X) > r1


def test_untrained_network_inversion_is_zero():
    assert inversion(_tiny(), np.zeros((4, 2))) == 0.0


def test_lp_and_retention_hand_computed():
    ev = [{"stage_accs": [0.2, 1.0]}, {"stage_accs": [0.5, 1.0]},
          {"stage_accs": [0.6, 0.9]}, {"stage_accs": [0.6, 0.7]}]
    lp = lp_curve(ev, 0)
    assert np.allclose(lp, [0.3, 0.1, 0.0], atol=1e-9)
    # suite 1 sagging: mean of last-3 LP = (0 - 0.1 - 0.2)/3 = -0.1
    assert abs(retention_trend(ev, 1) - (-0.1)) < 1e-9
    assert lp_curve([], 0) == [] and retention_trend([], 0) == 0.0
