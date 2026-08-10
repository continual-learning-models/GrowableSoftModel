"""Param-interface batch S4 (docs/system/22 items 8-12): the five
trajectory-verdict thresholds as kwargs on BOTH surfaces."""
import core.teaching as teaching
import reference_net.instrument as instrument


class TestMirrorEquality:
    def test_threshold_values_identical(self):
        for name in ("V_MAX", "R_MIN", "SPIKE", "STUCK_GAIN",
                     "STUCK_WINDOW"):
            assert getattr(teaching, name) == \
                getattr(instrument, name), name


class _FakeLC:
    """Minimal lc exposing score_matrix for teaching.trajectory."""
    def __init__(self, accs):
        self._rows = [{"t": i, "stage_accs": a}
                      for i, a in enumerate(accs)]

    def score_matrix(self, mid):
        return self._rows


def _verdict(accs, **kw):
    return teaching.trajectory(_FakeLC(accs), "m", **kw)["verdict"]


class TestTeachingKwargs:
    def test_default_verdicts_unchanged(self):
        # steady rise -> REAL under defaults
        rise = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        assert _verdict(rise) == "REAL"
        # flat -> STUCK under defaults
        flat = [[0.5]] * 6
        assert _verdict(flat) == "STUCK"

    def test_r_min_flips_constructed_verdict(self):
        # stage-0 retention 0.80 with current stage rising
        accs = [[1.0, 0.1], [1.0, 0.4], [0.8, 0.9]]
        assert _verdict(accs) == "FALSE_SWAP"          # r_min=0.90
        assert _verdict(accs, r_min=0.75) in ("REAL", "STUCK")

    def test_stuck_gain_flips(self):
        slow = [[0.500], [0.502], [0.504], [0.506], [0.508],
                [0.510]]
        assert _verdict(slow) == "STUCK"               # gain < 0.01
        assert _verdict(slow, stuck_gain=0.001) == "REAL"


class TestInstrumentKwargs:
    def test_same_rule_same_inputs(self, tmp_path):
        ins = instrument.Instrument(root=tmp_path)
        accs = [[1.0, 0.1], [1.0, 0.4], [0.8, 0.9]]
        ins._matrix = lambda sid: [{"stage_accs": a} for a in accs]
        assert ins.trajectory("s")["verdict"] == "FALSE_SWAP"
        assert ins.trajectory("s", r_min=0.75)["verdict"] in (
            "REAL", "STUCK")


class TestSpikeOverride:
    def test_spike_threshold_override_flips(self):
        # jump of 0.20 then partial fall-back: spike under the
        # default 0.15 threshold; not a spike at threshold 0.30
        accs = [[0.30], [0.30], [0.30], [0.30], [0.50],
                [0.40]]
        assert _verdict(accs) == "FALSE_SPIKE"
        assert _verdict(accs, spike=0.30) != "FALSE_SPIKE"
