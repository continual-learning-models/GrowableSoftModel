"""Param-interface batch S6 (docs/system/22 item 29): max_rules
mining kwarg + config threading."""
from generator.rules import _MAX_RULES, induce_rules


def _rows(n):
    return [{"input": {"f": float(i % 4)},
             "target": "A" if i % 4 < 2 else "B"}
            for i in range(n)]


class TestMaxRules:
    def test_default_cap_preserved(self):
        rl = induce_rules(_rows(32), ["f"], ["A", "B"])
        assert len(rl.rules) <= _MAX_RULES

    def test_cap_honored(self):
        rl = induce_rules(_rows(32), ["f"], ["A", "B"],
                          max_rules=1)
        assert len(rl.rules) <= 1
