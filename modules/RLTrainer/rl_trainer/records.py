"""RewardRecord — ONE schema, two sources (doc 86 §3.2;
FR-4.1 evidence plane). S-loop consumes ledger_gain records;
P-loop consumes env_return records; the gate reads neither
(its own evaluation stream). Channel separation (FR-4.2) is
enforced at the consumers (TX-02)."""

_SOURCES = ("ledger_gain", "env_return")
_OBJECTS = ("structure", "weights")
_REQUIRED = ("source", "object", "scope", "value",
             "baseline", "life_id", "provenance")


def validate_reward_record(rec):
    """Return None if valid, else a message naming the
    offending key (loud, never silent)."""
    if not isinstance(rec, dict):
        return "record must be a dict"
    for k in _REQUIRED:
        if k not in rec:
            return f"missing key {k!r}"
    if rec["source"] not in _SOURCES:
        return (f"unknown source {rec['source']!r} "
                f"(allowed {_SOURCES})")
    if rec["object"] not in _OBJECTS:
        return (f"unknown object {rec['object']!r} "
                f"(allowed {_OBJECTS})")
    if not isinstance(rec["value"], (int, float)):
        return "value must be a number"
    if rec["baseline"] is not None and \
            not isinstance(rec["baseline"], (int, float)):
        return "baseline must be a number or None"
    if ("batch" not in rec) and ("episode" not in rec):
        return "missing key 'batch' or 'episode'"
    if not isinstance(rec["provenance"], dict):
        return "provenance must be a dict"
    return None


def make_env_return_record(episode_id, ret, life_id,
                           episode, provenance):
    rec = {"source": "env_return", "object": "weights",
           "scope": str(episode_id), "value": float(ret),
           "baseline": None, "life_id": str(life_id),
           "episode": int(episode),
           "provenance": dict(provenance)}
    msg = validate_reward_record(rec)
    if msg:
        raise ValueError(f"invalid env_return record: {msg}")
    return rec
