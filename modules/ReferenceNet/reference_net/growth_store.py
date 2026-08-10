"""growth_store — history & content management for the growth
control surface (doc 55 v1.11 s3-B5/B7; FR-13/FR-15/FR-18 +
the house datastore doctrine: FILES ARE TRUTH, write-once;
SQLite is a derived, rebuildable index — sms/datastore
precedent). THE ONLY NEW MODULE OF PART B (C-6).

Contents:
  snapshot(host, tag)        full device-free state capture
  rollback(host, record)     EXACT in-place restore (bitwise);
                             the next move is solely the
                             CALLER's choice — rollback
                             triggers NO automatic follow-up
                             (FR-13c)
  auto_snapshot(host, gpp)   policy-gated pre-event capture
                             (growth_auto_snapshot, default ON;
                             retention growth_snapshot_keep)
  trial(host, apply_fn, X, y, budget_steps)
                             budgeted probe: snapshot -> apply
                             -> train <= budget -> measure ->
                             UNCONDITIONAL rollback -> report
                             (FR-15); abortable (C-5)
  GrowthStore(root)          the on-disk store: write-once
                             file tree + events.jsonl
                             (append-only) + index.db
                             (SQLite, reindex() rebuilds —
                             dropping it loses NOTHING)

Rollback is FINITE (FR-18): the in-memory ring guarantees only
the retention window; disk-persisted snapshots restore beyond
it at the caller's discretion.
"""
import hashlib
import json
import pickle    # safe: locally produced model states only
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA_VERSION = 1
_DEFAULT_KEEP = 8


# ---------------- snapshots (FR-13) ----------------

def snapshot(host, tag=None):
    """Full state capture: device-free pickle bytes + ledger
    length + a content id. O(model size); read-only."""
    blob = pickle.dumps(host)
    rec = {"id": uuid.uuid4().hex[:12],
           "tag": tag,
           "sha": hashlib.sha256(blob).hexdigest(),
           "ledger_len": len(getattr(host, "gain_ledger", [])),
           "blob": blob}
    ring = getattr(host, "_snapshots", None)
    if ring is None:
        ring = host._snapshots = []
    ring.append(rec)
    keep = int(_policy(host).get("growth_snapshot_keep",
                                 _DEFAULT_KEEP))
    while len(ring) > keep:
        ring.pop(0)                    # FINITE window (FR-18)
    return rec


def rollback(host, record):
    """EXACT in-place restore of the captured state (bitwise —
    parameters, optimizer slots, counters, couplings). Appends
    a 'rollback' ledger event with provenance where the host
    has a ledger. Ends the system's action: what happens next
    is the caller's choice (FR-13c)."""
    obj = pickle.loads(record["blob"])
    discarded = len(getattr(host, "gain_ledger", [])) \
        - record["ledger_len"]
    # OBSERVER state survives a rollback (E2E find, close-out
    # F8): the snapshot ring and the monitoring rig belong to
    # the caller's SESSION, not to the model version — a
    # rollback restores the MODEL, it must not silence the
    # instruments or destroy the remaining history window.
    # G7 (doc 61 I-A): the live BACKEND also belongs to the
    # caller's session — rollback restores the MODEL, never
    # silently migrates the device. Keep the live _bk and
    # re-ingest the restored tensors onto it (the class's own
    # hook); on the numpy judge the hook is an identity.
    keep = {k: host.__dict__[k] for k in
            ("_snapshots", "_monitor", "_bk")
            if k in host.__dict__}
    host.__dict__.clear()
    host.__dict__.update(obj.__dict__)
    host.__dict__.update(keep)
    if "_bk" in keep and hasattr(host, "_ingest_state"):
        host._ingest_state()
    if getattr(host, "gain_ledger", None) is not None:
        # AUDIT-ONLY record: governance events must NEVER enter
        # the gain machinery (_pending_gain untouched — the
        # scale guard learned this lesson first; same law).
        host.gain_ledger.append(
            {"event": "rollback",
             "site": record.get("tag") or record["id"],
             "params_added": 0, "gain": None,
             "trigger": "caller",
             "provenance": {"snapshot": record["id"],
                            "events_discarded": int(
                                max(discarded, 0))}})
    return host


def auto_snapshot(host, gpp):
    """Pre-growth-event capture when the policy key is on
    (default ON). Called by the operator carriers; C-5: the
    switch is the caller's."""
    if gpp.get("growth_auto_snapshot", True):
        return snapshot(host, tag="auto:pre-growth")
    return None


def _policy(host):
    p = getattr(host, "_growth_policy", None)
    if p is not None:
        return p
    try:
        from .growthpolicy import DEFAULT_GROWTH_POLICY
        return DEFAULT_GROWTH_POLICY
    except Exception:            # standalone use
        return {}


# ---------------- bounded trial (FR-15) ----------------

def trial(host, apply_fn, X, y, budget_steps, abort=None):
    """Budgeted probe: snapshot -> apply_fn(host) -> train at
    most budget_steps -> measure -> UNCONDITIONAL rollback.
    Returns the report; applying for real afterwards is a
    separate caller action (FR-15). abort: optional callable
    -> True stops early (C-5 interrupt); rollback still
    unconditional."""
    import time
    t0 = time.perf_counter()
    rec = snapshot(host, tag="trial")
    pre_loss = float(host.train_step(X, y))
    report = {"budget_steps": int(budget_steps),
              "steps_run": 0, "aborted": False,
              "loss_before": pre_loss}
    try:
        apply_fn(host)
        losses = []
        for i in range(int(budget_steps)):
            if abort is not None and abort():
                report["aborted"] = True
                break
            losses.append(float(host.train_step(X, y)))
            report["steps_run"] = i + 1
        report["losses"] = losses
        report["loss_after"] = losses[-1] if losses else None
        report["realized_gain"] = (
            pre_loss - losses[-1]) if losses else None
    finally:
        rollback(host, rec)
        if getattr(host, "gain_ledger", None) is not None:
            host.gain_ledger[-1]["provenance"]["kind"] = "trial"
    # FR-16 (58 D-1.4): total trial duration incl. rollback
    report["wall_ms"] = (time.perf_counter() - t0) * 1e3
    return report


# ---------------- the on-disk store (B7) ----------------

class GrowthStore:
    """Write-once file tree (TRUTH) + derived SQLite index.
      <root>/snapshots/<id>.pkl      (write-once)
      <root>/plans/<sha>.json        (write-once)
      <root>/trials/<id>.json        (write-once)
      <root>/events.jsonl            (append-only)
      <root>/index.db                (derived cache; reindex()
                                      rebuilds from the files)"""

    def __init__(self, root):
        self.root = Path(root)
        for d in ("snapshots", "plans", "trials"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / "index.db")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY "
            "KEY, v TEXT)")
        self._db.execute(
            "INSERT OR REPLACE INTO meta VALUES ('schema', ?)",
            (str(SCHEMA_VERSION),))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER "
            "PRIMARY KEY, ts REAL, kind TEXT, trigger TEXT, "
            "component TEXT, params_added INTEGER, json TEXT)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS files (name TEXT "
            "PRIMARY KEY, kind TEXT, sha TEXT)")
        self._db.commit()

    # ----- truth writes (write-once / append-only) -----

    def _write_once(self, rel, data):
        f = self.root / rel
        if f.exists():
            raise ValueError(f"write-once: {rel} exists")
        f.write_bytes(data if isinstance(data, bytes)
                      else data.encode())
        return f

    def save_snapshot(self, record):
        rel = f"snapshots/{record['id']}.pkl"
        self._write_once(rel, record["blob"])
        self._index_file(rel, "snapshot", record["sha"])
        return rel

    def load_snapshot(self, snap_id):
        blob = (self.root /
                f"snapshots/{snap_id}.pkl").read_bytes()
        return {"id": snap_id, "tag": None, "blob": blob,
                "sha": hashlib.sha256(blob).hexdigest(),
                "ledger_len": 0}

    def save_plan(self, plan):
        text = json.dumps(plan, indent=1, sort_keys=True)
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        rel = f"plans/{sha}.json"
        if not (self.root / rel).exists():
            self._write_once(rel, text)
            self._index_file(rel, "plan", sha)
        return sha

    def save_trial(self, report):
        tid = uuid.uuid4().hex[:12]
        rel = f"trials/{tid}.json"
        self._write_once(rel, json.dumps(report, indent=1,
                                         sort_keys=True))
        self._index_file(rel, "trial", "")
        return tid

    def append_event(self, kind, trigger, component,
                     params_added, payload):
        line = json.dumps(
            {"ts": time.time(), "kind": kind,
             "trigger": trigger, "component": str(component),
             "params_added": int(params_added),
             "payload": payload}, sort_keys=True)
        with open(self.root / "events.jsonl", "a") as f:
            f.write(line + "\n")
        self._index_event(json.loads(line))

    # ----- derived index -----

    def _index_file(self, name, kind, sha):
        self._db.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?)",
            (name, kind, sha))
        self._db.commit()

    def _index_event(self, ev):
        self._db.execute(
            "INSERT INTO events (ts, kind, trigger, component,"
            " params_added, json) VALUES (?,?,?,?,?,?)",
            (ev["ts"], ev["kind"], ev["trigger"],
             ev["component"], ev["params_added"],
             json.dumps(ev, sort_keys=True)))
        self._db.commit()

    def reindex(self):
        """Rebuild the WHOLE db from the file tree — the db is
        cache, never truth (dropping it loses nothing)."""
        self._db.execute("DELETE FROM events")
        self._db.execute("DELETE FROM files")
        ev_path = self.root / "events.jsonl"
        if ev_path.exists():
            for line in ev_path.read_text().splitlines():
                if line.strip():
                    self._index_event(json.loads(line))
        for kind, d in (("snapshot", "snapshots"),
                        ("plan", "plans"),
                        ("trial", "trials")):
            for f in sorted((self.root / d).iterdir()):
                sha = hashlib.sha256(
                    f.read_bytes()).hexdigest() \
                    if kind == "snapshot" else ""
                self._index_file(f"{d}/{f.name}", kind, sha)
        self._db.commit()

    def query_events(self, **where):
        """Filter by kind / trigger / component."""
        q = "SELECT json FROM events"
        keys = [k for k in ("kind", "trigger", "component")
                if k in where]
        if keys:
            q += " WHERE " + " AND ".join(f"{k}=?"
                                          for k in keys)
        rows = self._db.execute(
            q, tuple(where[k] for k in keys)).fetchall()
        return [json.loads(r[0]) for r in rows]
