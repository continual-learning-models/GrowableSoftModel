"""The INSTRUMENT (PLAN Part I S3, S4; R1, R3, R4, R6, R7).

Primitives the Teacher operates. Contains NO pedagogy (never advances a
curriculum by itself) and learns NO domain content. Everything measured,
logged, and reversible.

Primitives: create_student, study, practice (attempt-based, verifier
outside), evaluate, trajectory (S4 metrics + REAL/FALSE verdicts),
growth_report, grow, rollback, attribution, card, list_students.

Storage (R6): students/<id>/{registry.json, events.jsonl,
score_matrix.jsonl, checkpoints/<event>.pkl}.
"""
from __future__ import annotations

import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .net import Network
from .curriculum import accuracy
from .trainer import collect_instability

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "students"

# S4 verdict thresholds (calibrated in WP3, frozen before WP4)
V_MAX = 0.05        # volatility ceiling for REAL
R_MIN = 0.90        # retention floor
SPIKE = 0.15        # jump size that must persist or be called a spike
STUCK_GAIN = 0.01   # best-so-far gain regarded as progress
STUCK_WINDOW = 4


class Instrument:
    def __init__(self, root: Path | str = DEFAULT_ROOT, seed: int = 7):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._students: dict[str, Network] = {}
        self._ckpts: dict[str, dict[str, Network]] = {}
        self.seed = seed

    # ---------- persistence helpers ----------
    def _dir(self, sid):
        d = self.root / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoints").mkdir(exist_ok=True)
        return d

    def _log(self, sid, kind, **fields):
        row = {"ts": round(time.time(), 2), "event": kind, **fields}
        with open(self._dir(sid) / "events.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def _persist(self, sid):
        with open(self._dir(sid) / "student.pkl", "wb") as f:
            pickle.dump(self._students[sid], f)

    def _ensure(self, sid):
        """Lazy-load a student from disk (cross-process persistence)."""
        if sid not in self._students:
            with open(self._dir(sid) / "student.pkl", "rb") as f:
                self._students[sid] = pickle.load(f)
            self._ckpts.setdefault(sid, {})
            cdir = self._dir(sid) / "checkpoints"
            for c in cdir.glob("*.pkl"):
                with open(c, "rb") as f:
                    self._ckpts[sid][c.stem] = pickle.load(f)
        return self._students[sid]

    def _matrix_path(self, sid):
        return self._dir(sid) / "score_matrix.jsonl"

    def _matrix(self, sid):
        p = self._matrix_path(sid)
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    # ---------- primitives ----------
    def create_student(self, sid: str, d_in: int = 3, hidden: int = 16):
        net = Network(d_in, hidden, seed=self.seed)
        self._students[sid] = net
        self._ckpts[sid] = {}
        self._persist(sid)
        (self._dir(sid) / "registry.json").write_text(json.dumps(
            {"id": sid, "d_in": d_in, "hidden": hidden,
             "created": time.time()}, indent=1))
        self._log(sid, "create", hidden=hidden)
        # 'depth' = height of the inclusion tree (tree height)
        return {"id": sid, "params": net.n_params(), "depth": net.depth()}

    def study(self, sid: str, X, y, steps: int = 200):
        """Supervised block (labeled examples). Returns final train MSE."""
        net = self._ensure(sid)
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1, 1)
        mse = None
        for _ in range(steps):
            mse = net.train_step(X, y)
        self._persist(sid)
        self._log(sid, "study", n=len(X), steps=steps, mse=round(mse, 6))
        return {"mse": mse, "steps": steps}

    def attempts(self, sid: str, X, n_attempts: int = 8, sigma: float = 0.05):
        """Practice phase 1: student produces varied attempts (seeded weight
        noise). Verifier lives OUTSIDE (S0). Returns (n, A) answers."""
        net = self._ensure(sid)
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed + len(self._matrix(sid)))
        outs = [net.predict(X)[:, 0]]        # attempt 0 = the student\'s own answer
        for _ in range(n_attempts - 1):
            pert = copy.deepcopy(net)
            pert.W2 = pert.W2 + rng.normal(0, sigma, pert.W2.shape)
            pert.W1 = pert.W1 + rng.normal(0, sigma, pert.W1.shape)
            outs.append(pert.predict(X)[:, 0])
        return np.stack(outs, axis=1).tolist()

    def practice_update(self, sid: str, X, passed_answers, steps: int = 10):
        """Practice phase 2. PROTOCOL: the caller verifies all attempts and
        returns, per problem, a verified-correct answer ONLY where the
        student\'s own answer (attempt 0) FAILED (else None) — practice
        fixes failures it can reach; it never perturbs what is already
        right. Beyond-reach problems are counted (STUCK evidence)."""
        net = self._ensure(sid)
        X = np.asarray(X, float)
        current = net.predict(X)
        # WITHIN-REACH filter: a verified answer far from the student\'s own
        # answer is a lucky hit, not competence -- wrenching the network
        # toward few lucky points destroys it (measured). Practice
        # consolidates what is within reach; acquisition is study\'s job.
        reach = 2.0 * 0.5   # 2 x verifier tolerance, raw units
        keep = [i for i, a in enumerate(passed_answers)
                if a is not None and abs(a - current[i, 0]) <= reach]
        beyond = len(passed_answers) - len(keep)
        if keep:
            # SELF-ANCHORED update: unpassed problems keep the student\'s own
            # current answers as targets (zero-drift anchor); only verified
            # answers pull. Practicing a few problems must consolidate, not
            # rewrite the student (a small-sample full-batch regression on
            # the passes alone provably destroys the network).
            targets = current.copy()
            for i in keep:
                targets[i, 0] = passed_answers[i]
            for _ in range(steps):
                net.train_step(X, targets, sgd_lr=1e-3)   # consolidation mode
        self._persist(sid)
        self._log(sid, "practice", n=len(X), passed=len(keep), beyond=beyond)
        return {"passed": len(keep), "beyond_reach": beyond}

    def evaluate(self, sid: str, suites):
        """suites: list of {name, X, y}. Appends a score-matrix row."""
        net = self._ensure(sid)
        accs = [accuracy(net.predict(np.asarray(s["X"], float)),
                         np.asarray(s["y"], float).reshape(-1, 1))
                for s in suites]
        row = {"t": len(self._matrix(sid)),
               "names": [s["name"] for s in suites],
               "stage_accs": accs}
        with open(self._matrix_path(sid), "a") as f:
            f.write(json.dumps(row) + "\n")
        return {"stage_accs": accs}

    # ---------- S4 trajectory metrics + verdict ----------
    def trajectory(self, sid: str, current_stage: int | None = None,
                   v_max=V_MAX, r_min=R_MIN, spike=SPIKE,
                   stuck_gain=STUCK_GAIN,
                   stuck_window=STUCK_WINDOW):
        # threshold kwargs: param-interface batch S4. MIRROR of
        # core/teaching.trajectory (that file is the documented
        # port of this method) — same names, same defaults.
        rows = self._matrix(sid)
        if len(rows) < 2:
            return {"verdict": "INSUFFICIENT", "n_evals": len(rows)}
        M = np.array([r["stage_accs"] for r in rows])   # (T, K)
        K = M.shape[1]
        k = K - 1 if current_stage is None else current_stage
        cur = M[:, k]
        best_hist = np.maximum.accumulate(cur)
        progress = float(best_hist[-1])
        gain_recent = float(best_hist[-1] - best_hist[-stuck_window]
                            if len(cur) > stuck_window else best_hist[-1])
        # volatility = mean BACKWARD DROP in the window (a steadily rising
        # curve has zero volatility; only regressions count as wiggle)
        w = cur[-stuck_window:]
        vol = float(np.mean(np.maximum(0.0, w[:-1] - w[1:]))) if len(w) > 1 else 0.0
        # retention over earlier stages
        if k > 0:
            peaks = M[:, :k].max(axis=0) + 1e-9
            retention = float(np.min(M[-1, :k] / peaks))
        else:
            retention = 1.0
        # verdict (R3 hysteresis handled by STUCK_WINDOW aggregation)
        spiked = bool(len(cur) >= 3 and cur[-2] - cur[-3] > spike
                      and cur[-1] < cur[-2] - spike / 2)
        if spiked or vol > v_max and gain_recent > 0:
            verdict = "FALSE_SPIKE"
        elif retention < r_min:
            verdict = "FALSE_SWAP"
        elif gain_recent < stuck_gain:
            verdict = "STUCK"
        else:
            verdict = "REAL"
        return {"verdict": verdict, "progress": round(progress, 4),
                "recent_gain": round(gain_recent, 4),
                "volatility": round(vol, 4),
                "retention": round(retention, 4), "n_evals": len(rows)}

    # ---------- growth surface (R4) ----------
    def growth_report(self, sid: str, top: int = 6):
        net = self._ensure(sid)
        rows = sorted(collect_instability(net), key=lambda r: -r[2])[:top]
        return {"candidates": [{"path": r[0], "node": int(r[1]),
                                "instability": round(float(r[2]), 4)}
                               for r in rows],
                "params": net.n_params(), "depth": net.depth()}

    def grow(self, sid: str, k_nodes: int = 2, hidden: int = 16):
        net = self._ensure(sid)
        ckpt_id = f"ckpt_{len(self._ckpts[sid])}"
        self._ckpts[sid][ckpt_id] = copy.deepcopy(net)
        with open(self._dir(sid) / "checkpoints" / f"{ckpt_id}.pkl", "wb") as f:
            pickle.dump(net, f)
        cands = sorted(collect_instability(net), key=lambda r: -r[2])
        grown = []
        for path, j, score, owner in cands:
            if len(grown) >= k_nodes:
                break
            if j in owner.inner \
                    or j in getattr(owner, "_port_js", set()):
                continue
            owner.grow(j, hidden=hidden)
            grown.append(f"{path}[{j}]")
        self._persist(sid)
        self._log(sid, "grow", nodes=grown, ckpt=ckpt_id,
                  params=net.n_params(), depth=net.depth())
        return {"ckpt": ckpt_id, "grown": grown,
                "params": net.n_params(), "depth": net.depth()}

    def rollback(self, sid: str, ckpt: str):
        self._ensure(sid)
        self._students[sid] = copy.deepcopy(self._ckpts[sid][ckpt])
        self._persist(sid)
        self._log(sid, "rollback", ckpt=ckpt)
        return {"restored": ckpt, "params": self._students[sid].n_params()}

    # ---------- attribution (R7) ----------
    def attribution(self, sid: str, suites):
        """For every composite node: distribution of its inner net's mean
        |output| across suites -> which stage 'uses' the grown structure."""
        net = self._ensure(sid)
        out = []

        def walk(n, path=""):
            kids = list(n.inner.items())
            port = getattr(n, "_port_site", None)
            if port is not None:
                # fullwidth port bodies (Growth Interface Reform)
                kids += [(s.get("key", g), s["body"])
                         for g, s in enumerate(port.bodies)]
            for j, inner in kids:
                mags = []
                for s in suites:
                    X = np.asarray(s["X"], float)
                    mags.append(float(np.mean(np.abs(inner.predict(
                        n._std_x(X))[:, 0]))))
                tot = sum(mags) + 1e-12
                out.append({"node": f"{path or 'root'}[{j}]",
                            "distribution": [round(m / tot, 3) for m in mags],
                            "majority_suite": int(np.argmax(mags))})
                walk(inner, f"{path}/{j}" if path else str(j))
        walk(net)
        return {"nodes": out, "suite_names": [s["name"] for s in suites]}

    # ---------- introspection ----------
    def card(self, sid: str):
        net = self._ensure(sid)
        return {"id": sid, "params": net.n_params(), "depth": net.depth(),
                "structure": net.structure(),
                "n_evals": len(self._matrix(sid))}

    def list_students(self):
        return sorted(self._students)
# ---- B3 extension appended to reference_net/instrument.py ----
# describe(): the ANATOMY report (FR-19) — what the user can
# SEE is exactly what the specs can ADDRESS (closure); keyed by
# PERMANENT component ids (kind#seed from the ledgered specs,
# assigned at birth, never renumbered; internal indices are
# attributes for display only). monitor(): the DYNAMICS time
# series (FR-16) at a caller-set cadence — hooks the existing
# training step; no new loop, no thread (C-6).
import json as _json

import numpy as _np


def _component_id(kind, specs):
    seed = (specs or {}).get("structure", {}).get("seed")
    return f"{kind}#{seed}" if seed is not None else None


def describe(host):
    """READ-ONLY structure report, JSON-serializable.
    Positions are CURRENT attributes; permanent ids never
    change (identity design, doc 52). Never mutates state
    (boxed by state hash)."""
    n_ = (lambda a: _np.asarray(host._bk.to_numpy(a))) \
        if hasattr(host, "_bk") else _np.asarray
    rep = {"kind": type(host).__name__,
           "params_total": int(host.n_params()),
           "components": [], "couplings": []}
    # trunk
    if hasattr(host, "W1"):
        rep["components"].append(
            {"id": "trunk", "kind": "trunk",
             "shapes": {"W1": list(n_(host.W1).shape),
                        "W2": list(n_(host.W2).shape)},
             "width": int(host.H)})
    # blocks (position = current index; id = kind#seed when the
    # birth event carries specs; legacy blocks get stable
    # per-artifact ids)
    ledger = getattr(host, "gain_ledger", [])
    block_specs = [r.get("specs") for r in ledger
                   if r.get("event") == "deepen"]
    for i, b in enumerate(getattr(host, "blocks", [])):
        sp = block_specs[i] if i < len(block_specs) else None
        rep["components"].append(
            {"id": _component_id("block", sp)
             or f"block@legacy{i}",
             "kind": "block", "position": i,
             "shapes": {k: list(_np.asarray(
                 n_(b[k])).shape) for k in b.keys()},
             "params": int(b.n_params()) if hasattr(
                 b, "n_params") else None})
    if getattr(host, "loop_block", None) is not None:
        lb = host.loop_block
        rep["components"].append(
            {"id": "loop", "kind": "loop",
             "shapes": {k: list(_np.asarray(n_(lb[k])).shape)
                        for k in lb.keys()}})
    # attention layers
    if hasattr(host, "L") and hasattr(host, "P"):
        for l in range(host.L):
            comp = {"id": f"layer@{l}", "kind": "attn_layer",
                    "position": l}
            if hasattr(host, "heads"):
                comp["heads"] = [int(h.d_h)
                                 for h in host.heads[l]]
            rep["components"].append(comp)
    # couplings (grown bodies + standalone)
    site = getattr(host, "_port_site", None)
    sites = ([("root", site)] if site is not None else []) + \
        sorted(getattr(host, "_port_sites", {}).items())
    for where, s in sites:
        for c in getattr(s, "bodies", []):
            A = _np.asarray(n_(c["A"]))
            rep["couplings"].append(
                {"site": str(where), "key": str(c.get("key")),
                 "A_shape": list(A.shape),
                 "span_in": None if c.span_in is None
                 else [int(i) for i in _np.asarray(c.span_in)],
                 "span_out": None if c.span_out is None
                 else [int(i) for i in
                       _np.asarray(c.span_out)],
                 "body_params": int(c.body.n_params()),
                 "endpoints": {"reads": "per birth specs",
                               "writes": str(where)}})
    _json.dumps(rep)          # JSON-safety is contractual
    return rep


def monitor_configure(host, cadence=50, window=256):
    """Arm the monitoring ring (FR-16): every `cadence`
    training steps a timestamped record of the assessment
    quantities lands in a bounded ring. Caller-set; off by
    default; switchable any time (C-5)."""
    host._monitor = {"cadence": int(cadence),
                     "window": int(window),
                     "count": 0, "records": []}
    return host._monitor


def monitor_tick(host):
    """Called from the host's train path (guard-only when
    unarmed: one attribute check, zero float ops — the
    fixed-path goldens stay bitwise)."""
    mon = getattr(host, "_monitor", None)
    if mon is None:
        return
    mon["count"] += 1
    if mon["count"] % mon["cadence"]:
        return
    from reference_net.method.gates import assess_growth
    rec = {"step": mon["count"],
           "params": int(host.n_params()),
           "assess": assess_growth(host)}
    mon["records"].append(rec)
    if len(mon["records"]) > mon["window"]:
        mon["records"].pop(0)


def monitor_export(host):
    mon = getattr(host, "_monitor", None)
    return _json.dumps(mon["records"] if mon else [])
