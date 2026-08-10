"""Course runner (the automatic loop): chains Phase-1 discipline and
Phase-2 growth into ONE callable primitive.

The loop it automates, per curriculum stage:

    study (cumulative) -> evaluate -> trajectory
        target met  -> COMMIT (gate) -> next stage
        STUCK       -> GROW (budget-checked) -> keep studying
        FALSE_*     -> remediate earlier stages (bounded), or PAUSE and
                       return control to the teacher (policy choice)

The teacher (LLM) sets the curriculum and policy, then delegates the
routine loop; the runner stops and reports on anything needing judgment
(FALSE verdicts beyond remediation, growth refusals, budget exhaustion).
Every action lands in the model's event log as usual; commits pass the
same reality gate — the runner has NO special powers.
"""
from __future__ import annotations

from core.lifecycle import Lifecycle
from core import teaching

DEFAULT_RUN_POLICY = {
    "max_blocks_per_stage": 12,   # study blocks before giving up a stage
    "steps_per_block": 300,
    "grow_on_stuck": True,
    "k_nodes": 2,
    "post_grow_blocks": 2,
    "remediate_on_false": True,
    "max_remediations_per_stage": 2,
    "grow_on_persistent_false": True,   # capacity see-saw signal (WP4
                                        # teacher's discovered pattern:
                                        # pacing right + retention sagging
                                        # = capacity, not pedagogy)
    "max_grows_per_stage": 3,
    "commit_each_stage": True,
    # S9-3 (opt-in; default False keeps the loop byte-identical):
    # alternate study blocks with attempt-0 practice blocks — the
    # model answers first; the curriculum's own labels verify; only
    # failed answers receive the verified target (practice_update's
    # within-reach filter applies as shipped).
    "practice_alternate": False,
    "practice_tol": 0.5,
    "practice_n": 64,
}


def run_course(lc: Lifecycle, model_id: str, curriculum: list,
               policy: dict | None = None) -> dict:
    """curriculum: [{name, examples: [rows], holdout: [rows]?,
    suite: {X, y}, target}, ...]
    Returns a full run report; stops early with a reason on anything
    requiring teacher judgment."""
    pol = {**DEFAULT_RUN_POLICY, **(policy or {})}
    suites = [{"name": st["name"], "X": st["suite"]["X"],
               "y": st["suite"]["y"]} for st in curriculum]
    report = {"stages": [], "stopped": None}

    for k, st in enumerate(curriculum):
        # M2 discipline: entering a stage feeds its fresh labeled reality
        # into the model's holdout stream, so the commit gate judges
        # against reality as the curriculum now defines it.
        if st.get("holdout"):
            lc.f.add_holdout(model_id, st["holdout"])
        cum = [r for s in curriculum[:k + 1] for r in s["examples"]]
        stage_log = {"name": st["name"], "blocks": 0, "grows": 0,
                     "remediations": 0, "verdicts": []}
        remediations = 0
        done = False
        for _ in range(pol["max_blocks_per_stage"]):
            lc.study(model_id, cum, steps=pol["steps_per_block"])
            stage_log["blocks"] += 1
            if pol["practice_alternate"] \
                    and stage_log["blocks"] < pol["max_blocks_per_stage"]:
                _practice_block(lc, model_id, st, pol)
                stage_log["blocks"] += 1
                stage_log["practice_blocks"] = \
                    stage_log.get("practice_blocks", 0) + 1
            accs = lc.evaluate(model_id, suites)["stage_accs"]
            traj = teaching.trajectory(lc, model_id, current_stage=k)
            stage_log["verdicts"].append(traj["verdict"])
            if accs[k] >= st["target"]:
                done = True
                break
            needs_growth = False
            if traj["verdict"] in ("FALSE_SPIKE", "FALSE_SWAP"):
                if (pol["remediate_on_false"]
                        and remediations < pol["max_remediations_per_stage"]
                        and k > 0):
                    earlier = [r for s in curriculum[:k] for r in s["examples"]]
                    lc.study(model_id, earlier, steps=pol["steps_per_block"])
                    remediations += 1
                    stage_log["remediations"] = remediations
                    continue
                # Pacing was corrected and the see-saw persists: that is a
                # CAPACITY signal (retention sags because the current scale
                # cannot hold old and new at once) -> escalate to growth.
                if (pol["grow_on_persistent_false"]
                        and stage_log["grows"] < pol["max_grows_per_stage"]):
                    needs_growth = True
                else:
                    report["stages"].append(stage_log)
                    report["stopped"] = {"reason": f"verdict "
                                         f"{traj['verdict']} beyond "
                                         "remediation and growth budgets",
                                         "stage": st["name"],
                                         "trajectory": traj}
                    return report
            if traj["verdict"] == "STUCK" and pol["grow_on_stuck"]:
                if stage_log["grows"] < pol["max_grows_per_stage"]:
                    needs_growth = True
            if needs_growth:
                g = lc.grow(model_id, k_nodes=pol["k_nodes"])
                if g.get("refusal"):
                    report["stages"].append(stage_log)
                    report["stopped"] = {"reason": f"growth refused: "
                                         f"{g['refusal']}",
                                         "stage": st["name"]}
                    return report
                stage_log["grows"] += 1
                remediations = 0          # new capacity: pacing resets
                for _ in range(pol["post_grow_blocks"]):
                    lc.study(model_id, cum, steps=pol["steps_per_block"])
                    stage_log["blocks"] += 1
                    accs = lc.evaluate(model_id, suites)["stage_accs"]
                if accs[k] >= st["target"]:
                    done = True
                    break
        if pol["commit_each_stage"]:
            stage_log["commit"] = lc.commit(model_id,
                                            note=f"course:{st['name']}")
        stage_log["final_accs"] = accs
        stage_log["mastered"] = bool(done)
        report["stages"].append(stage_log)
        if not done:
            report["stopped"] = {"reason": "stage not mastered within "
                                 "block budget", "stage": st["name"]}
            return report
    report["completed"] = True
    return report


def _practice_block(lc, model_id, stage, pol):
    """S9-3 practice: attempt-0 protocol against the stage's own
    labeled rows. The model answers FIRST; the teacher-side labels
    verify; ONLY failures get the verified target."""
    rows = stage["examples"][:pol["practice_n"]]
    inputs = [r["input"] for r in rows]
    answers = lc.attempts(model_id, inputs)
    passed = []
    for r, a in zip(rows, answers):
        own = float(a[0])
        target = float(r["target"])
        passed.append(target if abs(own - target) > pol["practice_tol"]
                      else None)
    lc.practice_update(model_id, inputs, passed)
