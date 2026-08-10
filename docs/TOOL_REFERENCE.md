# Tool & Verb Reference (static index)

The servers are self-documenting at runtime (MCP
`instructions` + per-tool schemas); this page is the static
index for browsing. Current at release `1.0.0`.

## Library MCP server — 52 tools (`mcp/mcp_server.py`)

Lifecycle / identity: create_model, list_models, card,
get_versions, describe, store, attribution.
Teaching & practice: study, attempts, practice_update, teach,
run_course, self_review, run_self.
Growth (structure): grow, widen, deepen, add_feature,
grow_attention, set_attention_selfproc, refound, remove_block,
remove_grown, remove_loop, loop.
Governance: commit, reset, rollback, set_policy, propose,
plan_validate, plan_run, trial, assess.
Serving & evaluation: infer, evaluate, predict_dist,
trajectory, growth_report, check_drift, discoveries,
innovation_report.
Holdout & data: add_holdout.
Substrates: list_substrates, recommend_substrate.
Standard-method baselines: standard_create, standard_train,
standard_evaluate, standard_infer, standard_save,
standard_load, standard_list.

## Library CLI (`cli/cli.py`)

Thin JSON dispatcher over the same verb surface:
`cli <verb> '<json-args>'` — any tool above is callable by
name; errors return a JSON usage object.

## Evaluative-learning entry points

P-loop: the `rl_trainer` module (`OrganPPORunner` in
`runner.py`, with `set_kl_reference`; `align_to` on the eval
provider); each round appends an `rl_round` audit record,
persisted to the model's `rl_audit.jsonl` via the facade.
S-loop (facade verbs): preference_inspect, preference_reset,
preference_prior_fold. Runbooks: `RL_TRAINER_RUNBOOK.md`,
`PREFERENCE_RUNBOOK.md`.

## SoftModelSystem operator — 50 verbs
## (`sms/operator/api.py`, exposed via its MCP server & CLI)

Data plane: schema_declare, ingest_csv, ingest_rows,
snapshot_build, quality_report, unlearn, data_list, data_show,
data_query, data_verify, data_reindex, data_index, runs_query.
Compute & training: compute_use, compute_mode, train_converge.
Growth & plans: grow, widen, deepen, deepen_scoped, advanced,
propose, plan_validate, plan_run, trial, remove_block,
remove_grown.
Governance & lifecycle: commit, rollback, get_versions,
model_policy, describe, assess, card_generate, bundle_save.
Preference: preference_inspect, preference_reset, prior_fold.
Generation & feedback: generate_answer, generate_dist,
generate_rollout, signal_emit, feedback_serve, feedback_cycle,
teacher_exemplars.
Evaluation: eval_robustness, eval_simbench,
innovation_report.
Acquisition & help: acquire_requests, help.
