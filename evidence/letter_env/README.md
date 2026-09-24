# LetterEnv evidence index

All retained LetterEnv evidence (both the DQN and PPO learner pillars) is
scoped under this directory. Repository-wide machinery evidence remains one
level above because it is not specific to this benchmark. Setup/compatibility
documentation and the operational training runbooks live in
`benchmarks/letter_env/`, not here -- this directory holds frozen,
hash-anchored artifacts only.

## Layout

- `validation/`: environment grounding, official LDBA integration, and
  corpus-freeze checks. Development-only, no learned-policy outcome, and
  shared by both learner pillars -- not duplicated under `dqn/` or `ppo/`.
- `dqn/development/training/`: frozen training plans and the
  pilot-then-confirmatory gate for the DQN pillar.
  `letter_env_dqn_training_plan.json` is the original `letter_env_dqn_v1`
  freeze, kept immutable on its fixed legacy filename, superseded and not
  the plan any current launch targets. The live plan is
  `letter_env_dqn_training_plan_letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized.json`
  -- versioned past `letter_env_dqn_v2_cached` for the throughput,
  curriculum-recency, and evidence-harmonization fixes, all documented in
  `benchmarks/letter_env/DQN_PROTOCOL.md`. Also holds the pilot-then-
  confirmatory gate (`letter_env_dqn_v2_cached_pilot_gate.json` -- the
  filename is not re-versioned alongside the plan, `gate_path()` is fixed),
  which records a state transition over time (`pilot_not_launched` ->
  `pilot_launched` -> `confirmatory_authorized`) rather than a single frozen
  document.
- `dqn/confirmatory/semantic/`: frozen 5,000-episode campaign plan, records,
  summary, report, paired significance analysis, and retained shards for the
  five DQN subjects.
- `ppo/development/training/`: frozen manifest + launch record for the PPO
  pillar's development pilot, mirroring `evidence/deepltl_point/development/training/`
  exactly -- see `benchmarks/letter_env/PPO_PROTOCOL.md`.
- `ppo/confirmatory/protocol/` and `ppo/confirmatory/training/`: frozen
  protocol, campaign manifest, launch record, and per-seed manifests.
- `ppo/confirmatory/semantic/`: frozen 5,000-episode campaign plan, records,
  summary, report, paired significance analysis, and retained shards for the
  five PPO subjects.

## Status

**Both learner pillars are confirmatory-complete and frozen.** The DQN pilot
seed `930000` was inspected and the code-enforced gate records the transition
to `confirmatory_authorized`; seeds `930001`-`930005` then trained to the
predeclared 5,000,000-step checkpoints. The PPO pilot seed `940000` was
inspected before seeds `940001`-`940005` were trained under the frozen
confirmatory protocol; the final frozen checkpoints are at 4,816,896 steps.

Each semantic campaign contains 5,000 complete records (5 subjects x 50
formulas x 20 episodes), zero missing or duplicate cells, and zero
official/independent verdict mismatches. Scenario maps are paired across the
five seeds within each algorithm and verified from retained map digests.
DQN and PPO have separate scenario namespaces, so their cross-algorithm
comparison is explicitly descriptive rather than paired.

Checkpoints and training logs themselves live in `v3.0/experiments/dqn/` and
`v3.0/.runtime/deep-ltl/experiments/ppo/` (both gitignored -- reproducible
from the frozen training plan/manifest, not retained as evidence).
