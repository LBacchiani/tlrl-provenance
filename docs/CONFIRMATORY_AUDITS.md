# Generic confirmatory semantic audits

`scripts/run_confirmatory_semantic_audit.py` is the benchmark-independent
campaign runner. It freezes the entire evaluation design before rollouts,
executes resumable process-isolated shards, and produces results only after an
exact fail-closed completeness check.

## Adapter contract

A benchmark adapter implements four methods:

1. `subjects()` returns learned policies/checkpoints with immutable metadata;
2. `tasks()` returns temporal formulas and optional descriptive families;
3. `provenance()` returns benchmark, grounding, and upstream identities; and
4. `open_session(subject, task)` returns an object whose `run_episode(...)`
   emits the common semantic record fields and whose `close()` releases the
   environment.

Environment rollout, proposition grounding, official verdicts, and the
independent temporal evaluator remain adapter responsibilities. Subject/task
enumeration, seed allocation, plan hashing, parallelism, resume validation,
record merging, exact confidence intervals, and reports are shared machinery.

## Safety model

- Running without `--execute` only resolves inventories and freezes the plan.
- Every evaluation seed is allocated before outcomes exist and checked for
  collisions with declared forbidden namespaces.
- Each worker owns one JSONL shard and fsyncs every completed episode.
- Resume accepts only records with the frozen campaign/plan identity.
- Unexpected, duplicate, missing, or official/independent-mismatching records
  stop the campaign. No partial summary is produced.
- Official success/violation flags are a tri-state interface: exactly one flag
  asserts a decided official verdict, while both false means `undecided`.
  Censoring and environment termination therefore remain distinct from an
  asserted official violation.
- Cross-formula and formula-family evidence differences are descriptive. A
  policy claim must compare policies while holding the formula fixed.

## Frozen campaigns

Each shipped campaign has a frozen configuration in
`evidence/<benchmark>/.../confirmatory/semantic/`:

| Campaign | Frozen configuration | Records |
|---|---|---|
| DeepLTL PointWorld (PPO) | `evidence/deepltl_point/confirmatory/semantic/deepltl_point_full_confirmatory_semantic_campaign.json` | 5,000 |
| LetterEnv (DQN) | `evidence/letter_env/dqn/confirmatory/semantic/letter_env_dqn_confirmatory_campaign_config.json` | 5,000 |
| LetterEnv (PPO) | `evidence/letter_env/ppo/confirmatory/semantic/letter_env_ppo_confirmatory_campaign_config.json` | 5,000 |
| FlatWorld (PPO) | `evidence/flatworld/ppo/confirmatory/semantic/flatworld_ppo_confirmatory_campaign_config.json` | 4,900 |

Running the runner directly against a frozen configuration is refused by
design: the in-place frozen plan records absolute source paths and source
digests, so any difference (another machine, a newer kernel) aborts with
"frozen plan differs". To re-run a campaign safely -- outputs redirected to a
scratch directory, then compared record by record with the frozen corpus --
use `scripts/reproduce_audit.py` (see the top-level README).

To design a new campaign, write a configuration modelled on one of the above
and plan it (this only resolves inventories and freezes the plan):

```bash
python scripts/run_confirmatory_semantic_audit.py --config my_campaign.json
```

After reviewing the frozen plan, add `--execute`.
