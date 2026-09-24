# Confirmatory semantic audit: deepltl-flatworld-official-eval-49

## Integrity

- Complete: **passed** (4900/4900 records).
- Official/independent mismatches: **0**.

## What the ordinary success rate hides

Overall `P_sat` is **0.882** (4322/4900; 95% exact CI [0.873, 0.891]).

Failure/disposition counts:

- `censored`: 416
- `decided_satisfaction`: 4322
- `forbidden_or_temporal_decided_violation`: 161
- `termination_without_relevant_progress`: 1

## Per-policy view

| Subject | P_sat | Satisfied / episodes | Mean necessary evidence/step |
|---|---:|---:|---:|
| ppo_seed_960001 | 0.894 | 876 / 980 | 0.300 |
| ppo_seed_960002 | 0.843 | 826 / 980 | 0.271 |
| ppo_seed_960003 | 0.896 | 878 / 980 | 0.284 |
| ppo_seed_960004 | 0.889 | 871 / 980 | 0.275 |
| ppo_seed_960005 | 0.889 | 871 / 980 | 0.263 |

## Formulas with the largest between-policy P_sat spread

These are descriptive screening results with the formula held fixed.

| Formula ID | P_sat range across policies | Necessary-evidence/step mean range |
|---|---:|---:|
| flatworld_023 | 0.850 | 0.243 |
| flatworld_043 | 0.850 | 0.000 |
| flatworld_011 | 0.750 | 0.038 |
| flatworld_016 | 0.650 | n/a |
| flatworld_012 | 0.600 | 0.165 |
| flatworld_048 | 0.600 | 0.246 |
| flatworld_021 | 0.550 | 0.020 |
| flatworld_003 | 0.350 | 0.104 |
| flatworld_035 | 0.350 | 0.241 |
| flatworld_030 | 0.300 | 0.101 |

## Interpretation boundary

The audit enriches `P_sat` with trace-level semantic evidence and failure diagnosis. Formula-family aggregates are descriptive: policy comparisons must hold the formula fixed, and structural evidence burden must not be relabeled as policy quality.
