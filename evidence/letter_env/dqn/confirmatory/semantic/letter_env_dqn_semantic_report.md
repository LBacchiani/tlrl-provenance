# Confirmatory semantic audit: deepltl-letterenv-dqn-official-eval-50

## Integrity

- Complete: **passed** (5000/5000 records).
- Official/independent mismatches: **0**.

## What the ordinary success rate hides

Overall `P_sat` is **0.769** (3843/5000; 95% exact CI [0.757, 0.780]).

Failure/disposition counts:

- `censored`: 1152
- `decided_satisfaction`: 3843
- `forbidden_or_temporal_decided_violation`: 5

## Per-policy view

| Subject | P_sat | Satisfied / episodes | Mean necessary evidence/step |
|---|---:|---:|---:|
| dqn_seed_930001 | 0.794 | 794 / 1000 | 0.637 |
| dqn_seed_930002 | 0.750 | 750 / 1000 | 0.597 |
| dqn_seed_930003 | 0.765 | 765 / 1000 | 0.638 |
| dqn_seed_930004 | 0.752 | 752 / 1000 | 0.612 |
| dqn_seed_930005 | 0.782 | 782 / 1000 | 0.629 |

## Formulas with the largest between-policy P_sat spread

These are descriptive screening results with the formula held fixed.

| Formula ID | P_sat range across policies | Necessary-evidence/step mean range |
|---|---:|---:|
| letter_env_006 | 0.450 | 0.083 |
| letter_env_021 | 0.400 | 0.053 |
| letter_env_023 | 0.350 | 0.023 |
| letter_env_026 | 0.350 | 0.035 |
| letter_env_007 | 0.300 | 0.025 |
| letter_env_020 | 0.300 | 0.033 |
| letter_env_034 | 0.300 | 0.021 |
| letter_env_040 | 0.300 | 0.040 |
| letter_env_046 | 0.300 | 0.057 |
| letter_env_014 | 0.300 | 0.052 |

## Interpretation boundary

The audit enriches `P_sat` with trace-level semantic evidence and failure diagnosis. Formula-family aggregates are descriptive: policy comparisons must hold the formula fixed, and structural evidence burden must not be relabeled as policy quality.
