# Confirmatory semantic audit: deepltl-letterenv-ppo-official-eval-50

## Integrity

- Complete: **passed** (5000/5000 records).
- Official/independent mismatches: **0**.

## What the ordinary success rate hides

Overall `P_sat` is **0.770** (3852/5000; 95% exact CI [0.758, 0.782]).

Failure/disposition counts:

- `censored`: 1039
- `decided_satisfaction`: 3852
- `forbidden_or_temporal_decided_violation`: 109

## Per-policy view

| Subject | P_sat | Satisfied / episodes | Mean necessary evidence/step |
|---|---:|---:|---:|
| ppo_seed_940001 | 0.766 | 766 / 1000 | 0.602 |
| ppo_seed_940002 | 0.776 | 776 / 1000 | 0.627 |
| ppo_seed_940003 | 0.771 | 771 / 1000 | 0.616 |
| ppo_seed_940004 | 0.793 | 793 / 1000 | 0.597 |
| ppo_seed_940005 | 0.746 | 746 / 1000 | 0.584 |

## Formulas with the largest between-policy P_sat spread

These are descriptive screening results with the formula held fixed.

| Formula ID | P_sat range across policies | Necessary-evidence/step mean range |
|---|---:|---:|
| letter_env_011 | 0.450 | 0.082 |
| letter_env_045 | 0.350 | 0.000 |
| letter_env_009 | 0.300 | 0.034 |
| letter_env_004 | 0.300 | 0.078 |
| letter_env_008 | 0.300 | 0.033 |
| letter_env_027 | 0.300 | 0.043 |
| letter_env_031 | 0.300 | 0.052 |
| letter_env_032 | 0.300 | 0.046 |
| letter_env_006 | 0.250 | 0.044 |
| letter_env_012 | 0.250 | 0.032 |

## Interpretation boundary

The audit enriches `P_sat` with trace-level semantic evidence and failure diagnosis. Formula-family aggregates are descriptive: policy comparisons must hold the formula fixed, and structural evidence burden must not be relabeled as policy quality.
