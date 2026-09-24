# Confirmatory semantic audit: deepltl-pointworld-official-eval-50

## Integrity

- Complete: **passed** (5000/5000 records).
- Official/independent mismatches: **0**.

## What the ordinary success rate hides

Overall `P_sat` is **0.877** (4384/5000; 95% exact CI [0.867, 0.886]).

Failure/disposition counts:

- `censored`: 169
- `decided_satisfaction`: 4384
- `forbidden_or_temporal_decided_violation`: 141
- `termination_after_relevant_progress`: 226
- `termination_without_relevant_progress`: 80

## Per-policy view

| Subject | P_sat | Satisfied / episodes | Mean necessary evidence/step |
|---|---:|---:|---:|
| ppo_seed_920001 | 0.823 | 823 / 1000 | 0.464 |
| ppo_seed_920002 | 0.845 | 845 / 1000 | 0.464 |
| ppo_seed_920003 | 0.879 | 879 / 1000 | 0.447 |
| ppo_seed_920004 | 0.899 | 899 / 1000 | 0.471 |
| ppo_seed_920005 | 0.938 | 938 / 1000 | 0.457 |

## Formulas with the largest between-policy P_sat spread

These are descriptive screening results with the formula held fixed.

| Formula ID | P_sat range across policies | Necessary-evidence/step mean range |
|---|---:|---:|
| deepltl_point_019 | 0.500 | 0.063 |
| deepltl_point_035 | 0.500 | 0.000 |
| deepltl_point_018 | 0.500 | 0.055 |
| deepltl_point_033 | 0.450 | 0.138 |
| deepltl_point_042 | 0.450 | 0.000 |
| deepltl_point_017 | 0.400 | 0.000 |
| deepltl_point_020 | 0.400 | 0.095 |
| deepltl_point_049 | 0.350 | 0.043 |
| deepltl_point_007 | 0.300 | 0.066 |
| deepltl_point_013 | 0.300 | 0.068 |

## Interpretation boundary

The audit enriches `P_sat` with trace-level semantic evidence and failure diagnosis. Formula-family aggregates are descriptive: policy comparisons must hold the formula fixed, and structural evidence burden must not be relabeled as policy quality.
