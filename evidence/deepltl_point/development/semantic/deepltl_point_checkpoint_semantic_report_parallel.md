# DeepLTL PointWorld semantic checkpoint analysis

Scope: development-only formula-conditioned analysis of existing trained checkpoints. No retraining is performed by this report.

## Headline

- Records: `1000`
- Checkpoints: `20`
- Episodes per formula per checkpoint: `1`
- Aggregate `P_sat`: `64.8%`
- Aggregate censoring: `8.7%`
- Official/independent mismatches: `0`

## Interpretation guardrails

- `cross_formula_necessary_evidence_is_descriptive_not_policy_quality`: `True`
- `eventually_chain_zero_necessity_can_be_formula_predictable_under_repeated_witnesses`: `True`
- `formula_family_effects_must_not_be_read_as_controller_robustness`: `True`
- `policy_learning_signal_requires_holding_formula_fixed_across_checkpoints_or_seeds`: `True`

In plain terms: compare policies/checkpoints with the formula held fixed. Do not read raw cross-formula evidence averages as controller robustness.

## By checkpoint

| checkpoint | episodes | P_sat | violations | censored | mean steps | zero-necessary satisfied | necessary/step median | necessary/step range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 0.0% | 50 | 0 | 138.460 | n/a | n/a | n/a-n/a |
| 262144 | 50 | 0.0% | 50 | 0 | 138.440 | n/a | n/a | n/a-n/a |
| 524288 | 50 | 14.0% | 43 | 19 | 640.340 | 57.1% | 0.000 | 0.000-0.954 |
| 786432 | 50 | 30.0% | 35 | 12 | 538.940 | 53.3% | 0.000 | 0.000-0.982 |
| 1048576 | 50 | 24.0% | 38 | 19 | 543.940 | 33.3% | 0.876 | 0.000-0.969 |
| 1310720 | 50 | 28.0% | 36 | 12 | 533.880 | 71.4% | 0.000 | 0.000-0.835 |
| 1572864 | 50 | 62.0% | 19 | 3 | 442.340 | 51.6% | 0.000 | 0.000-0.972 |
| 1835008 | 50 | 80.0% | 10 | 2 | 387.840 | 47.5% | 0.570 | 0.000-0.932 |
| 2097152 | 50 | 78.0% | 11 | 1 | 307.720 | 51.3% | 0.000 | 0.000-0.967 |
| 2359296 | 50 | 92.0% | 4 | 1 | 355.740 | 41.3% | 0.716 | 0.000-0.955 |
| 2621440 | 50 | 94.0% | 3 | 1 | 301.280 | 42.6% | 0.645 | 0.000-0.918 |
| 2883584 | 50 | 84.0% | 8 | 2 | 333.420 | 45.2% | 0.579 | 0.000-0.959 |
| 3145728 | 50 | 90.0% | 5 | 3 | 333.480 | 42.2% | 0.686 | 0.000-0.947 |
| 3407872 | 50 | 96.0% | 2 | 1 | 296.420 | 39.6% | 0.691 | 0.000-0.947 |
| 3670016 | 50 | 82.0% | 9 | 4 | 366.080 | 43.9% | 0.635 | 0.000-0.976 |
| 3932160 | 50 | 80.0% | 10 | 4 | 339.060 | 45.0% | 0.615 | 0.000-0.940 |
| 4194304 | 50 | 92.0% | 4 | 2 | 317.720 | 39.1% | 0.668 | 0.000-0.922 |
| 4456448 | 50 | 94.0% | 3 | 1 | 315.620 | 38.3% | 0.739 | 0.000-0.970 |
| 4718592 | 50 | 84.0% | 8 | 0 | 293.180 | 42.9% | 0.630 | 0.000-0.970 |
| 4980736 | 50 | 92.0% | 4 | 0 | 289.500 | 41.3% | 0.629 | 0.000-0.981 |

## By formula family and checkpoint

| family | checkpoint | episodes | P_sat | mean necessary evidence | zero-necessary satisfied | necessary/step median |
|---|---:|---:|---:|---:|---:|---:|
| eventually_chain | 0 | 20 | 0.0% | 6.050 | n/a | n/a |
| eventually_chain | 262144 | 20 | 0.0% | 2.650 | n/a | n/a |
| eventually_chain | 524288 | 20 | 20.0% | 310.550 | 100.0% | 0.000 |
| eventually_chain | 786432 | 20 | 40.0% | 136.600 | 100.0% | 0.000 |
| eventually_chain | 1048576 | 20 | 20.0% | 413.950 | 100.0% | 0.000 |
| eventually_chain | 1310720 | 20 | 50.0% | 284.200 | 100.0% | 0.000 |
| eventually_chain | 1572864 | 20 | 80.0% | 112.700 | 100.0% | 0.000 |
| eventually_chain | 1835008 | 20 | 95.0% | 44.750 | 100.0% | 0.000 |
| eventually_chain | 2097152 | 20 | 100.0% | 0.000 | 100.0% | 0.000 |
| eventually_chain | 2359296 | 20 | 100.0% | 0.050 | 95.0% | 0.000 |
| eventually_chain | 2621440 | 20 | 100.0% | 0.000 | 100.0% | 0.000 |
| eventually_chain | 2883584 | 20 | 95.0% | 0.000 | 100.0% | 0.000 |
| eventually_chain | 3145728 | 20 | 95.0% | 0.000 | 100.0% | 0.000 |
| eventually_chain | 3407872 | 20 | 95.0% | 39.650 | 100.0% | 0.000 |
| eventually_chain | 3670016 | 20 | 90.0% | 41.200 | 100.0% | 0.000 |
| eventually_chain | 3932160 | 20 | 90.0% | 40.150 | 100.0% | 0.000 |
| eventually_chain | 4194304 | 20 | 90.0% | 47.150 | 100.0% | 0.000 |
| eventually_chain | 4456448 | 20 | 90.0% | 0.000 | 100.0% | 0.000 |
| eventually_chain | 4718592 | 20 | 90.0% | 8.600 | 100.0% | 0.000 |
| eventually_chain | 4980736 | 20 | 95.0% | 1.600 | 100.0% | 0.000 |
| until_chain | 0 | 30 | 0.0% | 21.333 | n/a | n/a |
| until_chain | 262144 | 30 | 0.0% | 31.300 | n/a | n/a |
| until_chain | 524288 | 30 | 10.0% | 231.200 | 0.0% | 0.918 |
| until_chain | 786432 | 30 | 23.3% | 178.633 | 0.0% | 0.872 |
| until_chain | 1048576 | 30 | 26.7% | 148.233 | 0.0% | 0.907 |
| until_chain | 1310720 | 30 | 13.3% | 185.133 | 0.0% | 0.746 |
| until_chain | 1572864 | 30 | 50.0% | 208.633 | 0.0% | 0.871 |
| until_chain | 1835008 | 30 | 70.0% | 191.433 | 0.0% | 0.821 |
| until_chain | 2097152 | 30 | 63.3% | 193.800 | 0.0% | 0.812 |
| until_chain | 2359296 | 30 | 86.7% | 262.067 | 0.0% | 0.834 |
| until_chain | 2621440 | 30 | 90.0% | 169.367 | 0.0% | 0.797 |
| until_chain | 2883584 | 30 | 76.7% | 223.833 | 0.0% | 0.831 |
| until_chain | 3145728 | 30 | 86.7% | 172.700 | 0.0% | 0.817 |
| until_chain | 3407872 | 30 | 96.7% | 196.300 | 0.0% | 0.799 |
| until_chain | 3670016 | 30 | 76.7% | 184.033 | 0.0% | 0.785 |
| until_chain | 3932160 | 30 | 73.3% | 180.133 | 0.0% | 0.791 |
| until_chain | 4194304 | 30 | 93.3% | 182.300 | 0.0% | 0.800 |
| until_chain | 4456448 | 30 | 96.7% | 200.100 | 0.0% | 0.833 |
| until_chain | 4718592 | 30 | 80.0% | 170.733 | 0.0% | 0.831 |
| until_chain | 4980736 | 30 | 90.0% | 172.100 | 0.0% | 0.807 |

## Failure catalogue

| checkpoint | formula | family | disposition | steps | atoms touched | possible | necessary |
|---:|---|---|---|---:|---|---:|---:|
| 0 | deepltl_point_000 | until_chain | environment_termination | 71 | none | 142 | 0 |
| 0 | deepltl_point_001 | eventually_chain | environment_termination | 227 | blue:13, green:26, yellow:24 | 359 | 95 |
| 0 | deepltl_point_002 | until_chain | environment_termination | 144 | none | 288 | 0 |
| 0 | deepltl_point_003 | eventually_chain | environment_termination | 211 | yellow:25 | 422 | 0 |
| 0 | deepltl_point_004 | eventually_chain | environment_termination | 126 | magenta:26 | 378 | 0 |
| 0 | deepltl_point_005 | until_chain | environment_termination | 174 | yellow:21 | 230 | 118 |
| 0 | deepltl_point_006 | until_chain | environment_termination | 93 | none | 186 | 0 |
| 0 | deepltl_point_007 | until_chain | environment_termination | 87 | none | 174 | 0 |
| 0 | deepltl_point_008 | eventually_chain | environment_termination | 83 | none | 166 | 0 |
| 0 | deepltl_point_009 | until_chain | decided_violation | 86 | yellow:1 | 172 | 0 |
| 0 | deepltl_point_010 | until_chain | decided_violation | 73 | green:1 | 146 | 0 |
| 0 | deepltl_point_011 | until_chain | environment_termination | 121 | none | 242 | 0 |
| 0 | deepltl_point_012 | until_chain | decided_violation | 108 | yellow:1 | 216 | 0 |
| 0 | deepltl_point_013 | until_chain | environment_termination | 99 | none | 198 | 0 |
| 0 | deepltl_point_014 | until_chain | environment_termination | 116 | none | 232 | 0 |
| 0 | deepltl_point_015 | eventually_chain | environment_termination | 98 | none | 294 | 0 |
| 0 | deepltl_point_016 | until_chain | environment_termination | 118 | none | 236 | 0 |
| 0 | deepltl_point_017 | eventually_chain | environment_termination | 233 | none | 699 | 0 |
| 0 | deepltl_point_018 | until_chain | environment_termination | 153 | none | 306 | 0 |
| 0 | deepltl_point_019 | until_chain | environment_termination | 231 | magenta:8 | 454 | 97 |
| 0 | deepltl_point_020 | until_chain | environment_termination | 193 | yellow:24 | 362 | 79 |
| 0 | deepltl_point_021 | eventually_chain | environment_termination | 155 | green:24 | 378 | 0 |
| 0 | deepltl_point_022 | eventually_chain | environment_termination | 146 | blue:26 | 266 | 26 |
| 0 | deepltl_point_023 | until_chain | environment_termination | 97 | none | 194 | 0 |
| 0 | deepltl_point_024 | until_chain | environment_termination | 180 | none | 360 | 0 |
| 0 | deepltl_point_025 | until_chain | environment_termination | 82 | none | 164 | 0 |
| 0 | deepltl_point_026 | until_chain | environment_termination | 108 | magenta:23 | 193 | 67 |
| 0 | deepltl_point_027 | eventually_chain | environment_termination | 80 | none | 160 | 0 |
| 0 | deepltl_point_028 | until_chain | environment_termination | 134 | green:14 | 254 | 79 |
| 0 | deepltl_point_029 | eventually_chain | environment_termination | 122 | yellow:21 | 303 | 0 |
| 0 | deepltl_point_030 | eventually_chain | environment_termination | 84 | none | 252 | 0 |
| 0 | deepltl_point_031 | eventually_chain | environment_termination | 165 | blue:19 | 330 | 0 |
| 0 | deepltl_point_032 | eventually_chain | environment_termination | 128 | magenta:28 | 356 | 0 |
| 0 | deepltl_point_033 | until_chain | environment_termination | 102 | none | 204 | 0 |
| 0 | deepltl_point_034 | eventually_chain | environment_termination | 172 | none | 344 | 0 |
| 0 | deepltl_point_035 | eventually_chain | environment_termination | 202 | yellow:9 | 597 | 0 |
| 0 | deepltl_point_036 | until_chain | environment_termination | 112 | magenta:22 | 158 | 66 |
| 0 | deepltl_point_037 | until_chain | environment_termination | 144 | none | 288 | 0 |
| 0 | deepltl_point_038 | eventually_chain | environment_termination | 193 | green:24, magenta:30, yellow:22 | 484 | 0 |
| 0 | deepltl_point_039 | until_chain | environment_termination | 190 | none | 380 | 0 |
| 0 | deepltl_point_040 | until_chain | environment_termination | 65 | none | 130 | 0 |
| 0 | deepltl_point_041 | until_chain | environment_termination | 161 | none | 322 | 0 |
| 0 | deepltl_point_042 | eventually_chain | environment_termination | 97 | none | 291 | 0 |
| 0 | deepltl_point_043 | until_chain | environment_termination | 167 | magenta:11 | 269 | 65 |
| 0 | deepltl_point_044 | eventually_chain | environment_termination | 225 | magenta:47, yellow:15 | 628 | 0 |
| 0 | deepltl_point_045 | until_chain | environment_termination | 227 | none | 454 | 0 |
| 0 | deepltl_point_046 | eventually_chain | environment_termination | 95 | none | 285 | 0 |
| 0 | deepltl_point_047 | eventually_chain | environment_termination | 151 | none | 302 | 0 |
| 0 | deepltl_point_048 | until_chain | environment_termination | 163 | none | 326 | 0 |
| 0 | deepltl_point_049 | until_chain | environment_termination | 131 | green:20 | 193 | 69 |
| 262144 | deepltl_point_000 | until_chain | decided_violation | 44 | green:1 | 88 | 0 |
| 262144 | deepltl_point_001 | eventually_chain | environment_termination | 115 | none | 230 | 0 |
| 262144 | deepltl_point_002 | until_chain | decided_violation | 87 | magenta:1 | 174 | 0 |
| 262144 | deepltl_point_003 | eventually_chain | environment_termination | 101 | none | 202 | 0 |
| 262144 | deepltl_point_004 | eventually_chain | environment_termination | 170 | yellow:50 | 460 | 0 |
| 262144 | deepltl_point_005 | until_chain | environment_termination | 85 | none | 170 | 0 |
| 262144 | deepltl_point_006 | until_chain | environment_termination | 153 | green:28 | 278 | 77 |
| 262144 | deepltl_point_007 | until_chain | decided_violation | 50 | blue:1 | 100 | 0 |
| 262144 | deepltl_point_008 | eventually_chain | environment_termination | 167 | magenta:28 | 306 | 28 |
| 262144 | deepltl_point_009 | until_chain | environment_termination | 204 | blue:14 | 422 | 0 |
| 262144 | deepltl_point_010 | until_chain | environment_termination | 204 | none | 408 | 0 |
| 262144 | deepltl_point_011 | until_chain | environment_termination | 96 | none | 192 | 0 |
| 262144 | deepltl_point_012 | until_chain | environment_termination | 135 | blue:25 | 245 | 76 |
| 262144 | deepltl_point_013 | until_chain | environment_termination | 184 | yellow:26 | 241 | 127 |
| 262144 | deepltl_point_014 | until_chain | environment_termination | 134 | magenta:29 | 197 | 71 |
| 262144 | deepltl_point_015 | eventually_chain | environment_termination | 93 | none | 279 | 0 |
| 262144 | deepltl_point_016 | until_chain | environment_termination | 195 | blue:15 | 287 | 103 |
| 262144 | deepltl_point_017 | eventually_chain | environment_termination | 190 | yellow:24 | 546 | 0 |
| 262144 | deepltl_point_018 | until_chain | environment_termination | 168 | none | 336 | 0 |
| 262144 | deepltl_point_019 | until_chain | environment_termination | 202 | yellow:49 | 263 | 141 |
| 262144 | deepltl_point_020 | until_chain | decided_violation | 107 | magenta:1 | 214 | 0 |
| 262144 | deepltl_point_021 | eventually_chain | environment_termination | 154 | magenta:13 | 449 | 0 |
| 262144 | deepltl_point_022 | eventually_chain | environment_termination | 221 | blue:13 | 429 | 13 |
| 262144 | deepltl_point_023 | until_chain | environment_termination | 121 | none | 242 | 0 |
| 262144 | deepltl_point_024 | until_chain | environment_termination | 89 | none | 178 | 0 |
| 262144 | deepltl_point_025 | until_chain | environment_termination | 218 | blue:47 | 483 | 0 |
| 262144 | deepltl_point_026 | until_chain | environment_termination | 164 | yellow:5 | 229 | 99 |
| 262144 | deepltl_point_027 | eventually_chain | environment_termination | 115 | none | 230 | 0 |
| 262144 | deepltl_point_028 | until_chain | environment_termination | 217 | green:15 | 419 | 156 |
| 262144 | deepltl_point_029 | eventually_chain | environment_termination | 107 | none | 321 | 0 |
| ... | ... | ... | ... | ... | 272 more | ... | ... |

