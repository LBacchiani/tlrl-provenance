# DeepLTL PointWorld full-prefix confirmatory semantic report

## Status and correction

This is the primary PointWorld semantic result for paper claims. It replaces
the earlier aggregate-only campaign as the authoritative analysis because the
replacement campaign:

- computes continuation-aware prefix semantics for every episode;
- performs full structural certificate verification for every episode;
- retains the actual labeled/geometric trace used by the audit;
- freezes the current adapter and runner sources in the plan; and
- verifies scenario pairing from the retained zone layouts.

The five already-trained PPO checkpoints were reused without retraining,
checkpoint reselection, or outcome-based policy selection. This is a new
evaluation sample, not a reconstruction of the old trajectories. Exact
MuJoCo trajectories are not guaranteed to replay bit-for-bit, which is why
the evaluated traces themselves are now retained and hash-anchored.

## Experimental design and integrity

```text
5 predeclared PPO checkpoints
× 50 frozen official DeepLTL PointLtl2 formulas
× 20 deterministic episodes per policy/formula
= 5,000 episodes
```

For each `(formula, episode_index)` cell, all five policies received the same
scenario seed and therefore the same generated zone layout. Evaluation seeds
remained unique across records.

All integrity checks passed:

- 5,000/5,000 records present, with no missing or duplicate cells;
- all 5,000 records used `verification_mode = full`;
- 5,000 retained-trace hashes independently recomputed;
- geometry independently reproduced every retained proposition label;
- all 1,000 paired `(formula, episode_index)` cells had one common zone
  layout across the five policies;
- every disposition agreed with its independently computed prefix status;
- zero official/independent mismatches.

The last statement now covers the complete three-valued prefix outcome. The
official monitor's 4,384 asserted satisfactions and 141 asserted violations
matched `definitely_satisfied` and `definitely_violated`, respectively. All
475 episodes for which the official monitor asserted neither outcome were
independently `open`; none concealed a decisive semantic outcome.

## Main result: what `P_sat` omits

The ordinary aggregate result is:

```text
P_sat = 4,384 / 5,000 = 0.8768
95% exact Clopper–Pearson interval: [0.8674, 0.8858]
```

The 616 non-satisfactions are not one homogeneous failure:

| Outcome | Episodes | Rate over all episodes | Share of non-satisfactions |
|---|---:|---:|---:|
| Decided temporal-logic violation | 141 | 2.82% | 22.9% |
| Censored with an open obligation | 169 | 3.38% | 27.4% |
| Environment termination after relevant progress | 226 | 4.52% | 36.7% |
| Environment termination without relevant progress | 80 | 1.60% | 13.0% |

This partition is one concrete demonstration of the broader contribution. A
Boolean success-rate report would merge explicit violation, unresolved
deadline, failed completion after relevant interaction, and failure to engage
with the requirement. The audit separates them without inventing a unique
causal explanation or guaranteed repair. Other benchmarks need not reproduce
these exact four classes: the cross-benchmark question is what additional,
benchmark-specific information becomes visible once `P_sat` is no longer the
only reported outcome.

### Practitioner reading of the four classes

- **Decided violation** is direct evidence that the observed prefix made the
  requirement false. Inspect the forbidden proposition and violating prefix.
- **Censoring** means that the 1,000-step cutoff arrived while the obligation
  remained semantically open. It is deadline non-completion, not a proved
  violation. Whether to change the horizon or the controller depends on
  whether 1,000 steps represents a real operational deadline.
- **Termination after relevant progress** means that the controller contacted
  at least one formula-relevant region before environmental termination. It
  localizes follow-up toward sequence execution, remaining obligations, and
  the physical termination cause; it does not by itself prove wrong ordering.
- **Termination without relevant progress** means no formula-relevant region
  was contacted. Navigation, observability, exploration, reachability, and
  the initial-state distribution are the appropriate first checks.

## Semantic support of successful traces

Necessary and possible evidence describe logical derivations in the observed
trace. They are not action counterfactuals, causal attributions, or direct
physical-robustness measurements.

### Eventually chains

All 1,776 satisfied Eventually-chain traces had exactly zero necessary
evidence. Multiple temporal witnesses leave no single proposition fact in
every derivation. This complete pattern is primarily formula-structural, not
a policy-quality signal.

Possible evidence remains useful for locating alternative witnesses, but zero
necessary evidence does not establish efficient behavior, distribution-shift
robustness, or control margin.

### Until chains

All 2,608 satisfied Until-chain traces had nonzero necessary evidence:

- mean necessary evidence per step: 0.802;
- median: 0.823;
- range: 0.228–0.997.

Until formulas make an avoidance prefix logically load-bearing. Those
intervals are useful inspection and stress-test targets, but a high ratio is
not inherently a bad policy and does not prove that a rollout was a physical
near miss.

The family contrast is exact in this sample—100% zero necessity for satisfied
Eventually chains and 0% for satisfied Until chains—and therefore reinforces
the negative boundary:

> Necessary-evidence burden is not a formula-independent policy-quality
> metric.

## Evidence strength by outcome

Raw evidence counts grow with trace length. All censored episodes have exactly
1,000 steps, so their large raw counts cannot be interpreted as evidence that
the policy was especially purposeful, confused, or close to completion.
Normalized values are more defensible:

| Outcome | Episodes | Mean steps | Mean necessary/step | Mean possible/step | Zero-necessary rate |
|---|---:|---:|---:|---:|---:|
| Decided satisfaction | 4,384 | 288.6 | 0.477 | 0.917 | 40.5% |
| Decided violation | 141 | 128.3 | 0.113 | 1.921 | 83.7% |
| Censored | 169 | 1,000.0 | 0.615 | 1.810 | 21.3% |
| Termination after progress | 226 | 301.0 | 0.408 | 2.062 | 16.8% |
| Termination without progress | 80 | 176.1 | 0.000 | 2.288 | 100.0% |

These measurements characterize logical support, not causal mechanism. In
particular:

- zero necessary evidence in a violation means no single retained
  proposition fact belongs to every derivation of that verdict; it does not
  mean that many independent action choices caused the failure;
- high necessary evidence in a success means many facts are load-bearing
  under the declared deletion semantics; it does not establish fragility to
  sensor noise, dynamics perturbation, or distribution shift; and
- evidence density in a censored prefix does not show how close that prefix
  was to eventual satisfaction.

## Policy-level results

| Policy | Satisfied | `P_sat` | Non-satisfaction profile |
|---|---:|---:|---|
| `ppo_seed_920001` | 823/1,000 | 0.823 | 58 censored, 32 violations, 62 progress terminations, 25 no-progress |
| `ppo_seed_920002` | 845/1,000 | 0.845 | 46 censored, 37 violations, 59 progress terminations, 13 no-progress |
| `ppo_seed_920003` | 879/1,000 | 0.879 | 10 censored, 29 violations, 60 progress terminations, 22 no-progress |
| `ppo_seed_920004` | 899/1,000 | 0.899 | 25 censored, 24 violations, 38 progress terminations, 14 no-progress |
| `ppo_seed_920005` | 938/1,000 | 0.938 | 30 censored, 19 violations, 7 progress terminations, 6 no-progress |

The 11.5-percentage-point spread shows substantial seed-to-seed variation
under identical algorithm, budget, and task corpus. More importantly, the
failure mixture differs: the aggregate improvement of seed `920005` is driven
especially by far fewer environment terminations after relevant progress.
That distinction is invisible in the leaderboard number.

## Formula-fixed policy comparisons

The scenario-paired analysis holds formula and zone layout fixed. With 20
episodes per policy/formula, raw cross-policy ranges still contain substantial
sampling noise, so they were calibrated by 2,000 seeded paired permutations,
with policy labels permuted only within matched scenarios and the add-one
Monte Carlo correction applied to every p-value:

- observed median cross-policy range across formulas: 0.200;
- permutation null mean: 0.149;
- aggregate permutation p-value: 0.0030;
- 12/50 formulas significant at uncorrected `p < 0.05`;
- 2/50 survive Bonferroni correction (`alpha = 0.001`).

The aggregate result supports real requirement-specific policy variation, but
does not license declaring every visible 20-episode gap meaningful. The two
Bonferroni-surviving screens are:

- `deepltl_point_019`,
  `!green U (magenta & (!blue U yellow))`, with success rates 16/20, 16/20,
  10/20, 18/20, and 20/20 across seeds `920001`–`920005`; and
- `deepltl_point_035`,
  `F(yellow & F(green & F(magenta)))`, with success rates 10/20, 16/20,
  14/20, 18/20, and 20/20 across the same seeds.

A descriptive deployment example remains useful with the proper caveat. On
`deepltl_point_002`, the weakest aggregate policy (`920001`) achieved 19/20,
tying the strongest aggregate policy and exceeding seed `920002`'s 15/20.
This cell is not a multiplicity-corrected discovery; it illustrates why a
requirement-level acceptance matrix is safer than assuming the aggregate
ordering holds for every requirement.

## What changes for a practitioner

1. **Report continuation status, not merely `done`.** Of 616
   non-satisfactions, 169 were still open and must not be relabeled as proved
   temporal violations.
2. **Route investigation using the failure partition.** Explicit violation,
   no engagement, engagement followed by termination, and unresolved deadline
   point to different evidence to inspect.
3. **Select policies at requirement level when requirements have unequal
   importance.** Aggregate rank can hide local reversals, while statistical
   calibration prevents overreacting to small cells.
4. **Use provenance as a trace descriptor.** Load-bearing intervals and
   alternative witnesses identify where to inspect or perturb a trace; they
   do not replace `P_sat` or claim causal robustness.
5. **Retain the evaluated trace.** Seeded simulation alone is not enough for
   permanent auditability when continuous dynamics are not bitwise replayable.

## Scope

This result covers one environment, one algorithm (PPO), five independent
training seeds, 50 official formulas, and 20 paired scenarios per cell. It
supports the claim that `P_sat` is semantically lossy and that the missing
information changes diagnosis and model-selection evidence. It does not
establish a universal relationship between evidence burden and controller
quality, nor robustness under unobserved interventions.

No fresh controlled runtime comparison was made for this stronger full-prefix
plus trace-retention mode. The earlier 30-episode overhead measurement is not
used to claim that the present configuration is universally cheap.

## Artifact identity

- Embedded frozen plan digest:
  `f2f84ac872689a82c5f044d2ad5c9b8d2ffbfd75e48ecf852d0bea9e08b42af5`
- Campaign file SHA-256:
  `4d913db1fd82c320fff7be6a7376e80000be632465bef3e83a34998490bab920`
- Plan file SHA-256:
  `606088f05da5d15e41c399d9bb7333bd7399ac5d3b646d713b3b437ed4d611e4`
- Records SHA-256:
  `3e96946ffddd34a7404d3b12dbf7a2460c0a3f0bfde9ef6f23ec5d2dcf5758e5`
- Summary SHA-256:
  `4a236e11512f6732c4faf037ae21150629806352130e72e547a5b843e00d9660`
- Independent full-analysis SHA-256:
  `e802bb329e177c535e3e65dc8dd7884a1c440ac5ec108dab08c64b9d33fa0e96`

The frozen plan records the exact current adapter, semantic engine, campaign
runner, checkpoint, vocabulary, upstream commit, task corpus, Rabinizer jar,
and dependency identities. Each record additionally contains its retained
trace and trace SHA-256.
