# DeepLTL PointWorld evidence index

All retained DeepLTL PointWorld evidence is scoped under this directory.
Repository-wide machinery evidence remains one level above because it is not
specific to this benchmark.

## Layout

- `validation/`: environment grounding, official LDBA integration, Rabinizer,
  and PPO/CUDA runtime checks.
- `development/training/`: the inspected development-pilot training manifest,
  launch record, and completion record.
- `development/semantic/`: the retained 1,000-record checkpoint audit and its
  summary/report. These results are developmental and cannot be promoted to
  confirmatory evidence.
- `confirmatory/protocol/`: the frozen paper-benchmark protocol.
- `confirmatory/training/`: the five-seed training manifests and operational
  launch/recovery records.
- `confirmatory/semantic/`: the primary frozen semantic campaign and plan,
  complete 5,000-record corpus, generated summary, independent validation, and
  practitioner-facing interpretation. Every (formula,
  episode index) cell shares one `scenario_seed` across all five policies, so
  cross-policy comparisons are controlled for scenario difficulty as well as
  formula identity. It reuses the same five PPO checkpoints (no retraining),
  computes full prefix and structural semantics for every episode, and retains
  each actual labeled/geometric trace with its own hash. The independently
  generated `deepltl_point_full_semantic_analysis.json` validates all traces,
  labels, dispositions, and paired layouts and performs the formula-fixed
  permutation analysis.

The main benchmark interpretation is:

`confirmatory/semantic/deepltl_point_semantic_report.md`

## Robustness and completeness verification

`confirmatory/semantic/` also retains a set of controlled checks against
the same five confirmatory checkpoints, reported in the paper's
`\S Robustness and Scope Checks`; none of these belongs to the
confirmatory corpus or contributes to a $\Psat$ estimate:

- `deepltl_point_audit_overhead_measurement.json`: real rollout-vs-audit
  wall-clock timing (10 formulas x 3 episodes, one checkpoint, CPU).
- `deepltl_point_stochastic_execution_stress_test.json`: 50 episodes with
  `deterministic=False`, checking the fail-closed guarantee does not depend
  on the confirmatory campaigns' deterministic-execution choice.
- `deepltl_point_extended_horizon_stress_test.json`: the same seeds and
  formulas rerun at 10x the official 1,000-step horizon.
- `deepltl_point_operator_generality_stress_test.json`: five hand-authored
  $G$/$R$/$W$/$X$/$X_w$ formulas (none in the confirmatory corpus) audited
  against 15 real traces. This is the probe that originally exposed the
  next-boundary completeness defect in `tlrl_provenance.prefix` (a formula
  such as $G(Xp)$ is unsatisfiable on every nonempty finite trace but its
  residual never syntactically collapses); the retained artifact reflects
  the result *after* the fix. It records zero incomplete decisions across
  all 75 formula--trace assessments; the strong-next probe reports all 15
  traces violated (8 by exact search, 7 by syntactic collapse). It exercises
  implementation coverage beyond $F/U$; it does not generalize the
  confirmatory campaign's behavioral findings to those operators.
- `deepltl_point_grounding_noise_stress_test.json`: verdict sensitivity to
  synthetic per-step label corruption on 500 real frozen episodes.

The completeness fix itself is regression-checked by
`evidence/prefix_exact_completeness_verification.json` (one level above,
repository-wide since it covers all three benchmarks' formula corpora), the
retained output of `scripts/verify_prefix_exact_completeness.py`. That
script reruns the fixed `assess_prefix` against all 149 real corpus
formulas under synthetic traces and against every one of this directory's
5,000 frozen confirmatory records directly. The separately implemented
reachability traversal deliberately shares the engine's already-tested
progression and reference-evaluation primitives, so this checks the new
search control logic rather than claiming an independent implementation of
those primitives. The artifact records status agreement, checker bound
hits, decision origins, and incomplete decisions; any mismatch, bound hit,
or incomplete decision makes the script fail.

## Relocation and immutability

The evidence files were relocated after the campaign completed, most recently
from `confirmatory/semantic_full/` to the canonical
`confirmatory/semantic/` directory. The superseded aggregate-only campaign
that previously occupied `confirmatory/semantic/` was removed. Frozen JSON
artifacts were not edited during relocation, so path fields inside them retain
their original launch-time `semantic_full` locations. Current documentation
uses the canonical benchmark-scoped layout.

SHA-256 sidecars travel with their artifacts. Sidecars were added for the
retained merged JSONL corpora and generated report, and the sidecar for the
practitioner report was refreshed after its documented interpretation update.

## Cleanup basis

Before removing worker shards, their JSONL records were compared with the
retained merged corpora as multisets:

| Corpus | Shard records | Merged records | Exact multiset match | Sorted-record digest |
|---|---:|---:|---|---|
| Confirmatory | 5,000 | 5,000 | yes | `ac3f62e942681ea4b1654ad6fc62fa2709d63490899214b88059953fddd82739` |
| Development checkpoint audit | 1,000 | 1,000 | yes | `ddb0184acf7e723e293a5dfbf9659b470bb1a7dec708b27001695f15b1cae675` |

The deleted evidence-root clutter consisted only of:

- shard-local copies and worker stdout/stderr after exact merged-corpus
  equivalence was established;
- smoke, seed-fix, batching, and partial-checkpoint outputs superseded by the
  retained complete development audit;
- preliminary audit subsets superseded by that retained development corpus;
- build, test-cache, bytecode-cache, and wheel-install-check products that are
  reproducible from source.

No artifact from the authoritative full confirmatory audit was deleted. The
removed aggregate-only campaign remains recoverable from Git history.
