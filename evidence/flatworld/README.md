# FlatWorld evidence index

All retained FlatWorld evidence is scoped under this directory. Setup and the
operational training runbook live in `benchmarks/flatworld/`, not here -- this
directory holds frozen, hash-anchored artifacts only.

## Layout

- `validation/`: environment grounding, official LDBA integration, and
  corpus-freeze checks. Development-only, no learned-policy outcome.
- `ppo/development/training/`: frozen manifest + launch record for the
  development pilot (seed `960000`), mirroring
  `evidence/deepltl_point/development/training/` and
  `evidence/letter_env/ppo/development/training/`.
- `ppo/confirmatory/protocol/`: frozen benchmark protocol.
- `ppo/confirmatory/training/`: campaign manifest, launch record, and
  per-seed manifests for the five confirmatory subjects.
- `ppo/confirmatory/semantic/`: frozen 4,900-episode campaign plan, records,
  summary, report, paired significance analysis, and retained per-seed
  shards.

## Status

**Confirmatory-complete for the PPO pillar.** The pilot seed `960000` was
inspected before seeds `960001`-`960005` were trained to the predeclared
5,000,000-step budget under the frozen confirmatory protocol.

The semantic campaign contains 4,900 complete records (5 subjects x 49
formulas x 20 episodes), zero missing or duplicate cells, and zero
official/independent verdict mismatches. FlatWorld's nine-circle geometry is
a frozen class constant, not a per-episode random draw, so scenario pairing
across the five seeds is verified directly from each record's retained
`start_position` rather than from a map digest. $\Psat=0.8820$
(4,322/4,900); the paired significance analysis (100,000 Monte Carlo
permutations, seed 0) finds 14/49 formulas significant before correction and
6/49 after Bonferroni correction, while the aggregate paired median-range
test is not significant ($p\approx0.997$) -- the two results answer
different questions, not a contradiction. See
`v3.0/paper/sections/evaluation.tex` (\S FlatWorld) for the interpreted
result.

FlatWorld is evaluated with PPO only, by design: algorithmic diversity
(policy-gradient versus value-based learning) is covered by LetterEnv
(`evidence/letter_env/README.md`), and FlatWorld instead contributes
environmental diversity -- continuous positions, overlapping colored zones,
and a longer 500-step horizon.

Checkpoints and training logs themselves live in
`v3.0/.runtime/deep-ltl/experiments/ppo/` (gitignored -- reproducible from
the frozen training plan/manifest, not retained as evidence).
