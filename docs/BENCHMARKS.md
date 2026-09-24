# v3 benchmark program

The semantic machinery is formula-general. Benchmarks are added separately to
test whether its semantic information is useful for deep temporal-logic-guided
RL; they are not used to redefine the machinery around one formula shape.

## Benchmark 1: DeepLTL PointWorld

The adapter targets the official `PointLtl2-v0` environment from DeepLTL
(ICLR 2025). It is deliberately formula-rich: the frozen official 50-task
evaluation dataset contains 20 ordered eventuality formulas and 30
avoidance/until formulas over four grounded zone propositions. The official
learned subject is a formula-conditioned recurrent PPO controller.

**Status: confirmatory-complete.** Five independently trained PPO policies
(seeds `920001`-`920005`, 5,000,000 steps each) were audited against all 50
formulas, 20 scenario-paired episodes per policy/formula (5,000 episodes
total), with every (formula, episode index) cell sharing one scenario seed
across all five policies so cross-policy comparisons are controlled for
scenario difficulty as well as formula identity. Zero official/independent
verdict mismatches. The interpretation, including significance testing for
the requirement-specific divergence claim, a failure-class evidence-strength
breakdown, is at
`evidence/deepltl_point/confirmatory/semantic/deepltl_point_semantic_report.md`.
Layout and artifact identity: `evidence/deepltl_point/README.md`. Training
protocol (historical record of an already-closed campaign, not an active
runbook): `benchmarks/deepltl_point/PPO_PROTOCOL.md`.

A separate, earlier development pilot (reserved seed `910001`, reduced
budget) established runtime readiness only and contributes no policy or
usefulness result; see `evidence/deepltl_point/development/`.

The adapter lives in `src/tlrl_benchmarks/deepltl_point`. It has no dependency
on any earlier prototype and does not vendor or import the DeepLTL repository at module import
time. It is frozen: other benchmarks never import from it or depend on its
internals, only copy patterns where needed.

The canonical `confirmatory/semantic/` corpus computes prefix semantics and
structural verification for all 5,000 episodes and retains hash-anchored
replayable traces.

### Critical interpretation rules

- A trace position is the proposition set after a physical action and before
  the corresponding official LDBA transition.
- Proposition labels are recomputed from raw agent/zone geometry. The official
  `info["propositions"]` value is retained only as a mandatory agreement check.
- Official success and violation flags are independently checked against the
  proof-producing temporal semantics.
- A time-limit truncation is censored. An unfulfilled eventuality at that point
  is not reported as a decided violation.
- The upstream `RemoveTruncWrapper` destroys the terminated/truncated
  distinction. Audit rollouts bypass only that outer wrapper and execute its
  inner Gymnasium `TimeLimit`, retaining the five-value step API.
- A wall or other environment termination without an official TL flag is kept
  separate from temporal satisfaction and violation.
- When the environment is randomized per episode, any per-requirement
  cross-policy comparison must use the same scenario for every policy being
  compared, not just the same formula -- otherwise divergence is confounded
  by scenario difficulty rather than measuring policy behavior.

Corpus freeze is documented in `benchmarks/deepltl_point/CORPUS_FREEZE.md`.
Regenerate the manifest with:

```bash
python scripts/freeze_deepltl_point_manifest.py
```

The real-environment smoke is reproducible in an isolated Python 3.10 runtime
using `scripts/smoke_deepltl_point_environment.py`. Its generated evidence is
development-only and records no trained-policy or usefulness outcome.

Rabinizer compatibility (shared with Benchmark 2, see below) is frozen in
`benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md`. Reproduce the
50-formula conversion audit with `scripts/verify_deepltl_rabinizer.py` and the
official wrapper-stack check with
`scripts/smoke_deepltl_official_ldba_env.py`. Both scripts produce hashed,
development-only evidence and inspect no learned-policy outcome.

## Benchmark 2: DeepLTL LetterEnv (DQN and PPO)

The adapter targets the official `LetterEnv-v0` environment from DeepLTL: a
7x7 toroidal grid with 12 letter propositions (two copies of each), the
official `LETTER_CURRICULUM`, and a frozen 50-formula evaluation corpus with
paired official world layouts. It contains two independently trained policy
families: a new Double-DQN value learner reusing DeepLTL's LetterEnv encoders,
and the official formula-conditioned recurrent PPO controller. Their purpose
is to test the audit under another environment, action space, and learning
setup. The paper does not treat their aggregate difference as a controlled
causal comparison between algorithms.

**Status: confirmatory-complete for both algorithms.** Five DQN policies
(seeds `930001`-`930005`, 5,000,000 steps each) and five PPO policies (seeds
`940001`-`940005`, final frozen checkpoints at 4,816,896 steps) were each
audited against all 50 formulas with 20 episodes per policy/formula: 5,000
DQN episodes and 5,000 PPO episodes. Both campaigns are complete, retain
zero official/independent verdict mismatches, and are frozen under
`evidence/letter_env/{dqn,ppo}/confirmatory/semantic/`.

The paired cross-seed tests (100,000 Monte Carlo permutations, seed 0) found
no corrected formula-level differences for either learner: DQN had 3/50
uncorrected tests and 0/50 Bonferroni survivors (aggregate permutation
p=0.347); PPO had 2/50 uncorrected tests and 0/50 Bonferroni survivors
(aggregate p=0.917). These conservative negative
tests do not erase the semantic findings: the two nearly identical aggregate
satisfaction rates (`0.769` DQN, `0.770` PPO) conceal materially different
outcome compositions -- DQN produced 1,152 censored episodes and 5 decided
violations, whereas PPO produced 1,039 censored episodes and 109 decided
violations. This is precisely the benchmark's central role: demonstrating
that `P_sat` omits operationally important information, not requiring one
favored phenomenon to recur everywhere.

Layout and artifact identity: `evidence/letter_env/README.md`. Historical
training protocols: `benchmarks/letter_env/DQN_PROTOCOL.md` and
`benchmarks/letter_env/PPO_PROTOCOL.md`.

The confirmatory design uses the environment's official random-map reset
distribution. Within each algorithm, a `scenario_seed` is shared across all
five subjects per (formula, episode index), and the retained map digests
verify the intended pairing before significance testing. DQN and PPO used
separate scenario namespaces, so cross-algorithm comparisons are descriptive,
not paired. The official fixed world layouts remain corpus provenance and a
canonical compatibility check, not the confirmatory scenario pool.

The adapter lives in `src/tlrl_benchmarks/letter_env`. It is fully
self-contained: it shares no code with the `deepltl_point` adapter, even
where the same pattern is needed, so nothing here can silently invalidate
Benchmark 1's frozen evidence.

Corpus freeze is documented in `benchmarks/letter_env/CORPUS_FREEZE.md` and
`evidence/letter_env/validation/`. LDBA/Rabinizer translation for this
benchmark's confirmatory evaluation depends on the same shared runtime
compatibility patches as Benchmark 1 -- see
`benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md`; verified for
`LetterEnv-v0` independently in
`evidence/letter_env/validation/letter_env_official_ldba_env_smoke.json`.

## Benchmark 3: FlatWorld (PPO)

The adapter targets the official `FlatWorld-v0` environment from DeepLTL:
continuous 2D positions, nine discrete actions (eight compass directions plus
stay), nine fixed colored circles (frozen class constant, not a per-episode
random draw), and a frozen 49-formula evaluation corpus (23 nested-eventually
chains, 26 until chains). Unlike LetterEnv's zero-or-one-proposition contract
and PointLtl2's disjoint zones, FlatWorld's zones may overlap, so more than
one proposition can be true at once. FlatWorld extends the evaluation's
*environmental* diversity, not its algorithmic diversity: algorithmic
diversity (policy-gradient versus value-based learning) is Benchmark 2's
role. FlatWorld is evaluated with the official DeepLTL formula-conditioned
recurrent PPO controller only, unmodified.

**Status: confirmatory-complete.** Five independently trained PPO policies
(seeds `960001`-`960005`, 5,000,000 steps each) were audited against all 49
formulas, 20 scenario-paired episodes per policy/formula (4,900 episodes
total). Because FlatWorld's geometry is a frozen constant, scenario pairing
is verified directly from each record's own retained `start_position`
rather than from a map digest. Zero official/independent verdict
mismatches. `P_sat = 0.8820` (4,322/4,900). The paired cross-seed tests
(100,000 Monte Carlo permutations, seed 0) found 14/49 uncorrected and 6/49
Bonferroni-surviving formula-level differences -- the most of any benchmark
in this project -- while the aggregate paired median-range test is not
significant (p=0.997); the two results answer different questions, not a
contradiction, since most of the corpus shows little seed-to-seed variation
while a handful of formulas show a lot. Layout and artifact identity:
`evidence/flatworld/README.md`. Training protocol (historical record of an
already-closed campaign, not an active runbook):
`benchmarks/flatworld/PPO_PROTOCOL.md`.

The adapter lives in `src/tlrl_benchmarks/flatworld`. Like the other two
benchmarks, it is fully self-contained and shares no code with either
sibling adapter.

Corpus freeze is documented in `benchmarks/flatworld/CORPUS_FREEZE.md`.

## Shared discipline across all three benchmarks

- All three benchmarks use the same DeepLTL checkout (fetched at a pinned
  commit into `.runtime/deep-ltl` by `scripts/setup_runtime.py`) and depend on
  the same two upstream compatibility patches (direct Rabinizer jar invocation,
  Windows file locking) -- `benchmarks/deep_ltl_runtime/`, not filed under any
  one benchmark's own folder, since none of them owns it.
- Frozen configs are content-hash-anchored; a training or campaign plan
  refuses to silently overwrite itself with a different configuration
  (`freeze_plan`'s "must match or raise" semantics in both
  `tlrl_provenance/campaign.py` and `scripts/train_letter_env_dqn.py`).
- No benchmark's adapter or confirmatory code imports from another
  benchmark's adapter; shared patterns are copied, not depended on.
- Every confirmatory campaign records `contamination_controls` (outcomes not
  inspected before training, best checkpoint not selected after outcomes)
  and a pilot/development stage precedes confirmatory training.
