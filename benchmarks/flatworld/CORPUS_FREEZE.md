# FlatWorld formula corpus and circle-geometry freeze

This stage freezes only the published DeepLTL `FlatWorld-v0` evaluation
corpus and its semantic grounding. It contains no learned-policy outcome.

The corpus consists of all 49 formulas in
`eval_datasets/FlatWorld-v0/tasks.txt`. Unlike `LetterEnv-v0` (a per-episode
randomized letter map, paired one-to-one with a `world_info_N.pkl` file per
formula) and `PointLtl2-v0` (randomized, disjoint-by-construction zones),
`FlatWorld-v0`'s nine colored circles are a **fixed class attribute**
(`FlatWorld.CIRCLES` in `envs/flatworld/flatworld.py`) baked directly into
the upstream source. Only the agent's starting position is randomized per
episode, as a pure function of the reset seed
(`FlatWorld.reset(seed=...)` reconstructs `self.rng = np.random.default_rng(seed)`
fresh on every call). There is therefore no per-formula world file to freeze
or pair -- what needs freezing is the formula corpus and the fixed geometry
itself.

Two colors (`red`, `green`) have two circles each; the other five colors have
one circle each -- nine circles, seven colors. Different-color circles are
allowed to overlap by upstream design (for example the `blue`, `green`, and
`aqua` circles share a common region), so **more than one proposition can be
simultaneously true** -- a real structural difference from both sibling
benchmarks that the adapter's grounding logic (`grounded_propositions`
returning a `frozenset` of arbitrary size, not zero-or-one) accounts for
directly.

The freeze script re-imports the live upstream `FlatWorld` class and compares
its `CIRCLES` attribute against this package's transcribed, hand-copied
constant, so any drift between the two -- a version bump, a typo in
transcription -- is caught at freeze time rather than trusted silently.

Regenerate the frozen artifacts with:

```bash
python scripts/freeze_flatworld_corpus.py
```

Generated artifacts:

- `src/tlrl_benchmarks/flatworld/corpus_manifest.json`
- `src/tlrl_benchmarks/flatworld/corpus_manifest.json.sha256`
- `evidence/flatworld/validation/flatworld_corpus_freeze.json`
- `evidence/flatworld/validation/flatworld_corpus_freeze.json.sha256`

The packaged and evidence copies must be byte-identical. Any formula,
geometry, or relevant source drift makes regeneration fail closed.

Already run once (2026-09-09): 49 formulas (23 Eventually-chains, 26
Until-chains), 9 circles, live-vs-frozen geometry match confirmed,
`manifest_sha256 = da2a6ad6def6f4b0513409a771962d8e7e1d36be8aafc85064a5b9616c8985c8`.
