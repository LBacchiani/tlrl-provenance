# LetterEnv formula/world corpus freeze

This stage freezes only the published DeepLTL `LetterEnv-v0` evaluation
corpus and its semantic grounding. It does not authorize or execute DQN
training and contains no learned-policy outcome.

The corpus consists of all 50 formulas in
`eval_datasets/LetterEnv-v0/tasks.txt`. Formula line `N` is paired with
`worlds/world_info_N.pkl`, following the upstream construction and evaluation
code. Each world is a 7x7 toroidal grid containing exactly two copies of every
letter `a` through `l`.

Two identities are recorded for every world:

- the SHA-256 of the official pickle bytes; and
- a canonical semantic SHA-256 over sorted `(row, column, letter)` triples.

The second digest detects meaningful layout changes without depending on
pickle serialization details. The freeze also records the pinned upstream
commit, the two existing native-Windows compatibility patches, and hashes of
the source files defining the environment, LDBA wrapper, horizon, dataset
construction, and formula/world pairing.

Regenerate the frozen artifacts with:

```bash
python scripts/freeze_letter_env_corpus.py
```

Generated artifacts:

- `src/tlrl_benchmarks/letter_env/corpus_manifest.json`
- `src/tlrl_benchmarks/letter_env/corpus_manifest.json.sha256`
- `evidence/letter_env/validation/letter_env_corpus_freeze.json`
- `evidence/letter_env/validation/letter_env_corpus_freeze.json.sha256`

The packaged and evidence copies must be byte-identical. Any formula, world,
pairing, or relevant source drift makes regeneration fail closed.
