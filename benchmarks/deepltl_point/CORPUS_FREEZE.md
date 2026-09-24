# DeepLTL PointWorld formula corpus freeze

This stage freezes only the published DeepLTL `PointLtl2-v0` evaluation
corpus and its semantic grounding. It does not authorize or execute PPO
training and contains no learned-policy outcome.

The corpus consists of all 50 formulas in
`eval_datasets/PointLtl2-v0/tasks.txt`: 20 ordered-eventuality formulas and
30 avoidance/until formulas over four grounded zone propositions (`blue`,
`green`, `magenta`, `yellow`).

Unlike `letter_env`, there is **no fixed paired world per formula**.
`PointLtl2-v0`'s zone geometry (two zones per color, placed by the
environment) is randomized per episode reset -- the same formula is paired
with a different scenario on every reset unless a `scenario_seed` is
explicitly shared across subjects (see `tlrl_provenance/campaign.py`'s
`scenario_namespace`, and the "Requirement-specific policy differences"
methodology in the confirmatory semantic report). This is a real semantic
difference from `letter_env`'s fixed paired `world_info_N.pkl` corpus, not
an oversight -- freezing this corpus establishes the formula set and
grounding rule, not a fixed scenario pool, because none exists to freeze.

The grounding rule is recorded as: a color proposition is true iff the
Euclidean agent-to-zone-center distance is at most the zone radius (`0.4`),
independently checked against the official environment's own
`info["propositions"]` at every retained step (mandatory agreement, not a
substitute for the independent computation).

What the freeze hashes:

- the formula corpus itself (`formula_corpus.normalized_lf_sha256`);
- the adapter source that defines grounding, label timing, and
  formula/task loading (`grounding.adapter_sha256`);
- the pinned upstream DeepLTL commit;
- the two runtime compatibility patches, shared with `letter_env` (see
  `benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md`);
- the real-environment and official-LDBA-wrapper smoke evidence
  (`evidence/deepltl_point/validation/`).

Regenerate the frozen artifact with:

```bash
python scripts/freeze_deepltl_point_manifest.py
```

Generated artifacts:

- `src/tlrl_benchmarks/deepltl_point/manifest.json`
- `src/tlrl_benchmarks/deepltl_point/manifest.json.sha256`

Unlike `letter_env`'s corpus freeze, there is no separate
`evidence/.../validation/*_corpus_freeze.json` copy -- the formula-corpus
hash is folded directly into this one manifest, alongside the runtime
compatibility and smoke-test evidence it also records. Any formula,
grounding-rule, source, or patch drift makes regeneration fail closed.

This manifest establishes runtime readiness and corpus identity only; it
predates and does not track confirmatory-campaign status. Confirmatory
completion is recorded separately -- see `evidence/deepltl_point/README.md`
and `docs/BENCHMARKS.md`.
