# FlatWorld PPO training protocol

This benchmark uses DeepLTL's published `FlatWorld-v0` environment, published
PPO hyperparameters (from the upstream repository's own `run_flatworld.py`
convenience script -- not this project's generic PPO defaults and not
LetterEnv's or PointLtl2's hyperparameters, which differ substantially), and
the frozen 49-formula evaluation corpus. The learner is the official DeepLTL formula-conditioned recurrent
PPO controller, unmodified.

This follows exactly the same path as `letter_env`'s and `deepltl_point`'s
PPO protocols -- same algorithm, same upstream trainer, same
freeze-manifest-then-launch discipline, only the environment differs.
Runtime compatibility (Rabinizer under Java 25, Windows file locking, Windows
console UTF-8 output) is shared across all three benchmarks and documented in
`benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md` plus the
"Windows console encoding and output buffering" section of
`benchmarks/letter_env/PPO_PROTOCOL.md`, not duplicated here -- the pilot
manifest below already bakes `PYTHONUTF8=1`/`PYTHONIOENCODING=utf-8`/
`PYTHONUNBUFFERED=1` into its `required_environment`. Formula corpus freeze:
`benchmarks/flatworld/CORPUS_FREEZE.md`.

**Status: confirmatory-complete.** The pilot (seed `960000`) was trained and
inspected first; seeds `960001`-`960005` then completed the full
5,000,000-step confirmatory campaign and were audited against all 49
formulas (4,900 episodes), zero official/independent verdict mismatches.
Frozen evidence lives under `evidence/flatworld/ppo/confirmatory/semantic/`;
see `evidence/flatworld/README.md` for the artifact layout and
the paper's Evaluation section (\S FlatWorld) for the interpreted
result. FlatWorld is evaluated with this PPO pillar only -- algorithmic
diversity (policy-gradient versus value-based learning) is covered by
LetterEnv, not duplicated here.

## 1. Development pilot

A dedicated pilot seed, disjoint from every other seed range in this project
(`deepltl_point` PPO: `910001`/`920001`-`920005`; `letter_env` value-based:
`930000`/`930001`-`930005`; `letter_env` PPO: `940000`/`940001`-`940005`),
trained and inspected before the confirmatory seeds below were launched:

```bash
python scripts/freeze_flatworld_ppo_pilot_manifest.py
python scripts/launch_flatworld_ppo_pilot.py
```

- pilot seed: `960000` (reserved; confirmatory seeds, if this pillar
  proceeds to a confirmatory campaign, would be `960001`-`960005`);
- `16` parallel environments, `4096` steps/process per update (DeepLTL's own
  published default for this environment, same as PointLtl2's -- much larger
  than LetterEnv's `128`, since FlatWorld's 500-step horizon needs more
  per-update signal), `10` epochs, discount `0.98`;
- target `5,000,000` steps -- a deliberate, documented reduced budget:
  DeepLTL's own `run_flatworld.py` default is `15,000,000`; `5,000,000`
  matches the same deviation already made, for the same reason, by this
  project's `deepltl_point` and `letter_env` PPO campaigns;
- pinned to GPU1 (`CUDA_VISIBLE_DEVICES=1`), mirroring `letter_env`'s split;
- manifest freeze already run once (2026-09-09) against this pillar's own
  prelaunch evidence (`flatworld_corpus_freeze.json`,
  `flatworld_environment_smoke.json`, `flatworld_official_ldba_env_smoke.json`
  -- all under `evidence/flatworld/validation/`, all hash-verified before the
  manifest was written); manifest + (once launched) launch record live at
  `evidence/flatworld/ppo/development/training/flatworld_ppo_5m_development_pilot_{manifest,launch}.json`.

Like `deepltl_point`'s and `letter_env`'s PPO protocols and unlike this
project's value-based protocol, this gate is **not** code-enforced -- there is no
confirmatory launcher guard, and the confirmatory campaign below should not
be launched until the pilot's outcomes have been reviewed.

## 2. Confirmatory campaign

Launched after the pilot's outcomes were inspected, using:

```bash
python scripts/launch_flatworld_ppo_confirmatory_campaign.py
```

Mirrors `letter_env`'s and `deepltl_point`'s confirmatory campaign launchers
exactly (frozen protocol read first, one campaign manifest plus one
immutable per-seed manifest written before any process starts,
self-supervising with crash detection), with the same two deliberate
differences as `letter_env`'s, both recorded in the frozen protocol itself:

- Single GPU (`CUDA_VISIBLE_DEVICES=1`) for every seed, not a two-GPU cycle.
- `--max-concurrent` defaults to `1` (fully sequential), not `2`. Running
  multiple PPO seeds concurrently on one GPU is exactly the failure mode
  that caused the `letter_env` value-based pilot's multi-hour throughput collapse
  earlier in this project -- avoided deliberately here, not by accident.

Frozen protocol:
`evidence/flatworld/ppo/confirmatory/protocol/flatworld_ppo_paper_benchmark_protocol.json`
(+ `.sha256`), seeds `960001`-`960005`, same hyperparameters as the pilot,
same 5,000,000-step budget (batch-boundary-rounded expected final step count:
5,046,272, identical arithmetic to `deepltl_point`'s PPO campaign since both
use the same `16 x 4096` batch size). The protocol additionally records a
FlatWorld-specific interpretation guardrail not needed by either sibling
benchmark: unlike LetterEnv (zero-or-one active proposition) and PointLtl2
(disjoint-by-construction zones), FlatWorld permits more than one proposition
to be simultaneously true, so raw atom-contact counts are not directly
comparable across benchmarks without accounting for that.

This PPO pillar's `scenario_namespace` is frozen before confirmatory
evaluation. The campaign engine's `scenario_seed` derivation
(`tlrl_provenance/campaign.py`) has no subject term, so every PPO subject is
paired on the same scenario within each formula and episode index.

## 3. Semantic audit

Runs through the same benchmark-independent campaign runner used by
every other benchmark in this project (`tlrl_provenance`), evaluated against
the frozen 49-formula corpus -- not a new corpus and not PPO-specific
evaluation code. PPO training itself needed no new adapter
code (`run_flatworld.py`/`train_ppo.py` already train `FlatWorld-v0`
directly); auditing PPO checkpoints uses the dedicated
`FlatworldPPOConfirmatoryAdapter` session.

## Default subject plan

- algorithm: official DeepLTL formula-conditioned recurrent PPO;
- pilot seed: `960000` (development only, not a confirmatory subject);
- confirmatory seeds: `960001`-`960005`, all complete;
- budget: 5,000,000 environment steps (16 environments x 4096 steps/process
  per update);
- environment: official DeepLTL `FlatWorld-v0`;
- primary checkpoint rule: final checkpoint, not best checkpoint after
  semantic outcomes.

The PPO pillar evaluates the official DeepLTL controller using the frozen
FlatWorld environment, corpus, and semantic audit. See
`evidence/flatworld/README.md` for the frozen artifact layout.
