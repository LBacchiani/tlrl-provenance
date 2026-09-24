# LetterEnv PPO training protocol

This benchmark uses DeepLTL's published `LetterEnv-v0` environment, published
PPO hyperparameters (from the upstream repository's own `run_letter.py`
convenience script -- not this project's generic PPO defaults and not
PointLtl2's hyperparameters, which differ substantially), and the same
frozen 50-formula evaluation corpus used by the `letter_env` DQN benchmark.
The learner is the official DeepLTL formula-conditioned recurrent PPO
controller, unmodified.

This follows exactly the same path as `deepltl_point`'s PPO protocol
(`benchmarks/deepltl_point/PPO_PROTOCOL.md`) -- same algorithm, same
upstream trainer, same freeze-manifest-then-launch discipline, only the
environment differs. Runtime compatibility (Rabinizer under Java 25,
Windows file locking, and now Windows console UTF-8 output -- see
"Windows console encoding" below) is shared with `deepltl_point` and
documented in `benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md`,
not duplicated here. Formula corpus freeze:
`evidence/letter_env/validation/letter_env_corpus_freeze.json` (the same
corpus the DQN benchmark uses).

**Status: confirmatory-complete and frozen.** The pilot was inspected before
the five confirmatory seeds trained. The 5,000-episode semantic campaign is
complete with zero missing records, duplicates, or official/independent
verdict mismatches.

## Windows console encoding and output buffering

Two Windows-specific issues were found and fixed before the pilot launch
that is actually running (both are in the frozen manifest's
`required_environment`, not applied by hand at launch time):

1. DeepLTL's text logger prints Greek `μ` in several column labels (`rμ`,
   `sμ`, `Pμ`, `Vμ`) on every logged row. Windows' default console codepage
   (cp1252) cannot encode it, crashing the very first log line with
   `UnicodeEncodeError`. Fixed by forcing UTF-8 I/O: `PYTHONIOENCODING=utf-8`
   (alongside the pre-existing `PYTHONUTF8=1`).
2. The text logger uses plain `print()` with no explicit flush (verified:
   `utils/logging/text_logger.py` has no `flush=True` anywhere). Redirected
   to a file rather than a terminal, Python defaults to full block
   buffering, so `stdout` -- including "Curriculum stage: N" and "Stage N
   completed." -- sat unflushed and invisible for the entire first pilot
   attempt despite training running correctly underneath. Fixed with
   `PYTHONUNBUFFERED=1`. Caught by trying to read curriculum-stage progress
   from a live run and finding the stdout log at 0 bytes despite an hour of
   real training; the pilot was killed and relaunched a second time once
   this was found and fixed, rather than left running blind.

Neither was encountered for `deepltl_point` previously (cause not
investigated -- possibly a different code path or console state at the
time); both fixes are unconditionally safe to carry forward regardless.

## 1. Development pilot

A dedicated pilot seed, disjoint from every other seed range in this
project (`deepltl_point` PPO: `910001`/`920001`-`920005`; `letter_env` DQN:
`930000`/`930001`-`930005`), trained and inspected before any confirmatory
seed is launched:

```bash
python scripts/freeze_letter_env_ppo_pilot_manifest.py
python scripts/launch_letter_env_ppo_pilot.py
```

- pilot seed: `940000` (development only); confirmatory seeds are
  `940001`-`940005`;
- `16` parallel environments, `128` steps/process per update (DeepLTL's own
  published default for this environment -- much smaller than PointLtl2's
  4096, since LetterEnv episodes are short), `8` epochs, discount `0.94`;
- target `5,000,000` steps -- a deliberate, documented reduced budget:
  DeepLTL's own `run_letter.py` default is `15,000,000`; `5,000,000` matches
  the same deviation already made, for the same reason, by this project's
  `deepltl_point` PPO campaign (whose own upstream default via `run_zones.py`
  is also `15,000,000`), and is sized to match the `letter_env` DQN pilot's
  budget for comparability;
- pinned to GPU1 (`CUDA_VISIBLE_DEVICES=1`) so it runs concurrently with the
  DQN pilot on GPU0 without contention;
- manifest + launch record:
  `evidence/letter_env/ppo/development/training/letter_env_ppo_5m_development_pilot_{manifest,launch}.json`.

Like `deepltl_point`'s PPO protocol and unlike `letter_env`'s DQN protocol,
this gate was enforced procedurally rather than in code. The pilot outcomes
were reviewed before the confirmatory launcher was used.

## 2. Confirmatory campaign

The verified launcher used for the completed campaign is:

```bash
python scripts/launch_letter_env_ppo_confirmatory_campaign.py
```

Mirrors `deepltl_point`'s confirmatory campaign launcher exactly (frozen
protocol read first, one campaign manifest plus one immutable per-seed
manifest written before any process starts, self-supervising with crash
detection), with two deliberate differences, both recorded in the frozen
protocol itself:

- Single GPU (`CUDA_VISIBLE_DEVICES=1`) for every seed, not a two-GPU cycle
  -- GPU0 is dedicated to the `letter_env` DQN pillar throughout this
  project.
- `--max-concurrent` defaults to `1` (fully sequential), not `2`. Running
  multiple PPO seeds concurrently on one GPU is exactly the failure mode
  that caused the DQN pilot's multi-hour throughput collapse earlier in
  this project -- avoided deliberately here, not by accident.

Frozen protocol:
`evidence/letter_env/ppo/confirmatory/protocol/letter_env_ppo_paper_benchmark_protocol.json`
(+ `.sha256`), seeds `940001`-`940005`, same hyperparameters as the pilot,
same 5,000,000-step budget. Verified before being trusted: protocol hash
integrity, seed-namespace disjointness (pilot seed, these five, and the
entire `letter_env` DQN pillar's `930000`-`930005` range all mutually
exclusive), and the exact built command compared field-by-field against
the already-verified pilot command before any real confirmatory training was
launched.

## 3. Semantic audit

The audit ran through the same benchmark-independent campaign runner used by
the other benchmarks (`tlrl_provenance`) against the shared frozen 50-formula
corpus. Five policies x 50 formulas x 20 episodes produced 5,000 complete
records. Overall `P_sat` was 0.770 (3,852/5,000), comprising 3,852 decided
satisfactions, 1,039 censored episodes, and 109 decided violations. The
paired cross-seed analysis found 3/50 uncorrected formula-level tests and no
Bonferroni survivors (aggregate permutation p=0.914). Canonical artifacts
are under `evidence/letter_env/ppo/confirmatory/semantic/`.

## Default subject plan

- algorithm: official DeepLTL formula-conditioned recurrent PPO;
- pilot seed: `940000` (development only, not a confirmatory subject);
- confirmatory seeds: `940001`-`940005`;
- budget: 5,000,000 environment steps (16 environments x 128 steps/process
  per update);
- environment: official DeepLTL `LetterEnv-v0`;
- primary checkpoint rule: final checkpoint, not best checkpoint after
  semantic outcomes.

The PPO pillar supplies a second mainstream learner on the same environment
and formula corpus. Its semantic campaign uses a separate scenario namespace
from DQN, so the cross-algorithm comparison is descriptive and is not a claim
that PPO is causally better or worse than DQN on this environment.
