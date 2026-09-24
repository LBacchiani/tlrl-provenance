# DeepLTL PointWorld PPO training protocol

This benchmark uses DeepLTL's published `PointLtl2-v0` environment, published
recurrent-PPO hyperparameters, and frozen 50-formula evaluation corpus. The
learner is the official DeepLTL formula-conditioned recurrent PPO
controller, unmodified.

Runtime compatibility (Rabinizer under Java 25, Windows file locking) is
shared with the `letter_env` benchmark and documented separately -- see
`benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md`, not
duplicated here. Formula corpus freeze:
`benchmarks/deepltl_point/CORPUS_FREEZE.md`.

**Status: confirmatory-complete.** This is a historical record of how the
already-closed campaign was run, not an active runbook -- no further PPO
training is planned for this benchmark. See `docs/BENCHMARKS.md` and
`evidence/deepltl_point/confirmatory/semantic/deepltl_point_semantic_report.md`
for the results.

## 1. Development pilot

A dedicated pilot seed, disjoint from the confirmatory seed set, trained and
was inspected before any confirmatory seed was launched:

```bash
python scripts/freeze_deepltl_point_ppo_pilot_manifest.py
python scripts/launch_deepltl_point_ppo_pilot.py
```

- pilot seed: `910001` (reserved, disjoint from the five confirmatory seeds
  below);
- `16` parallel environments, `4096` steps/process per update, target
  `5,000,000` steps;
- manifest + launch record:
  `evidence/deepltl_point/development/training/deepltl_point_ppo_5m_development_pilot_{manifest,launch}.json`.

Unlike `letter_env`'s DQN protocol, this gate was **not** code-enforced --
the confirmatory launcher below does not itself check whether the pilot was
inspected. The pilot/confirmatory separation was maintained by not running
the confirmatory launch script until the pilot's outcomes had been reviewed,
recorded via `contamination_controls.confirmatory_training_authorized` in
the pilot manifest (frozen `false`; never programmatically flipped).

## 2. Confirmatory campaign

```bash
python scripts/launch_deepltl_point_ppo_confirmatory_campaign.py --max-concurrent 2 --gpu-cycle "0,1"
```

Reads the already-frozen protocol
(`evidence/deepltl_point/confirmatory/protocol/deepltl_point_paper_benchmark_protocol.json`)
and launches all five confirmatory seeds (`920001`-`920005`), recording one
campaign-level launch artifact plus one immutable per-seed manifest before
starting any process. `--max-concurrent` limits concurrent jobs since each
PPO run spawns multiple environment workers.

One seed (`920003`) hit a transient CUDA-driver crash mid-campaign;
`scripts/resume_deepltl_point_ppo_confirmatory_after_cuda_crash.py` is the
incident-specific recovery controller that was used -- it keeps the
original manifests and launch record immutable, supervised the still-running
seeds, and retried only the crashed one. It is a one-off recovery record, not
a general-purpose resume tool.

## 3. Semantic audit

The confirmatory semantic campaign (`tlrl_provenance` audit against all 5,000
episodes) is run through the benchmark-independent campaign runner
documented in `docs/CONFIRMATORY_AUDITS.md`, not a PointWorld-specific script.
Its frozen config, plan, records, and generated report live under
`evidence/deepltl_point/confirmatory/semantic/`; see
`evidence/deepltl_point/README.md` for the full layout.

## Default subject plan

- algorithm: official DeepLTL formula-conditioned recurrent PPO;
- pilot seed: `910001` (development only, not a confirmatory subject);
- confirmatory seeds: `920001` through `920005`;
- budget: 5,000,000 environment steps per seed (16 environments x 4,096
  steps/process per update);
- environment: official DeepLTL `PointLtl2-v0`;
- primary checkpoint rule: final checkpoint, not best checkpoint after
  semantic outcomes.
