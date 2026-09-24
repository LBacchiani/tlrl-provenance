# Trained checkpoints

Every learned controller audited in the paper is shipped here, together with
its training log, so that all results can be reproduced without retraining.
`CHECKSUMS.sha256` lists the SHA-256 of every file below; verify with

```bash
python scripts/verify_artifacts.py
```

## Layout

```
experiments/
  ppo/<Environment>/<experiment>/<seed>/     DeepLTL formula-conditioned recurrent PPO
  dqn/LetterEnv-v0/<experiment>/<seed>/      Double-DQN value learner (this repository's trainer)
  CHECKSUMS.sha256
```

Each `<seed>/` directory holds

| File | Content |
|---|---|
| `eval/<steps>.pth` | model snapshots taken during training (the audited checkpoint is one of these) |
| `status.pth` | final full training state (model + optimizer + counters) |
| `ltl_net.pth`, `vocab.pkl` | PPO only: pretrained LTL-formula encoder and its vocabulary |
| `log.csv` / `training_log.csv` | per-update training log |
| `launch.stdout.log`, `launch.stderr.log` | launcher output (PPO) |

PPO experiment directories additionally hold `experiment_config.json`, the
exact configuration DeepLTL's trainer recorded.

## Confirmatory policies (the ones audited in the paper)

| Benchmark | Algorithm | Experiment | Seeds | Audited snapshot |
|---|---|---|---|---|
| PointWorld (`PointLtl2-v0`) | PPO | `v3_confirm_ppo_5m_seed_<seed>` | 920001-920005 | `eval/4980736.pth` |
| LetterEnv (`LetterEnv-v0`) | DQN | `letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized` | 930001-930005 | `eval/5000000.pth` |
| LetterEnv (`LetterEnv-v0`) | PPO | `v3_confirm_ppo_5m_letter_env_seed_<seed>` | 940001-940005 | `eval/4816896.pth` |
| FlatWorld (`FlatWorld-v0`) | PPO | `v3_confirm_ppo_5m_flatworld_seed_<seed>` | 960001-960005 | `eval/4980736.pth` |

## Development-pilot policies

One reserved development seed per pipeline (`910001` PointWorld PPO, `930000`
LetterEnv DQN, `940000` LetterEnv PPO, `960000` FlatWorld PPO) was trained before
the confirmatory seeds to establish runtime readiness. They contribute no result
to the paper's confirmatory claims and are shipped only because retained
development evidence refers to them.

### Two DQN experiment folders

`experiments/dqn/LetterEnv-v0/` has two folders because the DQN trainer's frozen plan hashes
the training script's own source, so a script change requires a new experiment name:

* `letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized/` - the final
  configuration: the five confirmatory seeds 930001-930005 audited in the paper.
* `letter_env_dqn_v2_cached_curriculum_recency_fix/` - only the development pilot seed 930000,
  trained under the preceding script version (a curriculum-stage bookkeeping fix that does not
  change reward, exploration or the learning update; see `benchmarks/letter_env/DQN_PROTOCOL.md`).

## Where PPO checkpoints must live to be used

DeepLTL's code and this repository's audit adapters look for PPO checkpoints in
`.runtime/deep-ltl/experiments/ppo/`. `python scripts/setup_runtime.py` copies
`experiments/ppo/` there (DQN checkpoints are read from `experiments/dqn/`
directly). PPO seed `920003` had one transient CUDA-driver failure; its failed attempt is
kept as `920003.failed-pretraining-nvcuda-*` (see
`evidence/deepltl_point/confirmatory/training/`).
