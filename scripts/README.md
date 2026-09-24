# Scripts

All scripts are run from the repository root inside the environment described
in the top-level README. `TLRL_OUTPUT_DIR=<dir>` redirects the output of every
evidence-generating script so frozen evidence is never overwritten.

## Supported entry points

| Script | Purpose |
|---|---|
| `setup_runtime.py` | Check Java; download + verify Rabinizer 4; clone + patch DeepLTL; install checkpoints; inflate compressed corpora. `--check` verifies only. |
| `verify_artifacts.py` | SHA-256 verification of all evidence sidecars and all checkpoints. |
| `reproduce_analysis.py` | Re-run analyses/verifications on frozen data; byte-compare with frozen outputs. |
| `reproduce_audit.py` | Re-run a frozen audit campaign from the shipped checkpoints; compare episode by episode. |
| `run_confirmatory_semantic_audit.py` | Generic frozen-plan, resumable, sharded audit runner (see `docs/CONFIRMATORY_AUDITS.md`). |
| `train_letter_env_dqn.py` | LetterEnv Double-DQN trainer. |

## Analyses and verification (evidence generators)

`analyze_{deepltl_point,letter_env,flatworld}_significance.py`,
`analyze_deepltl_point_full_semantics.py`, `verify_semantic_kernel.py`,
`verify_prefix_exact_completeness.py`, `verify_coq_provenance_theorem.py`
(needs `coqc`), `verify_deepltl_rabinizer.py`,
`stress_test_{residual_size_guard,grounding_noise,stochastic_execution,extended_horizon,operator_generality}.py`,
`measure_deepltl_point_audit_overhead.py`.

## Environment smoke checks

`smoke_*_environment.py`, `smoke_*_official_ldba*.py`,
`smoke_letter_env_confirmatory_session.py`, `record_deepltl_training_smoke.py`.

## Historical records of the original campaigns

These produced the frozen protocols/manifests/launch records. They are kept for
provenance; they are guarded so they refuse to overwrite existing runs and are
not needed for reproduction: `freeze_*` (corpus freezes and pilot manifests),
`launch_*_ppo_*` (PPO campaign launchers),
`resume_deepltl_point_ppo_confirmatory_after_cuda_crash.py`,
`run_deepltl_point_*checkpoint_semantic_audit.py`,
`run_deepltl_point_preliminary_semantic_audit.py` (shared helpers),
`summarize_deepltl_point_checkpoint_semantics.py`,
`benchmark_letter_env_dqn_*.py` (DQN performance micro-benchmarks).
