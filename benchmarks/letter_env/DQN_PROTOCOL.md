# LetterEnv DQN training protocol

This benchmark uses DeepLTL's published `LetterEnv-v0` environment, published
training curriculum, and frozen 50-formula evaluation corpus.  The learner is
new v3 code: a DQN/Double-DQN value learner using the same LetterEnv
convolutional observation encoder and LTL sequence encoder style used by
DeepLTL's PPO model.

**Status: confirmatory-complete and frozen.** The pilot gate was satisfied,
five confirmatory seeds trained to 5,000,000 steps, and the subsequent
5,000-episode semantic audit completed with zero missing records, duplicates,
or official/independent verdict mismatches.

LDBA/Rabinizer translation and training-log file locking depend on the same
runtime compatibility patches as the `deepltl_point` benchmark -- see
`benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md`. They are not
specific to either benchmark and are not duplicated here.

The protocol is frozen by:

```bash
python scripts/train_letter_env_dqn.py --name letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized
```

The final confirmatory subjects use trainer name
`letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized`
(see "Performance" and "Curriculum advancement" below for what that name
means); the command above writes:

- `evidence/letter_env/dqn/development/training/letter_env_dqn_training_plan_letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized.json`
- `evidence/letter_env/dqn/development/training/letter_env_dqn_training_plan_letter_env_dqn_v2_cached_curriculum_recency_fix_evidence_harmonized.json.sha256`

An earlier, immutable `letter_env_dqn_v1` plan
(`letter_env_dqn_training_plan.json`) also exists in the same directory --
that name is preserved as-is and never overwritten
(`training_plan_path` keeps it on a fixed legacy filename), it is not the
plan any current launch command targets. The intermediate
`letter_env_dqn_v2_cached` plan (pre-curriculum-fix) has been deleted as
stale -- its own training output was discarded and superseded, and unlike
`v1` it was never designated a permanent historical record.

It does not train unless `--execute` or `--smoke-steps` is passed.

## Performance

The original implementation was measured at ~35-39 sps steady-state (5M
steps -> ~39 hours per seed) -- unacceptable at 5-seed confirmatory scale.
Profiling (real, wall-clock-timed, not estimated) found ~88% of per-update
time was going into DeepLTL's own `preprocessing.preprocess_obss`, most of
that traced to a single upstream inefficiency: `BatchedSequences.from_seqs`
(`.runtime/deep-ltl/src/preprocessing/batched_sequences.py`) moves its
padded token tensor to the GPU *before* the Python loop that fills it in,
so every element write is a separate GPU round trip instead of one CPU-side
loop followed by one bulk transfer.

**Adopted fix**: `replay_preprocessing="cached"` (the default;
`DQNConfig.replay_preprocessing`, values `"cached"` / `"upstream"`).
Encodes each observation into a compact `CachedObservation` once, at
collection time, instead of re-tokenizing raw dicts on every replay sample;
`collate_cached_observations` builds the padded batch on CPU before one
device transfer, same fix as above but self-contained in
`tlrl_benchmarks/letter_env/dqn.py` rather than patching the vendored
checkout. Verified, not assumed, via `scripts/benchmark_letter_env_dqn_preprocessing.py`:

- correctness: model outputs bit-identical (`torch.allclose(..., rtol=0.0, atol=0.0)`) between `cached` and `upstream` given the same inputs;
- a real, matched-seed, same-step end-to-end training comparison: **2.56x** wall-clock speedup (45.5 -> 116.3 steps/sec at the settings that benchmark script uses), with identical recent-episode statistics between the two modes;
- `final_model_bitwise_equal` is `false` after hundreds of real gradient updates, `max_abs_difference` ~0.03 -- expected and not attributable to this fix: CUDA GRU training is not forced deterministic on this stack (no `torch.use_deterministic_algorithms`), so two runs of even the *unmodified* upstream code would also diverge this way. Inputs and pre-update outputs are proven bit-identical; only the itself-already-nondeterministic backward/optimizer path can differ downstream.

At the pilot's real settings, the caching fix measured roughly 90-116 sps
depending on run, down from the original projected ~39 hours per seed. This
range documents observed engineering performance rather than a portable
throughput guarantee; the complete confirmatory campaign is the evidence that
the final configuration was operationally feasible.

**Investigated and rejected**: merging the online network's two per-update
calls (`model(obs_batch)` for `q_selected`, `model(next_obs_batch)` for
Double DQN's action selection) into one forward pass over the concatenated
batch. Real, measured **1.38x** additional speedup, but failed strict
correctness verification: `q_selected` differed by `max_abs_diff=1.49e-08`
between the merged and unmerged paths (all other outputs matched exactly).
Root cause understood, not just observed: GPU matmul/GRU kernels can select
different internal algorithms or reduction orders depending on batch size,
so the same 128 rows aren't guaranteed bit-identical whether processed
alone or as a slice of a larger batch, even with no BatchNorm/Dropout
anywhere in this model (verified: none exist in `set_network.py`,
`conv_env_net.py`, `standard_env_net.py`, `torch_utils.py`, or the GRU path
of `ltl_net.py`). The magnitude is smaller than the already-accepted CUDA
non-determinism above, but "smaller than an accepted noise source" was
judged not the same bar as "bulletproof," so this was reverted in full --
**do not re-attempt this exact optimization without a variant that
actually clears bit-identical verification** (e.g. a fixed, padded-to-one-
size combined batch, untested).

Thread-count tuning (`num_threads`) was investigated *before* the caching
fix existed, under the old preprocessing-bottlenecked profile, and found a
solo pilot benefited from more threads than the 5-concurrent-safe default.
That finding was not used to tune the frozen confirmatory subjects after the
bottleneck moved. The final protocol conservatively fixed eight threads for
all subjects; no throughput-optimality claim is made.

## Curriculum advancement (stage-1 stall, diagnosed and fixed)

The first cached-preprocessing pilot (trainer name `letter_env_dqn_v2_cached`,
seed `930000`) never advanced past `curriculum_stage 1` for its entire
observed run, despite a pooled, recent-window success rate holding at
95-98% -- a real, initially confusing gap that was investigated rather than
assumed benign.

**Root cause, understood not just observed.** `LETTER_CURRICULUM`'s gate
(`.runtime/deep-ltl/src/sequence/samplers/curriculum.py`,
`Curriculum.update_task_success`) advances a stage only when the *mean of
each individual task variant's own success average* clears the stage
threshold (0.95) -- not the pooled recent-episode rate. Stage 1 is a
`RandomCurriculumStage` sampling uniformly from a combinatorial space of
roughly 4,500 distinct `(reach, avoid)` letter-pair variants (12
propositions, `nr` in {1,2}, `na` in {0,1,2}). The project's own
`EpisodeTracker` (`scripts/train_letter_env_dqn.py`) fed that gate a
per-task-variant average accumulated from *every distinct variant ever
sampled since the stage started*, with no expiry: a variant sampled once,
early, while the policy was still weak, kept contributing that stale,
low value to the aggregate mean indefinitely, since most of the ~4,500
variants are resampled too rarely to ever refresh their recorded average.
A law-of-large-numbers check (uniform noise averaged over ~4,500
independent entries should converge tightly to the true rate, not sit
persistently below threshold) indicates this staleness, not a genuine
inability to solve stage-1 tasks, is the dominant explanation. A secondary,
compounding factor: DQN's permanent `final_epsilon=0.05` exploration floor
injects action-level noise into every training-time rollout that PPO's
on-policy behavior policy does not have, disproportionately affecting
harder task variants (less step-budget slack per stray random action).

**Independent corroboration.** A preliminary (non-confirmatory, exploratory)
audit of a real, substantially-trained checkpoint from that same pilot
(3,750,000 of 5,000,000 steps) against the full frozen 50-formula
evaluation corpus -- 50 formulas x 5 episodes, seed 930000 -- found 73.2%
success against the corpus, versus the 95-98% pooled training-time rate on
stage 1's own (narrower, depth-1-only) task distribution. 41 of 50 formulas
were "mixed" (same formula, sometimes solved in under 20 steps, sometimes
censored, depending only on the random map drawn), and zero formulas were
censored in all 5 of their episodes -- consistent with real but
inconsistent transfer to task depths the curriculum had never made the
direct training target, not with an unsolvable subset. This scan is
preliminary evidence for the diagnosis above, not confirmatory data, and is
not treated as such.

**Fix applied.** `EpisodeTracker.curriculum_success()`
(`scripts/train_letter_env_dqn.py`) now excludes any task variant not
sampled within the last `window` (500) episodes from the aggregate passed
to the gate, reusing the same `window` constant already used to bound
per-task history -- a stale variant's outdated average no longer
contributes once it falls out of recent circulation. Verified: a
standalone behavioral test (a goal touched once then dropped for >window
episodes is excluded; re-touching it reinstates it) and the full existing
suite (42/42 `letter_env`-scoped tests) pass unchanged. This is a pure
measurement/bookkeeping correction -- it does not touch reward, the success
definition, exploration, or the learning update in any way.

**Action taken.** The pre-fix pilot's output
(`experiments/dqn/LetterEnv-v0/letter_env_dqn_v2_cached/930000/`) was
deleted (gitignored, superseded, not confirmatory evidence) and the same
pilot seed (`930000`) relaunched under a new trainer name,
`letter_env_dqn_v2_cached_curriculum_recency_fix` -- versioned rather than
overwritten because `build_plan`'s frozen-plan hash covers the training
script's own source, and this fix is a genuine source change the freeze
mechanism correctly refused to let through silently under the old name.
Config is otherwise identical (5,000,000 steps, same architecture,
hyperparameters, curriculum, seed). The later evidence-harmonized version is
the final frozen training configuration. This historical diagnosis is not
used as a paper claim about curriculum-stage advancement; the confirmatory
semantic campaign evaluates the resulting fixed checkpoints directly.

## Pilot-then-confirmatory gate

Mirrors the `deepltl_point` PPO precedent: a dedicated pilot seed, disjoint
from the confirmatory seed set, is trained and its outcomes inspected before
any confirmatory seed is allowed to train. This is enforced, not just
documented -- `--execute`/`--execute-all` on a confirmatory seed raises
`RuntimeError` until the pilot has been launched and explicitly authorized.

1. Train the pilot seed (`930000`, disjoint from the five confirmatory
   seeds below). This is never gated:

   ```bash
   python scripts/train_letter_env_dqn.py --execute --seed 930000 --device cuda --num-threads 8
   ```

2. Inspect the pilot's `training_log.csv` (learning signal present, no
   crashes, curriculum progressing sensibly). Do not inspect it against the
   50-formula confirmatory evaluation corpus -- that corpus stays untouched
   until the confirmatory campaign itself runs.

3. Authorize the confirmatory seeds (refuses if step 1 hasn't happened):

   ```bash
   python scripts/train_letter_env_dqn.py --authorize-confirmatory-training
   ```

4. Only now will the confirmatory seeds run:

   ```bash
   python scripts/train_letter_env_dqn.py --execute --seed 930001 --device cuda --num-threads 8
   ```

   or all five sequentially:

   ```bash
   python scripts/train_letter_env_dqn.py --execute-all --device cuda --num-threads 8
   ```

`--num-threads 8` was used uniformly above as the conservative frozen setting.

Gate state lives in
`evidence/letter_env/dqn/development/training/letter_env_dqn_v2_cached_pilot_gate.json`
(+ `.sha256`; the filename is fixed by `gate_path()` and is not re-versioned
alongside the training plan), with a full transition history
(`pilot_not_launched` -> `pilot_launched` -> `confirmatory_authorized`).
Unlike the training plan, this file is not freeze-once-and-match -- its
whole purpose is to record that transition over time.

Default subject plan:

- algorithm: Double DQN;
- pilot seed: `930000` (inspected before any confirmatory training);
- confirmatory seeds: `930001` through `930005`;
- budget: 5,000,000 environment steps per seed;
- environment: official DeepLTL `LetterEnv-v0`;
- horizon: 75 steps;
- reward: official DeepLTL `SequenceWrapper` reach-avoid reward;
- curriculum: official `LETTER_CURRICULUM`;
- primary checkpoint rule: final checkpoint, not best checkpoint after
  semantic outcomes.
- device/thread rule: GPU execution with `--device cuda --num-threads 8`
  (see "Performance" for why this isn't further tuned yet). Whatever value
  is actually in effect is recorded in the frozen plan, which is the source
  of truth, not this document.

The DQN benchmark is meant to add learner diversity relative to the first
PointWorld/PPO benchmark.  It is not a claim that DQN is causally better or
worse than PPO; the paper claim remains that aggregate `P_sat` hides semantic
structure in both successes and non-successes.

## Confirmatory semantic audit

The five frozen DQN checkpoints were evaluated on all 50 formulas with 20
scenario-paired episodes per policy/formula. The resulting 5,000 records are
complete and contain zero official/independent mismatches. Overall `P_sat`
was 0.769 (3,843/5,000), comprising 3,843 decided satisfactions, 1,152
censored episodes, and 5 decided violations. The paired cross-seed analysis
found 3/50 uncorrected formula-level tests and no Bonferroni survivors
(aggregate permutation p=0.353). Canonical artifacts are under
`evidence/letter_env/dqn/confirmatory/semantic/`.
