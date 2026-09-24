# DeepLTL PointWorld paper-benchmark protocol

Status: frozen protocol for a paper-eligible benchmark campaign, not a
confirmatory result.

Frozen on: 2026-08-31

## Purpose

This benchmark tests whether semantic provenance adds useful information beyond
the ordinary TL-guided RL success rate, `P_sat`.

The benchmark is not used to invent formulas, tune formulas, or choose a nice
subset after seeing results. It uses the official DeepLTL PointWorld evaluation
formulas as the task corpus.

## Benchmark unit

- Environment: `PointLtl2-v0`.
- Formula corpus: all 50 official DeepLTL PointWorld evaluation formulas.
- Formula source: `src/tlrl_benchmarks/deepltl_point/tasks.txt`.
- Atoms: `blue`, `green`, `magenta`, `yellow`.
- Formula families are descriptive only:
  - `eventually_chain`: formulas whose normalized source starts with `F`.
  - `until_chain`: all remaining official formulas.

No formula may be dropped after inspecting semantic audit results.

## Subject algorithms

DeepLTL PointWorld contributes one clean TL-guided deep-RL subject algorithm:

- formula-conditioned recurrent PPO, using the upstream DeepLTL training stack.

The local upstream clone contains only `src/train/train_ppo.py`; no native
DeepLTL SAC or DQN trainer is present. Therefore SAC and DQN are not primary
DeepLTL subjects unless a separately frozen engineering extension is created
before any result inspection. Algorithmic diversity for the paper is handled at
the portfolio level, by using PPO/SAC/DQN where they are native and appropriate
in other benchmarks.

## Training design

Confirmatory DeepLTL training uses five independent training seeds:

- `920001`
- `920002`
- `920003`
- `920004`
- `920005`

The previous seed `910001` is development-only and cannot be promoted into the
primary confirmatory result because its checkpoint-sweep outcomes have already
been inspected.

Per seed:

- target training budget: 5,000,000 environment steps;
- expected batch-rounded budget: 5,046,272 environment steps under the current
  PPO configuration;
- device: GPU when available;
- number of parallel processes: 16;
- steps per process: 4096;
- batch size: 2048;
- learning rate: 0.0003;
- discount: 0.998;
- entropy coefficient: 0.003;
- epochs: 10;
- checkpoint save interval: every upstream save event.

The primary final policy for each seed is the latest evaluation checkpoint
produced by the upstream run. The best checkpoint is never selected as the
primary policy after seeing outcomes; best-checkpoint curves are secondary
learning-dynamics diagnostics only.

## Evaluation design

Two evaluation layers are frozen.

### Primary final-policy audit

For each training seed and each of the 50 formulas:

- run 20 deterministic evaluation episodes;
- use fixed evaluation seeds derived from the tuple
  `(training_seed, formula_index, episode_index)`;
- record official DeepLTL success/violation flags;
- independently evaluate the grounded LTL trace with the v3 semantic engine;
- compute semantic provenance quantities and failure classes.

Primary DeepLTL sample size:

```text
5 training seeds * 50 formulas * 20 episodes = 5000 final-policy episodes
```

### Secondary checkpoint-trajectory audit

For each training seed, evaluate the following checkpoints when present:

- initialization checkpoint;
- approximately 0.5M steps;
- approximately 1.0M steps;
- approximately 1.5M steps;
- approximately 2.0M steps;
- approximately 3.0M steps;
- approximately 4.0M steps;
- latest evaluation checkpoint.

For each selected checkpoint and formula:

- run 3 deterministic evaluation episodes;
- use fixed evaluation seeds derived from the tuple
  `(training_seed, checkpoint_rank, formula_index, episode_index)`.

Secondary trajectory sample size, if all checkpoints are present:

```text
5 training seeds * 8 checkpoints * 50 formulas * 3 episodes = 6000 trajectory episodes
```

This trajectory audit is secondary and descriptive. It is intended to show
coarse learning dynamics, detect non-monotonic formula reliability, and expose
large semantic shifts. It is not powered as a confirmatory test for small
within-formula correlations between training progress and evidence structure.
A small or null evidence-learning association should be interpreted as
consistent with the development pilot unless a predeclared effect-size analysis
states otherwise.

## Recorded quantities

For every evaluated episode, record:

- training seed;
- checkpoint path and checkpoint digest;
- checkpoint step;
- formula id;
- formula text;
- formula family;
- evaluation seed;
- episode return;
- episode length;
- termination/truncation flags;
- official success and violation flags;
- independent temporal verdict;
- official/independent mismatch flag;
- semantic atom counts;
- possible evidence count;
- necessary evidence count;
- possible-evidence-per-step;
- necessary-evidence-per-step;
- zero-necessary-success flag;
- failure class.

Failure classes:

- decided satisfaction;
- forbidden or temporal decided violation;
- censored timeout/truncation;
- environment termination without relevant progress;
- environment termination after relevant progress.

## Primary analyses

The benchmark supports three primary analyses.

1. Verdict agreement.

   Report whether the independent v3 semantic evaluator agrees with the
   official DeepLTL automaton outcome. Any mismatch is a machinery defect until
   proven otherwise.

2. Semantic enrichment of `P_sat`.

   For each formula and policy seed, report how satisfied episodes are
   supported:

   - how much possible evidence exists;
   - how much necessary evidence exists;
   - whether satisfaction has zero necessary evidence because witnesses are
     redundant;
   - whether the formula requires a long load-bearing avoidance prefix.

   These quantities explain the semantic shape of success; they are not raw
   reward, return, or success rate.

3. Failure diagnosis.

   For each failed episode, distinguish direct TL violation, censoring,
   no-progress termination, and relevant-progress-but-wrong-pattern
   termination. These classes imply different practitioner actions.

## Interpretation guardrails

Do not compare raw necessary-evidence averages across unrelated formula shapes
as if they were policy quality.

In particular:

- `F`-chains can have zero necessary evidence simply because multiple witness
  times satisfy the same eventuality;
- `U`-chains can have high necessary evidence because the formula structurally
  requires maintaining an avoidance prefix;
- policy-learning claims require holding the formula fixed and comparing
  checkpoints, seeds, or algorithms;
- formula-family summaries are descriptive, not causal claims about controller
  robustness.

The correct practitioner-facing interpretation is:

> `P_sat` says whether the formula was satisfied. Semantic provenance explains
> what kind of satisfaction or failure occurred.

## Eligibility checks

A DeepLTL campaign is paper-eligible only if all of the following hold:

1. all 50 official formulas are evaluated;
2. no formula is dropped after outcome inspection;
3. all five confirmatory seeds finish or are reported under an
   intention-to-train rule;
4. official and independent temporal verdicts are reported for every episode;
5. missing or duplicate records are zero after merging;
6. training seed namespaces and evaluation seed namespaces are disjoint from the
   development pilot;
7. the semantic engine version and source digest are embedded in final
   artifacts;
8. all failures, censoring events, and environment terminations are retained.

At launch time, the concrete `--eval-seed-base` values must be recorded and
checked against all development-pilot evaluation seed bases and all
confirmatory training seeds.

This benchmark remains usable even if a trained policy performs poorly. Poor
performance changes the empirical result; it does not justify dropping the
benchmark.

## Portfolio-level algorithmic diversity rule

The paper-level empirical portfolio should include at least two deep-RL
algorithms, and preferably three, across all benchmarks:

- PPO where TL-conditioned recurrent policies are native or already validated;
- SAC where the benchmark has continuous control and a clean SAC implementation;
- DQN where the benchmark has discrete actions and a clean DQN implementation.

DeepLTL PointWorld supplies the TL-conditioned PPO benchmark. Other benchmarks
should supply SAC and/or DQN rather than forcing unsupported algorithms into the
DeepLTL runtime.
