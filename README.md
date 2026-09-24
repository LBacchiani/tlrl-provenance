# tlrl-provenance

Replication package for the paper **"What Made It Satisfy? A Semantic Audit for
Temporal-Logic-Guided Deep Reinforcement Learning"**.

It contains

* **`tlrl_provenance`** - a proof-producing finite-trace (LTLf) semantic kernel:
  given a formula and a labeled trace it returns the verdict *and* auditable
  evidence explaining it (see [docs/SEMANTICS.md](docs/SEMANTICS.md));
* **`tlrl_benchmarks`** - adapters that audit learned controllers on three
  deep-RL benchmarks from DeepLTL (PointWorld, LetterEnv, FlatWorld);
* **all trained policies** (20 audited policies - 5 seeds x {PointWorld PPO,
  LetterEnv DQN, LetterEnv PPO, FlatWorld PPO} - plus 4 development pilots) in
  [`experiments/`](experiments/README.md);
* **all frozen audit campaigns and results** in [`evidence/`](evidence/), with
  SHA-256 sidecars; and
* scripts to verify, re-analyse and re-run everything.

> **Third-party requirements you must know about**
>
> * **Java (>= 11) is required.** LTL-to-automaton translation is done by
>   **Rabinizer 4**, a Java program. Java is *checked* but never installed by
>   this repository; install a JRE/JDK yourself (e.g. <https://adoptium.net>).
>   The reported experiments ran on Java 25.
> * **Rabinizer 4 is not redistributed here.** (It is GPL-3.0 software from the
>   Technical University of Munich.) `scripts/setup_runtime.py` downloads the
>   exact build used in the paper from its official host and refuses it unless
>   its SHA-256 matches. DeepLTL (MIT) is likewise cloned at a pinned commit.
>   See [Third-party software](#third-party-software).

## Contents

1. [Repository layout](#repository-layout)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Verify the installation](#verify-the-installation)
5. [Reproducing the paper's results](#reproducing-the-papers-results)
6. [Using the library](#using-the-library)
7. [Retraining the policies (optional)](#retraining-the-policies-optional)
8. [Artifact identity and known caveats](#artifact-identity-and-known-caveats)
9. [Third-party software](#third-party-software), [Citation](#citation), [License](#license)

## Repository layout

```
src/tlrl_provenance/     semantic kernel (LTLf evaluation, provenance, prefix semantics, statistics, campaigns)
src/tlrl_benchmarks/     benchmark adapters: deepltl_point/, letter_env/ (DQN + PPO), flatworld/
tests/                   test suite (run with pytest)
scripts/                 setup, verification, reproduction, analysis, training - indexed in scripts/README.md
experiments/             trained checkpoints + logs, with CHECKSUMS.sha256   (see experiments/README.md)
evidence/                frozen protocols, plans, per-episode records, summaries, reports (+ .sha256 sidecars)
benchmarks/              per-benchmark corpus freezes, training protocols, runtime patches
formal/coq/              machine-checked proof of the provenance-evidence theorem
docs/                    SEMANTICS.md, BENCHMARKS.md, CONFIRMATORY_AUDITS.md
environment/             pinned Python dependencies
```

Created by `scripts/setup_runtime.py` (git-ignored, never committed):
`rabinizer-4/` (Rabinizer 4) and `.runtime/deep-ltl/` (patched DeepLTL checkout).

## Requirements

| | Kernel + tests only | Full benchmark stack |
|---|---|---|
| Python | >= 3.10 | **3.10** (numpy 1.23.5, torch 2.2.2, gymnasium 0.28.1, mujoco 2.3.3) |
| Java | not needed | **>= 11 on `PATH` (or `JAVA_HOME`)** |
| git | - | required (clones DeepLTL) |
| Disk | ~0.5 GB | ~3 GB (checkpoints + evidence + PyTorch + DeepLTL) |
| GPU | - | not needed for auditing/reproduction (optional for retraining) |
| Network | install only | install/setup only |

Tested with **CPython 3.10 and Java 25**. On Windows, clone close to the drive
root or enable long paths (`git config --global core.longpaths true`); the
longest tracked path is 153 characters.

## Setup

### A. Kernel only (no Java, no benchmarks)

```bash
python -m pip install -e ".[dev]"
python -m pytest -q          # benchmark-dependent tests skip themselves if torch is absent
```

### B. Full setup (kernel + benchmarks + reproduction)

```bash
git clone https://github.com/LBacchiani/tlrl-provenance.git
cd tlrl-provenance
# (Zenodo users: unzip the archive instead of cloning, then cd into it.)

# 1. Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate                # Windows PowerShell: py -3.10 -m venv .venv ; .venv\Scripts\Activate.ps1
#   or with conda:  conda create -n tlrl python=3.10 && conda activate tlrl

# 2. Pinned dependencies, then this package
python -m pip install -r environment/requirements-benchmarks.txt
python -m pip install -e ".[dev]"

# 3. Provision third-party runtime: checks Java, downloads + verifies Rabinizer 4,
#    clones + patches DeepLTL, installs its simulator, installs PPO checkpoints,
#    inflates the compressed PointWorld corpus
python scripts/setup_runtime.py
```

`setup_runtime.py` is idempotent and prints one line per step. Useful flags:
`--check` (verify only), `--force` (re-download/re-clone), `--skip-pip`,
`--skip-checkpoints`. If Java is missing it stops with an explanatory message.

*GPU note:* the default PyPI `torch==2.2.2` is enough for everything except
GPU training; for CUDA use
`pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121`.

## Verify the installation

```bash
python scripts/setup_runtime.py --check   # Java, Rabinizer hash + LTL->LDBA smoke test, DeepLTL commit/patches, imports
python scripts/verify_artifacts.py        # SHA-256 of every evidence file and every checkpoint
python -m pytest -q                       # full test suite
```

## Reproducing the paper's results

Three levels, cheapest first. None overwrites the frozen evidence.

**Level 1 - integrity** (seconds): `python scripts/verify_artifacts.py`
checks every frozen result and checkpoint against recorded SHA-256 digests.

**Level 2 - re-analysis** (CPU only): regenerate every statistic,
significance test and verification record from the frozen per-episode
corpora and compare **byte-for-byte** with the frozen output:

```bash
python scripts/reproduce_analysis.py            # or: --list, --only significance
```

**Level 3 - re-audit from the checkpoints** (rolls out the shipped policies
again in the real environments and compares every episode with the frozen
record):

```bash
python scripts/reproduce_audit.py letter_env_ppo --max-shards 2   # quick: 2-shard sample
python scripts/reproduce_audit.py letter_env_ppo --full           # entire 5,000-episode campaign
```

Campaigns: `deepltl_point`, `letter_env_dqn`, `letter_env_ppo`, `flatworld_ppo`.

Measured on a clean Windows 11 / Python 3.10 install: the quick mode (2 of 50 shards,
200 episodes) reproduced **all 200 episodes exactly** for LetterEnv DQN, LetterEnv PPO
and FlatWorld PPO (about one minute each), and 199/200 for PointWorld (one MuJoCo
trajectory diverged by 5 steps, see caveats). Level 2 regenerates all 8 outputs: 6 byte-identical, and 2 identical except for fields
the tool ignores (`machinery_verification.json`: timing, platform, Python version and
source digest; `deepltl_point_full_semantic_analysis.json`: the p-value convention below).
Frozen campaign settings (seeds, formulas, 20 episodes per policy/formula,
scenario pairing) are reused unchanged; outputs go to `reproduction/<campaign>/`.
See `docs/CONFIRMATORY_AUDITS.md` for the design.

Where each paper result lives:

| Result | Frozen artifact |
|---|---|
| PointWorld PPO audit (5 policies x 50 formulas x 20 episodes) | `evidence/deepltl_point/confirmatory/semantic/` |
| LetterEnv DQN and PPO audits | `evidence/letter_env/{dqn,ppo}/confirmatory/semantic/` |
| FlatWorld PPO audit (49 formulas) | `evidence/flatworld/ppo/confirmatory/semantic/` |
| Paired permutation / significance analyses | `*_significance_analysis.json` next to each audit |
| Robustness checks (stochastic execution, 10x horizon, operator generality, grounding noise, overhead) | `evidence/deepltl_point/confirmatory/semantic/*_stress_test.json` |
| Kernel verification, prefix-completeness check, residual-size guard | `evidence/*.json` |
| Coq proof of the evidence-extraction theorem | `formal/coq/` (re-check with `coqc`; see its README) |

Interpretation guide: `docs/BENCHMARKS.md` and each benchmark's `evidence/*/README.md`.

## Using the library

```python
from tlrl_provenance import LabeledTrace, evaluate_with_provenance, parse_formula

formula = parse_formula("G(request -> F(grant or cancel)) and G(not collision)")
trace = LabeledTrace([{"request"}, {"grant"}, set()])
certificate = evaluate_with_provenance(formula, trace)
print(certificate.verdict.value)
print([fact.to_source() for fact in certificate.necessary_evidence])
```

Supported logic, certificates, bounded/exact support enumeration, proposition
grounding and the fail-closed `audit_traces` API are described in
[docs/SEMANTICS.md](docs/SEMANTICS.md). Supported operators: atoms, `top`,
`bottom`, `!`, `and`, `or`, `->`, `<->`, `X`, `Xw`, `F`, `G`, `U`, `R`, `W`
(finite-trace semantics).

## Retraining the policies (optional)

Retraining is **not needed** to reproduce anything: the trained policies are
shipped, and re-training is not bit-reproducible (GPU non-determinism), so a
retrained policy is a new sample, not the audited one.

* **DQN (LetterEnv):** `scripts/train_letter_env_dqn.py` (protocol:
  `benchmarks/letter_env/DQN_PROTOCOL.md`). Its frozen plans in `evidence/` make
  it refuse to re-run under the original experiment names; pass a new `--name`.
* **PPO (all benchmarks):** DeepLTL's own trainer. The exact command, seed and
  hyper-parameters of every run are recorded in
  `evidence/*/confirmatory/training/*_seed_*_manifest.json` (field `command`),
  e.g. for LetterEnv seed 940001, run from `.runtime/deep-ltl` with
  `PYTHONPATH=src/` and `RABINIZER_JAR=<repo>/rabinizer-4/lib/rabinizer.jar`:
  `python src/train/train_ppo.py --env LetterEnv-v0 --steps_per_process 128 --batch_size 256 --lr 0.0003 --discount 0.94 --gae_lambda 0.95 --entropy_coef 0.01 --epochs 8 --num_steps 5000000 --model_config LetterEnv-v0 --curriculum LetterEnv-v0 --name <new_name> --seed 940001 --device gpu --num_procs 16 --log_csv --save --no-log_wandb`.
  The `launch_*` / `freeze_*` scripts are the immutable records of how the
  original campaigns were launched; they refuse to overwrite existing runs.

## Artifact identity and known caveats

* **Kernel version.** The semantic kernel in `src/tlrl_provenance/` is the final
  revision used for the paper. Each frozen plan records the kernel's source
  digest *at freeze time*, and the campaigns were frozen at different
  revisions (PointWorld `9010148258f2...`, FlatWorld `e527a4cad102...`,
  LetterEnv `766e22d85c82...`; the shipped kernel is `7a210d404fa5...`). Re-running
  a campaign therefore embeds a different digest, which is why
  `reproduce_audit.py` compares episode outcomes rather than digests, and why
  running `run_confirmatory_semantic_audit.py` in place against a frozen plan is
  refused. `evidence/prefix_exact_completeness_verification.json` documents
  the regression check of the kernel against the frozen PointWorld corpus.
* **Superseded p-value convention.** `deepltl_point_full_semantic_analysis.json`
  was generated before the add-one Monte Carlo correction was added to the kernel
  (its p-values are `count/10000`; the shipped kernel gives `(count+1)/10001`).
  All other fields regenerate identically. The paper's PointWorld p-values come
  from `deepltl_point_significance_analysis.json`, which regenerates byte-identically.
* **Frozen files are not edited.** They contain absolute paths of the original
  Windows machine and the original development layout (e.g. `semantic_full/`);
  editing them would invalidate their hashes.
* **Simulator determinism.** PointWorld uses MuJoCo; exact trajectories are not
  guaranteed to replay bit-for-bit across platforms, which is why the audited
  traces themselves are retained and hash-anchored in the frozen corpus
  (Level 2 is exact; Level 3 is exact only where the simulator is).
* **Storage.** `deepltl_point_full_confirmatory_semantic_records.jsonl` (153 MB)
  is stored gzip-compressed (GitHub's file limit); its sidecar hashes the
  uncompressed bytes, and `setup_runtime.py` / `verify_artifacts.py` handle it.
* **Line endings.** `.gitattributes` disables newline conversion for frozen
  artifacts; do not re-encode them.

## Third-party software

| Software | Role | License | How obtained |
|---|---|---|---|
| Rabinizer 4 (Java) | LTL -> LDBA translation | GPL-3.0 | downloaded by `setup_runtime.py` from <https://rabinizer.model.in.tum.de/files/rabinizer-4.zip>; zip SHA-256 `df0d63d8...a1df`, jar SHA-256 `46efc53d...486d` |
| DeepLTL (Jackermeier & Abate, ICLR 2025) | environments, PPO trainer | MIT | cloned at commit `3157200b5910`, with `benchmarks/deep_ltl_runtime/patches/*.patch` applied |
| Safety-Gymnasium, torch_ac | via DeepLTL | Apache-2.0, MIT | via DeepLTL |

The two patches only adapt DeepLTL's process/OS boundary (calling the Rabinizer
jar directly; Windows file locking); see
`benchmarks/deep_ltl_runtime/DEEPLTL_RUNTIME_COMPATIBILITY.md`.


## License

MIT (see [LICENSE](LICENSE)) for the code and data in this repository.
Third-party components keep their own licenses.
