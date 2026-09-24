# DeepLTL runtime compatibility record

Both v3 benchmarks (`deepltl_point`, `letter_env`) vendor the same
`.runtime/deep-ltl` checkout and depend on the two patches below -- LDBA
translation goes through the same `ltl/automata/rabinizer.py` regardless of
which environment or algorithm is training, and both benchmarks' training
loggers go through the same `utils/logging/file_logger.py`. This record and
the patches live at the repository level (`benchmarks/deep_ltl_runtime/`),
not under either benchmark's own folder, since neither one owns them. The
verification evidence below was gathered while building `deepltl_point`
first; `letter_env`'s own independent LDBA/environment smoke
(`evidence/letter_env/validation/letter_env_official_ldba_env_smoke.json`)
confirms the same patches work for its formulas and environment too.

## Frozen inputs

- DeepLTL repository commit:
  `3157200b5910cd7f5c493ffba8913994f2207eaf`.
- DeepLTL `src/ltl/automata/rabinizer.py` SHA-256 before the compatibility
  patch: `bf7f70e8461b30e3c6387779ccf79d67d4b70d55041739f96ec3071119e2c8a2`.
- Rabinizer 4 archive: `https://rabinizer.model.in.tum.de/files/rabinizer-4.zip`,
  SHA-256 `df0d63d8f997fa868e595d3fe609b6c91cea053fa966368fd352f9d71878a1df`
  (downloaded and verified by `scripts/setup_runtime.py`; not redistributed here).
- Rabinizer 4 jar inside it: `rabinizer-4/lib/rabinizer.jar`.
- Rabinizer jar SHA-256:
  `46efc53d769d1016c75b6b76f7cb2db259d4426053591993a62aa874f3ef486d`.
- Windows launcher SHA-256:
  `3aa50769c24573f4e7948a3d6ce4f43c7d25968f6ab58cadffdd3ee375d60f18`.
- Installed Java: Oracle Java 25.0.1 LTS.

## Observed incompatibilities

The installed Rabinizer is operational under Java 25. Its Windows launcher
prints help and its jar converts PointWorld formulas to HOA. Two details make
the unmodified DeepLTL call fail on this machine:

1. DeepLTL hard-codes `rabinizer4/bin/ltl2ldba`, while the existing directory
   is `rabinizer-4` and Windows requires the `.bat` launcher when using that
   path directly.
2. DeepLTL passes `-d`. This Rabinizer 4 CLI no longer accepts that option.
   The specialised `owl.translations.ltl2ldba.LTL2LDBAModule` already emits a
   non-generalised Buchi automaton by default, evidenced by `acc-name: Buchi`
   and `Acceptance: 1 Inf(0)` in its HOA output.
3. DeepLTL's CSV logger imports Unix-only `fcntl`. The Windows compatibility
   patch provides the same exclusive critical section through standard-library
   `msvcrt` locking.
4. The trainer's text logger emits Greek `mu`/`sigma` symbols. Windows CP1252
   cannot encode them, so recorded launches set `PYTHONUTF8=1`.

## Frozen repair

The patch in `patches/deepltl-rabinizer-java25.patch` changes only the
DeepLTL-to-Rabinizer process boundary:

- the jar is supplied by the `RABINIZER_JAR` environment variable;
- Java invokes the same Rabinizer `LTL2LDBAModule` directly;
- the obsolete `-d` flag is omitted;
- `-p` and `-e`, including epsilon-transition retention, are unchanged;
- process failures and empty/non-HOA output fail closed.

It does not change the environment, formulas, LDBA parser, policy,
architecture, optimiser, reward, or evaluation. The runtime clone remains a
disposable, ignored working copy; the patch file and verification evidence are
the provenance-bearing artifacts.

This is a platform/CLI compatibility repair, not a methodological variant.
All 50 frozen tasks converted and passed the upstream HOA parser and
`LDBA.check_valid()` under Java 25. The generated record is
`evidence/deepltl_point/validation/deepltl_point_rabinizer_verification.json`
with its SHA-256 sidecar.

The patched upstream Python API was then exercised through the complete
official environment wrapper stack. A real `PointLtl2-v0` instance built the
automaton, preserved Gymnasium termination/truncation, and traversed
`F (green & F yellow)` to acceptance under deterministic contact probes. That
record is
`evidence/deepltl_point/validation/deepltl_point_official_ldba_env_smoke.json`
with its
SHA-256 sidecar. Both records are development-only compatibility evidence;
they contain no learned-policy or usefulness result.

The separate `patches/deepltl-windows-file-lock.patch` changes only the CSV
configuration-file lock. A live CUDA smoke subsequently completed rollout,
one PPO update, logging, and resumable checkpoint creation. Its hashes and
runtime are recorded in
`evidence/deepltl_point/validation/deepltl_point_ppo_gpu_smoke.json`.
