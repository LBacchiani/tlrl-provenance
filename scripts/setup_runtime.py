#!/usr/bin/env python
"""Provision the external runtime that the benchmark experiments depend on.

This repository deliberately does NOT redistribute third-party software.  This
script fetches, verifies, and wires up everything the benchmarks need:

1. Java (checked, never installed): Rabinizer 4 is a Java program.
2. Rabinizer 4 (downloaded from its official host, SHA-256 pinned):
   ``rabinizer-4/lib/rabinizer.jar``.
3. DeepLTL (cloned from GitHub at a pinned commit) plus the two compatibility
   patches in ``benchmarks/deep_ltl_runtime/patches/`` -> ``.runtime/deep-ltl``.
4. The DeepLTL-side simulator package ``safety_gymnasium`` (editable install
   from the DeepLTL checkout) unless ``--skip-pip`` is given.
5. The trained PPO checkpoints, copied from ``experiments/ppo`` into the
   DeepLTL checkout where the upstream code and the audit adapters expect them.
6. The evidence corpora that are stored compressed (inflated and verified
   against their SHA-256 sidecars).

Every step is idempotent; re-running the script repairs a partial setup.
``--check`` only verifies and never modifies anything.

Usage (from the repository root, inside the Python 3.10 benchmark environment):

    python scripts/setup_runtime.py            # provision everything
    python scripts/setup_runtime.py --check    # verify an existing setup
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]

# --- Pinned third-party inputs -------------------------------------------------

RABINIZER_URL = "https://rabinizer.model.in.tum.de/files/rabinizer-4.zip"
RABINIZER_ZIP_SHA256 = "df0d63d8f997fa868e595d3fe609b6c91cea053fa966368fd352f9d71878a1df"
RABINIZER_JAR_SHA256 = "46efc53d769d1016c75b6b76f7cb2db259d4426053591993a62aa874f3ef486d"
RABINIZER_DIR = ROOT / "rabinizer-4"
RABINIZER_JAR = RABINIZER_DIR / "lib" / "rabinizer.jar"
RABINIZER_MAIN = "owl.translations.ltl2ldba.LTL2LDBAModule"

DEEPLTL_URL = "https://github.com/mathiasj33/deep-ltl.git"
DEEPLTL_COMMIT = "3157200b5910cd7f5c493ffba8913994f2207eaf"
DEEPLTL_DIR = ROOT / ".runtime" / "deep-ltl"
# SHA-256 of src/ltl/automata/rabinizer.py before / after the compatibility patch
# (LF blob as committed upstream; the CRLF working-tree copy on Windows hashes differently).
DEEPLTL_RABINIZER_PY_ORIGINAL_SHA256 = "129cdd5892709f503c02449f8b5a0f9b7719017d2a15f5533a8c2fdfaddf7fbe"
PATCH_DIR = ROOT / "benchmarks" / "deep_ltl_runtime" / "patches"
PATCHES = ("deepltl-rabinizer-java25.patch", "deepltl-windows-file-lock.patch")

MIN_JAVA_MAJOR = 11  # Rabinizer 4 / DeepLTL documentation require Java 11+.


# --- Helpers -------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def say(message: str) -> None:
    print(message, flush=True)


class SetupError(RuntimeError):
    pass


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)
    except FileNotFoundError as exc:
        raise SetupError(f"required program not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SetupError(
            f"command failed ({exc.returncode}): {' '.join(command)}\n{exc.stdout}\n{exc.stderr}"
        ) from exc


# --- 1. Java -------------------------------------------------------------------


def find_java() -> str:
    java = shutil.which("java")
    if java is None and os.environ.get("JAVA_HOME"):
        candidate = Path(os.environ["JAVA_HOME"]) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if candidate.is_file():
            java = str(candidate)
    if java is None:
        raise SetupError(
            "Java was not found. Rabinizer 4 (used to translate LTL into LDBAs) is a Java program and "
            f"needs a Java runtime >= {MIN_JAVA_MAJOR} on PATH (or JAVA_HOME set). Install e.g. Temurin/OpenJDK "
            "(https://adoptium.net) and re-run."
        )
    return java


def java_major(java: str) -> int:
    completed = subprocess.run([java, "-version"], capture_output=True, text=True)
    text = completed.stderr + completed.stdout
    match = re.search(r'version "(\d+)(?:\.(\d+))?', text)
    if not match:
        raise SetupError(f"could not parse `java -version` output:\n{text}")
    major = int(match.group(1))
    return int(match.group(2)) if major == 1 and match.group(2) else major


def check_java() -> str:
    java = find_java()
    major = java_major(java)
    if major < MIN_JAVA_MAJOR:
        raise SetupError(f"Java {major} found at {java}; Rabinizer 4 requires Java >= {MIN_JAVA_MAJOR}.")
    say(f"[java] OK: Java {major} at {java}")
    return java


# --- 2. Rabinizer 4 --------------------------------------------------------------


def rabinizer_ok() -> bool:
    return RABINIZER_JAR.is_file() and sha256_file(RABINIZER_JAR) == RABINIZER_JAR_SHA256


def install_rabinizer(force: bool) -> None:
    if rabinizer_ok() and not force:
        say(f"[rabinizer] OK: {RABINIZER_JAR.relative_to(ROOT)} (SHA-256 verified)")
        return
    say(f"[rabinizer] downloading {RABINIZER_URL}")
    say("[rabinizer] Rabinizer 4 is third-party software (GPL-3.0) and is not redistributed in this repository.")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "rabinizer-4.zip"
        request = urllib.request.Request(RABINIZER_URL, headers={"User-Agent": "tlrl-provenance-setup"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
        except OSError as exc:
            raise SetupError(
                f"could not download Rabinizer 4 from {RABINIZER_URL}: {exc}\n"
                f"Download it manually, verify SHA-256 {RABINIZER_ZIP_SHA256}, and unzip it so that "
                f"{RABINIZER_JAR} exists."
            ) from exc
        actual = sha256_file(archive)
        if actual != RABINIZER_ZIP_SHA256:
            raise SetupError(
                f"Rabinizer archive checksum mismatch (expected {RABINIZER_ZIP_SHA256}, got {actual}). "
                "Refusing to use an unverified archive; the experiments are pinned to this exact build."
            )
        if RABINIZER_DIR.exists():
            shutil.rmtree(RABINIZER_DIR)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                target = (ROOT / member).resolve()
                if not str(target).startswith(str(RABINIZER_DIR.resolve())):
                    raise SetupError(f"unexpected path in Rabinizer archive: {member}")
            bundle.extractall(ROOT)
    for launcher in (RABINIZER_DIR / "bin").glob("*"):
        if launcher.suffix != ".bat":
            launcher.chmod(launcher.stat().st_mode | 0o111)
    if not rabinizer_ok():
        raise SetupError(
            f"{RABINIZER_JAR} does not match the pinned SHA-256 {RABINIZER_JAR_SHA256} after extraction."
        )
    say(f"[rabinizer] installed {RABINIZER_JAR.relative_to(ROOT)} (SHA-256 verified)")


def smoke_rabinizer(java: str) -> None:
    completed = subprocess.run(
        [java, "-classpath", str(RABINIZER_JAR), RABINIZER_MAIN, "-i", "F (a & F b)", "-p", "-e"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0 or not completed.stdout.startswith("HOA: v1"):
        raise SetupError(f"Rabinizer smoke test failed:\n{completed.stdout}\n{completed.stderr}")
    say("[rabinizer] smoke test OK (LTL -> LDBA/HOA translation works)")


# --- 3. DeepLTL --------------------------------------------------------------------


def git(*args: str, cwd: Path | None = None) -> str:
    return run(["git", *args], cwd=str(cwd) if cwd else None).stdout.strip()


def deepltl_patched() -> bool:
    target = DEEPLTL_DIR / "src" / "ltl" / "automata" / "rabinizer.py"
    return target.is_file() and "RABINIZER_JAR" in target.read_text(encoding="utf-8")


def install_deepltl(force: bool) -> None:
    if DEEPLTL_DIR.exists() and force:
        shutil.rmtree(DEEPLTL_DIR, onerror=_make_writable)
    if not (DEEPLTL_DIR / ".git").exists():
        say(f"[deepltl] cloning {DEEPLTL_URL}")
        DEEPLTL_DIR.parent.mkdir(parents=True, exist_ok=True)
        # core.autocrlf=false keeps LF endings so the patches apply byte-exactly on Windows too.
        git("-c", "core.autocrlf=false", "clone", "--no-checkout", DEEPLTL_URL, str(DEEPLTL_DIR))
        git("config", "core.autocrlf", "false", cwd=DEEPLTL_DIR)
    head = git("rev-parse", "HEAD", cwd=DEEPLTL_DIR) if (DEEPLTL_DIR / ".git").exists() else ""
    tracked_file = DEEPLTL_DIR / "src" / "ltl" / "automata" / "rabinizer.py"
    if head != DEEPLTL_COMMIT or not tracked_file.is_file():
        # --force also materialises the working tree after a --no-checkout clone.
        git("checkout", "--quiet", "--force", DEEPLTL_COMMIT, cwd=DEEPLTL_DIR)
    if git("rev-parse", "HEAD", cwd=DEEPLTL_DIR) != DEEPLTL_COMMIT:
        raise SetupError("DeepLTL checkout is not at the pinned commit")
    if not deepltl_patched():
        original = DEEPLTL_DIR / "src" / "ltl" / "automata" / "rabinizer.py"
        if sha256_file(original) != DEEPLTL_RABINIZER_PY_ORIGINAL_SHA256:
            raise SetupError(
                "src/ltl/automata/rabinizer.py differs from the pinned upstream file (unexpected local edits "
                "or line-ending conversion). Re-run with --force to re-clone."
            )
        for name in PATCHES:
            patch = PATCH_DIR / name
            git("apply", "--whitespace=nowarn", str(patch), cwd=DEEPLTL_DIR)
            say(f"[deepltl] applied {name}")
    say(f"[deepltl] OK: {DEEPLTL_DIR.relative_to(ROOT)} @ {DEEPLTL_COMMIT[:12]} with compatibility patches")


def _make_writable(func, path, _exc) -> None:  # pragma: no cover - Windows read-only .git objects
    os.chmod(path, 0o700)
    func(path)


def install_safety_gymnasium() -> None:
    package = DEEPLTL_DIR / "src" / "envs" / "zones" / "safety-gymnasium"
    probe = subprocess.run([sys.executable, "-c", "import safety_gymnasium"], capture_output=True, text=True)
    if probe.returncode == 0:
        say("[pip] safety_gymnasium already importable")
        return
    say("[pip] installing DeepLTL's safety-gymnasium (editable)")
    run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(package)])


# --- 5. Checkpoints ----------------------------------------------------------------


def install_checkpoints() -> None:
    source = ROOT / "experiments" / "ppo"
    if not source.is_dir():
        raise SetupError(f"missing checkpoint directory: {source}")
    destination = DEEPLTL_DIR / "experiments" / "ppo"
    copied = skipped = 0
    for file in sorted(p for p in source.rglob("*") if p.is_file()):
        target = destination / file.relative_to(source)
        if target.is_file() and target.stat().st_size == file.stat().st_size and sha256_file(target) == sha256_file(file):
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        copied += 1
    say(f"[checkpoints] PPO checkpoints in {destination.relative_to(ROOT)}: {copied} copied, {skipped} already current")


# --- 6. Compressed evidence ----------------------------------------------------------


def restore_evidence() -> None:
    restored = 0
    for archive in sorted((ROOT / "evidence").rglob("*.jsonl.gz")):
        target = archive.with_suffix("")  # strips .gz
        sidecar = target.with_name(target.name + ".sha256")
        expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
        if target.is_file() and sha256_file(target) == expected:
            continue
        say(f"[evidence] inflating {archive.relative_to(ROOT)}")
        digest = hashlib.sha256()
        partial = target.with_name(target.name + ".part")
        with gzip.open(archive, "rb") as source, partial.open("wb") as out:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != expected:
            partial.unlink()
            raise SetupError(f"{archive.name} does not inflate to the SHA-256 recorded in {sidecar.name}")
        partial.replace(target)
        restored += 1
    say(f"[evidence] compressed corpora verified ({restored} inflated)")


# --- Environment checks ----------------------------------------------------------------


def check_python_environment() -> list[str]:
    problems: list[str] = []
    if sys.version_info[:2] != (3, 10):
        problems.append(
            f"Python {platform.python_version()} in use; the benchmark stack was validated on Python 3.10 "
            "(numpy 1.23.5 / torch 2.2.2 / gymnasium 0.28.1)."
        )
    for module in ("scipy", "numpy", "torch", "gymnasium", "mujoco", "safety_gymnasium"):
        probe = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
        if probe.returncode != 0:
            problems.append(f"cannot import `{module}` (install the benchmark extras: pip install -e \".[benchmarks]\")")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify an existing setup; change nothing")
    parser.add_argument("--force", action="store_true", help="re-download Rabinizer and re-clone DeepLTL")
    parser.add_argument("--skip-pip", action="store_true", help="do not pip-install safety-gymnasium")
    parser.add_argument("--skip-checkpoints", action="store_true", help="do not copy PPO checkpoints into DeepLTL")
    args = parser.parse_args()

    try:
        java = check_java()
        if args.check:
            if not rabinizer_ok():
                raise SetupError("Rabinizer jar missing or not matching the pinned SHA-256")
            say("[rabinizer] OK (SHA-256 verified)")
            smoke_rabinizer(java)
            if not (deepltl_patched() and git("rev-parse", "HEAD", cwd=DEEPLTL_DIR) == DEEPLTL_COMMIT):
                raise SetupError("DeepLTL checkout missing, not at the pinned commit, or not patched")
            say("[deepltl] OK")
            problems = check_python_environment()
            for problem in problems:
                say(f"[env] WARNING: {problem}")
            return 1 if problems else 0
        install_rabinizer(args.force)
        smoke_rabinizer(java)
        install_deepltl(args.force)
        if not args.skip_pip:
            install_safety_gymnasium()
        if not args.skip_checkpoints:
            install_checkpoints()
        restore_evidence()
    except SetupError as exc:
        say(f"\nSETUP FAILED: {exc}")
        return 2
    say("\nSetup complete. Next: python scripts/setup_runtime.py --check && python -m pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
