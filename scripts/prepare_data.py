#!/usr/bin/env python3
"""Fetch, generate, and checksum every benchmark. Idempotent.

No benchmark blob is committed. This script reconstructs `data/` from public
sources and records a checksum for everything it produces, so a reviewer can
confirm they are looking at the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VENV_PY = ROOT / ".venv" / "bin" / "python"

PROOFWRITER_URL = (
    "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
)
PROOFWRITER_SHA256 = "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26"
PW_MEMBER = "proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl"

BBH_BASE = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"
BBH_TASKS = (
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
)

REPOS = {
    "prontoqa_src": "https://github.com/asaparov/prontoqa.git",
    "folio_src": "https://github.com/Yale-LILY/FOLIO.git",
}

# The experimental grid. Hops and seeds here define exactly what gets generated.
HOPS = (1, 2, 3, 4, 5)
SEEDS = (1, 2, 3)
TRIALS_PER_HOP = 120


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


def clone_repos() -> dict[str, str]:
    commits = {}
    for name, url in REPOS.items():
        dest = DATA / name
        if not dest.exists():
            log(f"cloning {url}")
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                capture_output=True,
            )
        commits[name] = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        log(f"{name} @ {commits[name][:12]}")
    return commits


def fetch_proofwriter() -> str:
    zpath = DATA / "proofwriter.zip"
    if not zpath.exists():
        log(f"downloading ProofWriter (214 MB) from {PROOFWRITER_URL}")
        urllib.request.urlretrieve(PROOFWRITER_URL, zpath)
    digest = sha256(zpath)
    if digest != PROOFWRITER_SHA256:
        raise SystemExit(
            f"ProofWriter checksum mismatch.\n  expected {PROOFWRITER_SHA256}\n  got      {digest}\n"
            "Refusing to proceed: the upstream artefact is not the one this study was built against."
        )
    log(f"ProofWriter sha256 verified: {digest[:16]}...")

    out = DATA / "pw_extract" / PW_MEMBER
    if not out.exists():
        log(f"extracting {PW_MEMBER}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extract(PW_MEMBER, DATA / "pw_extract")
    return digest


def generate_prontoqa() -> None:
    """Generate ProntoQA at each hop count and seed via the authors' own script."""
    src = DATA / "prontoqa_src"
    out = DATA / "prontoqa_gen"
    out.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for hops in HOPS:
            target = out / f"{hops}hop_seed{seed}.json"
            if target.exists():
                continue
            log(f"generating ProntoQA hops={hops} seed={seed} n={TRIALS_PER_HOP}")
            subprocess.run(
                [
                    str(VENV_PY),
                    "run_experiment.py",
                    "--model-name",
                    "json",
                    "--model-size",
                    "dummy",
                    "--num-trials",
                    str(TRIALS_PER_HOP),
                    "--min-hops",
                    str(hops),
                    "--max-hops",
                    str(hops),
                    "--ontology",
                    "fictional",  # fictional: no world-knowledge shortcut, no contamination
                    "--few-shot-examples",
                    "0",
                    "--seed",
                    str(seed),
                ],
                cwd=src,
                check=True,
                capture_output=True,
            )
            produced = src / f"{hops}hop_0shot_seed{seed}.json"
            if not produced.exists():
                cands = sorted(src.glob(f"{hops}hop_*seed{seed}.json"))
                if not cands:
                    raise SystemExit(f"ProntoQA produced no file for hops={hops} seed={seed}")
                produced = cands[0]
            produced.replace(target)


def fetch_bbh() -> None:
    out = DATA / "bbh"
    out.mkdir(parents=True, exist_ok=True)
    for task in BBH_TASKS:
        dest = out / f"{task}.json"
        if dest.exists():
            continue
        url = f"{BBH_BASE}/{task}.json"
        try:
            log(f"downloading BBH {task}")
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: BBH {task} unavailable ({e}). It is an optional OOD check.")


def write_checksums(commits: dict[str, str], pw_digest: str) -> None:
    lines = [
        "# Benchmark provenance. Regenerate with `make data`.",
        f"# ProofWriter archive: {PROOFWRITER_URL}",
        f"{pw_digest}  proofwriter.zip",
    ]
    for name, commit in commits.items():
        lines.append(f"# git {name} @ {commit}")

    for path in sorted((DATA / "prontoqa_gen").glob("*.json")):
        lines.append(f"{sha256(path)}  prontoqa_gen/{path.name}")
    for path in sorted((DATA / "bbh").glob("*.json")) if (DATA / "bbh").exists() else []:
        lines.append(f"{sha256(path)}  bbh/{path.name}")
    folio = DATA / "folio_src/data/v0.0/folio-validation.jsonl"
    if folio.exists():
        lines.append(f"{sha256(folio)}  folio_src/data/v0.0/folio-validation.jsonl")
    pw = DATA / "pw_extract" / PW_MEMBER
    if pw.exists():
        lines.append(f"{sha256(pw)}  pw_extract/{PW_MEMBER}")

    (DATA / "checksums.sha256").write_text("\n".join(lines) + "\n")
    log(f"wrote {DATA / 'checksums.sha256'} ({len(lines)} lines)")


def main() -> int:
    DATA.mkdir(exist_ok=True)
    commits = clone_repos()
    pw_digest = fetch_proofwriter()
    generate_prontoqa()
    fetch_bbh()
    write_checksums(commits, pw_digest)

    counts = {
        "prontoqa_gen files": len(list((DATA / "prontoqa_gen").glob("*.json"))),
        "bbh files": len(list((DATA / "bbh").glob("*.json"))) if (DATA / "bbh").exists() else 0,
    }
    log(f"done: {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
