"""Code provenance for paper decisions.

`code_hash(root)` is the sha256 over every `*.py` file under `root` (sorted relative paths, each path
and its bytes fed into the digest), so it changes whenever the paper code changes, committed or not.
`snapshot_code(root, dest)` copies those files to `dest/<code_hash>/` the first time a hash is seen,
so a decision's `code_hash` can always be matched to the exact code that produced it even when the
working tree was never committed. An existing archive is never overwritten.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def code_hash(root: Path = PAPER_ROOT) -> str:
    """sha256 (hex) over the relative path and bytes of every `*.py` file under `root`."""
    digest = hashlib.sha256()
    for p in _py_files(root):
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(p.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def snapshot_code(root: Path, dest: Path) -> str:
    """Archive the `*.py` files under `root` to `dest/<code_hash>/` unless that archive exists; returns the hash."""
    h = code_hash(root)
    target = dest / h
    if target.exists():
        return h
    tmp = dest / f".{h}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    for p in _py_files(root):
        out = tmp / p.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
    tmp.rename(target)
    return h


def git_version(repo: Path = PAPER_ROOT.parents[2]) -> str:
    """Git sha of `repo`, suffixed `-dirty` when src/ has local changes; `unknown` outside a checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
