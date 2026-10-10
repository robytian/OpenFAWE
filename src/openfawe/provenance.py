"""Explicit input, source and environment identities; atomic status updates."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_identity():
    package = Path(__file__).resolve().parent
    files = {"openfawe/" + p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}
    root = package.parent.parent
    for name in ("pyproject.toml", "requirements-core.lock", "requirements-monitor.lock"):
        if (root / name).is_file():
            files[name] = sha256_file(root / name)
    git = {"available": False}
    try:
        def call(*args):
            return subprocess.check_output(["git", *args], cwd=package, stderr=subprocess.DEVNULL,
                                           timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).decode("utf-8").strip()
        git = {"available": True, "head": call("rev-parse", "HEAD"),
               "status": call("status", "--porcelain=v1", "--untracked-files=all")}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"files_sha256": files, "git": git}


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]))}
