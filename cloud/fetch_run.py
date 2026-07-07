"""Fetch a prototype's artifact bundle from the Modal artifacts volume.

Local-only helper: shells out to the ``modal`` CLI (no Modal Python import), so
it runs in any environment that has ``modal`` on PATH and an authenticated token.

    python cloud/fetch_run.py --prototype 51 [--run-id <id>] [--register]

It lists ``functional-emotions-artifacts``, picks the requested (or latest) run
directory under ``prototype<N>/``, downloads it into the local
``artifacts/prototype<N>/`` directory, and with ``--register`` invokes
``fe-register-run`` on the downloaded bundle. Progress is printed as JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VOLUME = "functional-emotions-artifacts"
REPOSITORY = Path(__file__).resolve().parents[1]


def _normalize(prototype: str) -> str:
    return prototype.strip().lower().removeprefix("prototype").replace(".", "")


def _subdir(prototype: str) -> str:
    return f"prototype{_normalize(prototype)}"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def list_run_dirs(subdir: str) -> list[str]:
    """Return run-directory names under ``<subdir>/`` on the artifacts volume."""

    result = _run(["modal", "volume", "ls", VOLUME, subdir, "--json"])
    entries = json.loads(result.stdout)
    names: list[str] = []
    for entry in entries:
        name = entry.get("Filename") or entry.get("path") or entry.get("name")
        if not name:
            continue
        names.append(Path(name).name)
    return sorted(names)


def pick_run(run_dirs: list[str], run_id: str | None) -> str:
    if not run_dirs:
        raise SystemExit(f"No run directories found on volume {VOLUME!r}")
    if run_id is None:
        return run_dirs[-1]
    for name in run_dirs:
        if name == run_id or run_id in name:
            return name
    raise SystemExit(f"Run id {run_id!r} not found; available: {', '.join(run_dirs)}")


def download(subdir: str, run_dir: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / run_dir
    if target.exists():
        shutil.rmtree(target)
    _run(["modal", "volume", "get", VOLUME, f"{subdir}/{run_dir}", str(destination)])
    return target


def register(bundle: Path) -> dict:
    executable = shutil.which("fe-register-run")
    cmd = [executable, str(bundle)] if executable else [
        sys.executable,
        "-m",
        "functional_emotions.registry",
        str(bundle),
    ]
    result = _run(cmd)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"stdout": result.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and optionally register a Modal run bundle")
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    subdir = _subdir(args.prototype)
    run_dirs = list_run_dirs(subdir)
    run_dir = pick_run(run_dirs, args.run_id)
    destination = REPOSITORY / "artifacts" / subdir
    bundle = download(subdir, run_dir, destination)

    report: dict = {
        "prototype": args.prototype,
        "run_dir": run_dir,
        "downloaded_to": str(bundle),
        "registered": None,
    }
    if args.register:
        report["registered"] = register(bundle)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
