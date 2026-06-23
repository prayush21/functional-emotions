from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tracking import register_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a completed experiment bundle")
    parser.add_argument("bundle", type=Path, help="Directory containing config/environment/metrics")
    parser.add_argument("--registry", type=Path, default=Path("results"))
    args = parser.parse_args()
    destination = register_bundle(args.bundle, args.registry)
    manifest = json.loads((destination / "manifest.json").read_text())
    print(json.dumps({"registered": str(destination), "run_id": manifest["run_id"]}, indent=2))


if __name__ == "__main__":
    main()
