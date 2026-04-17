"""스모크용 CLI.

python examples/run_cli.py --scanned <path> [--master <path>] [--descriptor fpfh|shot]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from depth_registration import register_pair  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scanned", required=True)
    ap.add_argument("--master", default=None)
    ap.add_argument("--descriptor", choices=["fpfh", "shot"], default="fpfh")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    r = register_pair(args.scanned, args.master,
                      descriptor=args.descriptor, device=args.device)
    printable = {k: v for k, v in r.items()
                 if k not in ("T", "R", "t",
                              "matches_scanned_xyz", "matches_master_xyz")}
    print(json.dumps(printable, indent=2, default=str))
    print(f"T =\n{r['T']}")


if __name__ == "__main__":
    main()
