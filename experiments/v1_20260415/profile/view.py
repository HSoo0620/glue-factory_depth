"""Pretty-print v1_20260415 timing CSV in transposed terminal layout.

Usage:
    python experiments/v1_20260415/profile/view.py
    python experiments/v1_20260415/profile/view.py --last 5
    python experiments/v1_20260415/profile/view.py --csv path/to/other.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_CSV = Path(__file__).parent / "logs/timing.csv"

SECTIONS: list[tuple[str, list[str]]] = [
    ("Settings", ["timestamp", "descriptor", "scan0", "scan1", "ckpt", "device"]),
    ("Result",   ["n_matches", "n_inliers", "skip_reg"]),
    ("Stages (s)", [
        "load_scan0", "load_scan1",
        "preprocess_scan0", "preprocess_scan1",
        "iss_desc_scan0", "iss_desc_scan1",
        "lg_forward", "extract_matches", "registration",
    ]),
    ("Total (s)", ["total"]),
]


def shorten(s: str, limit: int) -> str:
    return s if len(s) <= limit else "..." + s[-(limit - 3):]


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="View timing CSV in terminal")
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    ap.add_argument("--last", type=int, default=0,
                    help="최근 N개 run 만 (0 이면 전체)")
    ap.add_argument("--ckpt-width", type=int, default=42)
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[err] CSV not found: {path}")
        return 1

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[info] empty CSV")
        return 0

    total_rows = len(rows)
    if args.last > 0:
        rows = rows[-args.last:]

    start = total_rows - len(rows) + 1
    labels = [f"#{start + i} [{r['descriptor']}]" for i, r in enumerate(rows)]

    all_keys = [k for _, keys in SECTIONS for k in keys]
    label_w = max(len(k) for k in all_keys)

    col_w = [len(l) for l in labels]
    for i, row in enumerate(rows):
        for k in all_keys:
            v = row.get(k, "")
            if k == "ckpt":
                v = shorten(v, args.ckpt_width)
            col_w[i] = max(col_w[i], len(str(v)))

    header_line = (
        " " * (label_w + 2)
        + "  ".join(l.ljust(col_w[i]) for i, l in enumerate(labels))
    )
    print(f"CSV: {path}")
    print(f"rows: {total_rows}" + (f" (showing last {len(rows)})" if args.last > 0 else ""))
    print()
    print(header_line)
    print("─" * len(header_line))

    for section_name, keys in SECTIONS:
        print(f"[{section_name}]")
        for k in keys:
            cells = []
            for i, row in enumerate(rows):
                v = row.get(k, "")
                if k == "ckpt":
                    v = shorten(v, args.ckpt_width)
                s = str(v)
                cells.append(s.rjust(col_w[i]) if is_number(s) else s.ljust(col_w[i]))
            print(f"  {k.ljust(label_w)}  " + "  ".join(cells))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
