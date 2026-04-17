"""Stage-level timing utility for v1_20260415 inference demo."""
from __future__ import annotations

import csv as _csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Union

_MetaVal = Union[str, int, float, bool]


class StageRecord:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.append((name, time.perf_counter() - t0))

    def _sum_by_names(self, names: list[str]) -> float:
        bucket: dict[str, float] = {}
        for n, t in self.records:
            bucket[n] = bucket.get(n, 0.0) + t
        return sum(bucket.get(n, 0.0) for n in names)

    def print_table(self, groups: dict[str, list[str]],
                    detailed: bool = False) -> None:
        total = sum(t for _, t in self.records)
        if detailed:
            for name, t in self.records:
                print(f"[{name:<18}]  {t:>7.3f}s")
            print("─" * 30)
        for gname, members in groups.items():
            s = self._sum_by_names(members)
            print(f"[{gname:<12}]  {s:>7.3f}s")
        print("─" * 30)
        print(f"[{'Total':<12}]  {total:>7.3f}s")

    def save_csv(self, path: Union[str, Path],
                 metadata: Mapping[str, _MetaVal],
                 stage_order: list[str] | None = None) -> None:
        """Append one row of (metadata + per-stage timings + total) to CSV.

        On first write, header row is emitted. When ``stage_order`` is given,
        it defines the fixed stage-column schema regardless of which stages
        were actually recorded; missing stages yield empty cells. Without it,
        columns come from the current ``records`` order.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        recorded: dict[str, float] = {n: t for n, t in self.records}
        if stage_order is None:
            stage_order = [n for n, _ in self.records]
        total = sum(t for _, t in self.records)

        row: dict[str, str] = {k: str(v) for k, v in metadata.items()}
        for n in stage_order:
            row[n] = f"{recorded[n]:.4f}" if n in recorded else ""
        row["total"] = f"{total:.4f}"

        header = list(metadata.keys()) + list(stage_order) + ["total"]
        exists = path.exists()
        with open(path, "a", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=header)
            if not exists:
                w.writeheader()
            w.writerow(row)
