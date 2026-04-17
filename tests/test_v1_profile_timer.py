import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/v1_20260415/profile"))

from _timer import StageRecord  # noqa: E402


def test_timer_records_single_block():
    rec = StageRecord()
    with rec.timer("block_a"):
        time.sleep(0.02)
    assert len(rec.records) == 1
    name, elapsed = rec.records[0]
    assert name == "block_a"
    assert 0.01 <= elapsed <= 1.0


def test_timer_records_multiple_blocks_in_order():
    rec = StageRecord()
    with rec.timer("a"):
        time.sleep(0.005)
    with rec.timer("b"):
        time.sleep(0.005)
    names = [n for n, _ in rec.records]
    assert names == ["a", "b"]


def test_print_table_4stage_summary():
    rec = StageRecord()
    rec.records = [
        ("load", 0.1),
        ("preprocess", 0.2),
        ("descriptor", 0.5),
        ("matching", 0.05),
        ("registration", 0.1),
    ]
    groups = {
        "Pre":   ["load", "preprocess"],
        "Desc":  ["descriptor"],
        "Match": ["matching"],
        "Reg":   ["registration"],
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec.print_table(groups, detailed=False)
    out = buf.getvalue()
    assert "Pre" in out and "Desc" in out
    assert "Match" in out and "Reg" in out
    assert "Total" in out
    assert "0.300" in out       # Pre sum
    assert "0.950" in out       # Total


def test_print_table_detailed_includes_individual():
    rec = StageRecord()
    rec.records = [("load", 0.123), ("descriptor", 0.456)]
    groups = {"Pre": ["load"], "Desc": ["descriptor"]}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec.print_table(groups, detailed=True)
    out = buf.getvalue()
    assert "load" in out and "descriptor" in out
    assert "0.123" in out and "0.456" in out


def test_save_csv_creates_header_and_row(tmp_path):
    rec = StageRecord()
    rec.records = [("load", 0.1), ("desc", 0.5)]
    csv_path = tmp_path / "t.csv"
    rec.save_csv(csv_path, {"descriptor": "SHOT", "device": "cuda"})
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "descriptor,device,load,desc,total"
    assert lines[1] == "SHOT,cuda,0.1000,0.5000,0.6000"


def test_save_csv_appends_without_duplicating_header(tmp_path):
    rec = StageRecord()
    rec.records = [("load", 0.2)]
    csv_path = tmp_path / "t.csv"
    rec.save_csv(csv_path, {"descriptor": "SHOT"})
    rec2 = StageRecord()
    rec2.records = [("load", 0.3)]
    rec2.save_csv(csv_path, {"descriptor": "FPFH"})
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("descriptor,")
    assert lines[1].startswith("SHOT,")
    assert lines[2].startswith("FPFH,")


def test_save_csv_creates_parent_directory(tmp_path):
    rec = StageRecord()
    rec.records = [("a", 0.1)]
    csv_path = tmp_path / "nested/dir/t.csv"
    rec.save_csv(csv_path, {"k": "v"})
    assert csv_path.exists()


def test_save_csv_stage_order_fills_missing_with_empty(tmp_path):
    rec = StageRecord()
    rec.records = [("a", 0.1), ("c", 0.3)]  # "b" not recorded (skipped stage)
    csv_path = tmp_path / "t.csv"
    rec.save_csv(csv_path, {"k": "v"}, stage_order=["a", "b", "c"])
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "k,a,b,c,total"
    assert lines[1] == "v,0.1000,,0.3000,0.4000"


def test_save_csv_stage_order_stable_across_runs(tmp_path):
    csv_path = tmp_path / "t.csv"
    order = ["a", "b"]
    r1 = StageRecord(); r1.records = [("a", 0.1)]              # b skipped
    r1.save_csv(csv_path, {"k": "v1"}, stage_order=order)
    r2 = StageRecord(); r2.records = [("a", 0.2), ("b", 0.3)]  # both present
    r2.save_csv(csv_path, {"k": "v2"}, stage_order=order)
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "k,a,b,total"
    assert lines[1] == "v1,0.1000,,0.1000"
    assert lines[2] == "v2,0.2000,0.3000,0.5000"
