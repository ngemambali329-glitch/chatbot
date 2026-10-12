"""The committed data/processed files must match what the build script produces from data/raw."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_knowledge import expand_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_expand_code():
    assert expand_code("4MTH271L_B") == [("4MTH271L", "B")]
    assert expand_code("5EEE361P/T") == [("5EEE361P", ""), ("5EEE361T", "")]
    assert expand_code("5EEE311L/5EEE351") == [("5EEE311L", ""), ("5EEE351", "")]


def test_processed_files_are_up_to_date(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_knowledge.py"), "--out", str(tmp_path)],
        check=True, capture_output=True,
    )
    for name in ["timetable_entries.json", "calendar_events.json", "handbook_chunks.json", "handbook_lines.json"]:
        built = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        committed = json.loads((ROOT / "data" / "processed" / name).read_text(encoding="utf-8"))
        assert built == committed, f"{name} is out of date: run python scripts/build_knowledge.py"


def test_calendar_month_boundary():
    events = json.loads((ROOT / "data" / "processed" / "calendar_events.json").read_text(encoding="utf-8"))
    assert ["FEBRUARY", 9, "Monday", "START OF SEMESTER I, TERM I LECTURES"] in events
    assert events[-1][0] == "DECEMBER"
