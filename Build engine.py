"""Build the searchable knowledge files from the three university documents.

Inputs  (data/raw/):
    timetable.pdf   2026 first-semester lecture timetable (a colour-coded grid)
    calendar.txt    2026 UNIZULU General Calendar (text extracted from the PDF)
    handbook.txt    2024 Engineering Department handbook (text extracted from the PDF)

Outputs (data/processed/):
    timetable_entries.json   [{"day", "time", "code", "group"}, ...]
    calendar_events.json     [[month, day, weekday, text], ...]
    handbook_chunks.json     [{"start": line index, "text": block of handbook lines}, ...]
    handbook_lines.json      ["cleaned handbook line", ...]

Run:  python scripts/build_knowledge.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Timetable
# --------------------------------------------------------------------------
CODE_RE = re.compile(r"\d[A-Z]{2,4}\d{2,3}[A-Z]?(?:_[A-Z])?(?:/[A-Z0-9]+)?")
TIME_RE = re.compile(r"^(\d\d)h(\d\d)-(\d\d)h(\d\d)$")
DAY_LABELS = {
    "MON": "Monday",
    "TUES": "Tuesday",
    "WED": "Wednesday",
    "THURS": "Thursday",
    "FRI": "Friday",
    "SAT": "Saturday",
}


def expand_code(token: str) -> list[tuple[str, str]]:
    """Turn a raw timetable token into (code, group) pairs.

    '4MTH271L_B'   -> [('4MTH271L', 'B')]
    '5EEE361P/T'   -> [('5EEE361P', ''), ('5EEE361T', '')]
    '5EEE311L/5EEE351' -> [('5EEE311L', ''), ('5EEE351', '')]
    """
    parts = token.split("/")
    first = parts[0]
    group = ""
    if "_" in first:
        first, group = first.split("_", 1)
    out = [(first, group)]
    for extra in parts[1:]:
        if re.fullmatch(r"[A-Z]", extra):
            out.append((re.sub(r"[A-Z]$", "", first) + extra, group))
        elif re.fullmatch(r"\d[A-Z]{2,4}\d{2,3}[A-Z]?", extra):
            out.append((extra, group))
    return out


def parse_timetable(pdf_path: Path) -> list[dict]:
    import pdfplumber

    entries: set[tuple[str, str, str, str]] = set()
    current_day: str | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=2)

            # Row labels (one per time slot) and day labels sit in the left margin.
            rows: list[tuple[float, str, str | None]] = []  # (y_centre, time, day_label)
            day_at_top: dict[float, str] = {}
            for w in words:
                if w["x0"] < 45 and w["text"] in DAY_LABELS:
                    day_at_top[round(w["top"], 0)] = DAY_LABELS[w["text"]]
            for w in sorted(words, key=lambda w: w["top"]):
                m = TIME_RE.match(w["text"])
                if m and w["x0"] < 60:
                    time = f"{m[1]}:{m[2]}-{m[3]}:{m[4]}"
                    yc = (w["top"] + w["bottom"]) / 2
                    label = day_at_top.get(round(w["top"], 0))
                    if label:
                        current_day = label
                    rows.append((yc, time, current_day))
            if not rows:
                continue

            first_y = rows[0][0] - 8
            last_y = rows[-1][0] + 8

            # Collect module-code words (split merged tokens such as '4MTH271L_B4MTH171L').
            found = []
            for w in words:
                if w["x0"] < 60 or not (first_y <= w["top"] <= last_y):
                    continue
                matches = list(CODE_RE.finditer(w["text"]))
                if not matches:
                    continue
                width = w["x1"] - w["x0"]
                for m in matches:
                    x0 = w["x0"] + width * m.start() / len(w["text"])
                    x1 = w["x0"] + width * m.end() / len(w["text"])
                    group = ""
                    # a lone A/B right after the code is its group letter
                    for g in words:
                        if (
                            re.fullmatch(r"[A-Z]", g["text"])
                            and abs(g["top"] - w["top"]) < 2.5
                            and 0 <= g["x0"] - x1 <= 8
                        ):
                            group = g["text"]
                            break
                    found.append((m.group(0), group, (x0 + x1) / 2, (w["top"] + w["bottom"]) / 2))

            # Assign each printed code to a time row: nearest row centre, and a code
            # printed twice in the same column fills two different rows.
            used: set[tuple[str, str, int, int]] = set()
            for token, group, xc, yc in sorted(found, key=lambda f: f[3]):
                order = sorted(range(len(rows)), key=lambda i: abs(rows[i][0] - yc))
                chosen = order[0]
                for i in order[:3]:
                    if (token, group, round(xc / 25), i) not in used:
                        chosen = i
                        break
                used.add((token, group, round(xc / 25), chosen))
                _, time, day = rows[chosen]
                if not day:
                    continue
                for code, grp in expand_code(token):
                    entries.add((day, time, code, grp or group))

    day_order = {d: i for i, d in enumerate(DAY_LABELS.values())}
    ordered = sorted(entries, key=lambda e: (day_order[e[0]], e[1], e[2], e[3]))
    return [{"day": d, "time": t, "code": c, "group": g} for d, t, c, g in ordered]


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
MONTHS = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
    "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
]
DOW = {
    "Mon": "Monday", "Tues": "Tuesday", "Wed": "Wednesday", "Thurs": "Thursday",
    "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday",
}
DAY_LINE = re.compile(r"^(Mon|Tues|Wed|Thurs|Fri|Sat|Sun)\s+(\d{1,2})\b\s*(.*)$")
ITEM_SPLIT = re.compile(r"\(\s*(?:i{1,3}|iv|v|vi{0,3}|ix|x)\s*\)", re.IGNORECASE)
NOISE = re.compile(r"^(2026|GENERAL CALENDAR|\d{1,3}|Meeting time with \*.*)$")


def parse_calendar(text_path: Path) -> list[list]:
    lines = text_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Start at the first month heading that is followed by a day line (skips the contents page).
    start = next(
        i for i, ln in enumerate(lines)
        if ln.strip() == "JANUARY" and DAY_LINE.match(lines[i + 1].strip())
    )

    events: list[list] = []
    month_idx = -1
    last_day: dict | None = None  # {"month","day","dow","buf":[...]}
    first_after_heading = False

    def flush():
        nonlocal last_day
        if not last_day:
            return
        raw = " ".join(last_day["buf"]).strip()
        raw = re.sub(r"\s+", " ", raw)
        if raw:
            parts = [p.strip(" .;") for p in ITEM_SPLIT.split(raw)]
            for p in parts:
                if p:
                    events.append([last_day["month"], last_day["day"], last_day["dow"], p])
        last_day = None

    for line in lines[start:]:
        s = line.strip()
        if s == "SENATE" and month_idx == 11:
            break
        if not s or NOISE.match(s):
            continue
        if s in MONTHS:
            flush()
            month_idx = MONTHS.index(s)
            first_after_heading = True
            continue
        m = DAY_LINE.match(s)
        if m:
            flush()
            day = int(m.group(2))
            month = month_idx
            # the grid starts on a Monday, so e.g. "Sat 31" under FEBRUARY is still January
            if first_after_heading and day >= 28 and month > 0:
                month -= 1
            first_after_heading = False
            last_day = {
                "month": MONTHS[month],
                "day": day,
                "dow": DOW[m.group(1)],
                "buf": [m.group(3)],
            }
            continue
        if last_day is not None:
            last_day["buf"].append(s)
    flush()
    return events


# --------------------------------------------------------------------------
# Handbook
# --------------------------------------------------------------------------
def read_handbook_lines(text_path: Path) -> list[str]:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    return [ln for ln in lines if ln]


def build_handbook_chunks(lines: list[str], window: int = 8, stride: int = 4, max_chars: int = 900) -> list[dict]:
    """Overlapping blocks of lines. 'start' is the index of the block's first line."""
    chunks = []
    for i in range(0, len(lines), stride):
        block = " ".join(lines[i:i + window])
        if len(block) > max_chars:
            block = block[:max_chars].rsplit(" ", 1)[0]
        if len(block) > 40:
            chunks.append({"start": i, "text": block})
    return chunks


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    timetable = parse_timetable(args.raw / "timetable.pdf")
    calendar = parse_calendar(args.raw / "calendar.txt")
    lines = read_handbook_lines(args.raw / "handbook.txt")
    chunks = build_handbook_chunks(lines)

    if not timetable or not calendar or not chunks:
        print("ERROR: one of the documents produced no data.", file=sys.stderr)
        return 1

    (args.out / "timetable_entries.json").write_text(json.dumps(timetable, indent=1), encoding="utf-8")
    (args.out / "calendar_events.json").write_text(json.dumps(calendar, indent=1), encoding="utf-8")
    (args.out / "handbook_chunks.json").write_text(json.dumps(chunks, indent=1), encoding="utf-8")

    (args.out / "handbook_lines.json").write_text(json.dumps(lines), encoding="utf-8")

    print(f"Timetable entries : {len(timetable)}")
    print(f"Calendar events   : {len(calendar)}")
    print(f"Handbook chunks   : {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
