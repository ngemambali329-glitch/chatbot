"""Answering logic for the UniZulu Engineering Chatbot (no Streamlit, no language model).

Routing, in order:
  1. Module code + a scheduling question  -> exact timetable lookup
  2. Calendar words (semester, exams...)  -> exact calendar-event search
  3. Everything else                      -> BM25 retrieval over the handbook
If nothing relevant is found the chatbot says the question is not covered.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

OUT_OF_SCOPE = (
    "That doesn't look like it's covered in the UniZulu Engineering timetable, "
    "calendar or handbook. Try rephrasing, or ask something from those documents."
)

TOKEN_RE = re.compile(r"[a-z0-9]+")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_WORDS = {
    "monday": "Monday", "mon": "Monday",
    "tuesday": "Tuesday", "tue": "Tuesday", "tues": "Tuesday",
    "wednesday": "Wednesday", "wed": "Wednesday",
    "thursday": "Thursday", "thu": "Thursday", "thur": "Thursday", "thurs": "Thursday",
    "friday": "Friday", "fri": "Friday",
    "saturday": "Saturday", "sat": "Saturday",
    "sunday": "Sunday", "sun": "Sunday",
}
MONTHS = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
    "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
]
MONTH_INDEX = {m: i for i, m in enumerate(MONTHS)}

CODE_RE = re.compile(r"\b\d[A-Z]{2,4}\d{2,3}[A-Z]?\b")
SCHEDULE_WORDS = {
    "time", "times", "when", "schedule", "scheduled", "timetable", "class", "classes",
    "lecture", "lectures", "tutorial", "tutorials", "practical", "practicals", "venue",
    "period", "periods", "slot", "slots", "meet", "meets",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stem(word: str, n: int) -> str:
    w = word.lower()
    if any(ch.isdigit() for ch in w):  # module codes such as 5eee421 stay whole
        return w
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w[:n]


@dataclass
class Answer:
    text: str
    source: str  # "timetable" | "calendar" | "handbook" | "none"
    context: list[str] = field(default_factory=list)  # retrieved text, for the LLM / "show sources"


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.timetable: list[dict] = json.loads((data_dir / "timetable_entries.json").read_text(encoding="utf-8"))
        self.calendar: list[list] = json.loads((data_dir / "calendar_events.json").read_text(encoding="utf-8"))
        raw_chunks = json.loads((data_dir / "handbook_chunks.json").read_text(encoding="utf-8"))
        self.chunks: list[str] = [c["text"] for c in raw_chunks]
        self.chunk_start: list[int] = [c["start"] for c in raw_chunks]
        self.lines: list[str] = json.loads((data_dir / "handbook_lines.json").read_text(encoding="utf-8"))

        # calendar IDF over event text (4-letter stems)
        self._cal_stems = [{stem(t, 4) for t in tokenize(e[3])} for e in self.calendar]
        df = Counter(s for stems in self._cal_stems for s in stems)
        n = len(self.calendar)
        self._cal_idf = lambda w: math.log(1 + n / (1 + df.get(stem(w, 4), 0)))
        self._cal_df = df

        # BM25 over handbook chunks (5-letter stems)
        self._bm25 = BM25Okapi([[stem(t, 5) for t in tokenize(c)] for c in self.chunks])


# --------------------------------------------------------------------------
# 1. Timetable
# --------------------------------------------------------------------------
def find_module_code(query: str) -> str | None:
    m = CODE_RE.search(query.upper())
    return m.group(0) if m else None


def find_day(query: str) -> str | None:
    for w in tokenize(query):
        if w in DAY_WORDS:
            return DAY_WORDS[w]
    return None


def is_schedule_question(query: str) -> bool:
    if find_day(query):
        return True
    return bool(set(tokenize(query)) & SCHEDULE_WORDS)


def lookup_schedule(kb: KnowledgeBase, code: str, day: str | None = None) -> list[dict]:
    code = code.upper()
    hits = [
        e for e in kb.timetable
        if e["code"] == code or (code[-1].isdigit() and e["code"].startswith(code))
    ]
    if day:
        hits = [e for e in hits if e["day"] == day]
    return hits


def format_schedule(code: str, hits: list[dict], all_hits: list[dict], day: str | None) -> str:
    if not hits:
        if not all_hits:
            return f"**{code}** was not found in the timetable."
        days = sorted({e["day"] for e in all_hits}, key=DAYS.index)
        return f"**{code}** does not appear to be scheduled on {day}. It is scheduled on: {', '.join(days)}."

    def label(e: dict) -> str:
        base = e["time"]
        if e["group"]:
            base += f" (Group {e['group']})"
        if e["code"] != code:
            base += f" [{e['code']}]"
        return base

    by_day: dict[str, list[str]] = {}
    for e in sorted(hits, key=lambda e: (DAYS.index(e["day"]), e["time"], e["code"])):
        by_day.setdefault(e["day"], []).append(label(e))
    if day:
        return f"**{code}** on {day}: {', '.join(by_day[day])}"
    lines = [f"**{code}** is scheduled as follows:"]
    for d, times in by_day.items():
        lines.append(f"- {d}: {', '.join(times)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 2. Calendar
# --------------------------------------------------------------------------
CAL_TRIGGERS = {
    "recess", "registration", "semester", "term", "exam", "exams", "examination",
    "examinations", "holiday", "holidays", "closure", "graduation", "orientation",
    "lecture", "lectures", "break", "deadline", "reopen", "reopens", "open", "opens",
    "closes", "closing", "calendar", "special", "supplementary", "aegrotat",
}
CAL_STOP = {
    "is", "the", "what", "when", "does", "do", "did", "for", "of", "on", "in", "a", "an",
    "are", "was", "were", "will", "would", "can", "could", "should", "to", "we", "you",
    "your", "our", "there", "here", "much", "many", "get", "got", "want", "shall", "have",
    "has", "had", "date", "dates", "list", "all", "every", "each", "me", "please", "show",
    "tell", "about", "does", "start", "starts", "begin", "begins", "end", "ends", "finish",
    "finishes", "commence", "commences", "university", "which", "how", "long", "day",
}
CAL_SYNONYMS = [
    (re.compile(r"\bfirst semester\b"), "semester i"),
    (re.compile(r"\bsecond semester\b"), "semester ii"),
    (re.compile(r"\bsemester 1\b"), "semester i"),
    (re.compile(r"\bsemester 2\b"), "semester ii"),
    (re.compile(r"\bterm 1\b"), "term i"),
    (re.compile(r"\bterm 2\b"), "term ii"),
    (re.compile(r"\bterm 3\b"), "term iii"),
    (re.compile(r"\bterm 4\b"), "term iv"),
    (re.compile(r"\bholidays?\b"), "public holiday"),
    (re.compile(r"\bbreak\b"), "recess"),
    (re.compile(r"\breg\b"), "registration"),
    (re.compile(r"\bexams?\b"), "examinations"),
]
START_WORDS = {"start", "starts", "begin", "begins", "commence", "commences", "open", "opens", "opening"}
END_WORDS = {"end", "ends", "finish", "finishes", "close", "closes", "closing"}


def is_calendar_query(query: str) -> bool:
    return bool(set(tokenize(query)) & CAL_TRIGGERS)


def _normalise_calendar_query(query: str) -> str:
    q = query.lower()
    for pattern, repl in CAL_SYNONYMS:
        q = pattern.sub(repl, q)
    return q


def search_calendar(kb: KnowledgeBase, query: str, max_hits: int = 3) -> list[list]:
    q = _normalise_calendar_query(query)
    words = tokenize(q)
    wordset = set(words)
    intent = "start" if wordset & START_WORDS else "end" if wordset & END_WORDS else None
    month = next((m for m in MONTHS if m.lower() in wordset), None)
    content = [w for w in words if w not in CAL_STOP and w not in DAY_WORDS and w.upper() not in MONTH_INDEX]
    present = [w for w in content if kb._cal_df.get(stem(w, 4), 0) >= 1]
    if not present:
        return []

    scored = []
    for event, stems in zip(kb.calendar, kb._cal_stems):
        if month and event[0] != month:
            continue
        matched = [w for w in present if stem(w, 4) in stems]
        if not matched:
            continue
        # the rarest query word must appear in the event
        anchor = max(present, key=kb._cal_idf)
        if stem(anchor, 4) not in stems:
            continue
        text_l = event[3].lower()
        score = sum(kb._cal_idf(w) for w in matched) / math.sqrt(len(stems))
        if intent == "start" and text_l.startswith(("start of", "opening", "university re-opens")):
            score *= 1.5
        if intent == "end" and text_l.startswith(("end of", "closure", "last day")):
            score *= 1.5
        scored.append((score, event))
    scored.sort(key=lambda s: (-s[0], MONTH_INDEX[s[1][0]], s[1][1]))
    if not scored:
        return []
    best = scored[0][0]
    return [e for sc, e in scored if sc >= 0.75 * best][:max_hits]


def _fmt_date(event: list) -> str:
    month, day, dow, _ = event
    return f"{dow} {day} {month.title()} 2026"


def format_calendar(kb: KnowledgeBase, hits: list[list]) -> str | None:
    if not hits:
        return None
    lines = []
    for ev in sorted(hits, key=lambda e: (MONTH_INDEX[e[0]], e[1])):
        text = ev[3]
        line = f"- **{_fmt_date(ev)}**: {text}"
        m = re.match(r"^start of (.*)$", text, re.IGNORECASE)
        if m:  # add the matching end date when the calendar has one
            key = m.group(1).strip().lower()
            ends = [e for e in kb.calendar if e[3].lower() == f"end of {key}"]
            if len(ends) == 1:
                line += f" (ends {_fmt_date(ends[0])})"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 3. Handbook retrieval (BM25)
# --------------------------------------------------------------------------
HB_STOP = {
    "what", "is", "the", "a", "an", "are", "of", "for", "to", "in", "on", "and", "or", "does",
    "do", "did", "how", "when", "which", "who", "where", "can", "i", "my", "me", "please",
    "tell", "about", "need", "needed", "get", "have", "has", "there",
}
BM25_THRESHOLD = 2.0
MIN_COVERAGE = 0.6  # at least 60% of the question's content words must appear in the chunk


HB_SYNONYMS = [
    (re.compile(r"\bmaths?\b", re.I), "mathematics"),
    (re.compile(r"\bmatric\b", re.I), "matriculation"),
    (re.compile(r"\bco-?ordinators?\b", re.I), "co-ordinator"),
    (re.compile(r"\bphysics\b", re.I), "physical"),
]


def _query_stems(query: str) -> list[str]:
    """Content stems of a question. Module codes are reduced to the handbook form
    (5EEE421L -> 5eee421) because the handbook lists modules without the L/T/P suffix."""
    for pattern, repl in HB_SYNONYMS:
        query = pattern.sub(repl, query)
    query = CODE_RE.sub(lambda m: m.group(0)[:-1] if m.group(0)[-1].isalpha() else m.group(0), query.upper())
    return [stem(w, 5) for w in tokenize(query) if w not in HB_STOP]


def retrieve(kb: KnowledgeBase, query: str, top_k: int = 3) -> list[tuple[float, str, float, int]]:
    """Return (score, chunk, coverage, chunk_index) best first. Coverage = share of the question's
    content words that the chunk contains; low coverage means an unrelated match."""
    q_stems = _query_stems(query)
    if not q_stems:
        return []
    unique = set(q_stems)
    scores = kb._bm25.get_scores(q_stems)
    boosted = []
    for i, base in enumerate(scores):
        if base <= 0:
            continue
        heading = {stem(t, 5) for t in tokenize(kb.chunks[i][:80])}
        chunk_stems = {stem(t, 5) for t in tokenize(kb.chunks[i])}
        boost = 2.0 * len(heading & unique)  # a chunk that starts with the topic wins
        coverage = len(chunk_stems & unique) / len(unique)
        boosted.append((base + boost, i, coverage))
    boosted.sort(reverse=True)

    results, taken = [], []
    for score, i, coverage in boosted:
        if any(abs(i - j) <= 1 for j in taken):  # skip near-duplicate overlapping chunks
            continue
        results.append((score, kb.chunks[i], coverage, i))
        taken.append(i)
        if len(results) >= top_k:
            break
    return results


def _trim(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(": "))
    return (cut[: end + 1] if end > 200 else cut.rsplit(" ", 1)[0] + " ...").strip()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def answer_query(kb: KnowledgeBase, query: str) -> Answer:
    query = query.strip()
    if not query:
        return Answer("Please type a question.", "none")

    code = find_module_code(query)
    if code and is_schedule_question(query):
        day = find_day(query)
        all_hits = lookup_schedule(kb, code)
        hits = lookup_schedule(kb, code, day) if day else all_hits
        return Answer(format_schedule(code, hits, all_hits, day), "timetable")

    if is_calendar_query(query):
        hits = search_calendar(kb, query)
        text = format_calendar(kb, hits)
        if text:
            return Answer(text, "calendar", [f"{_fmt_date(h)}: {h[3]}" for h in hits])

    results = retrieve(kb, query)
    if not results or results[0][0] < BM25_THRESHOLD or results[0][2] < MIN_COVERAGE:
        return Answer(OUT_OF_SCOPE, "none")
    context = [c for _, c, _, _ in results]
    start = kb.chunk_start[results[0][3]]
    section = " ".join(kb.lines[start:start + 16])  # the matching passage and the lines after it
    nxt = section.find(" Title ", 10)  # module records start with "Title": stop at the next one
    if nxt != -1:
        section = section[:nxt]
    return Answer(f"According to the handbook: {_trim(section, 900)}", "handbook", context)
