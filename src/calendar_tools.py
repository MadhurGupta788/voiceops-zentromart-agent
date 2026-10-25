"""Business-day calendar: Monday to Saturday, skipping holidays.csv. The LLM never does date arithmetic itself."""
import re
from datetime import date, timedelta
import pandas as pd
from .config import DATASET_DIR
from .data import WORD_DIGITS, _DW

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MON = "|".join(MONTHS)
DATE_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})\b(?P<rest>.{{0,60}})", re.I)
_YEAR_RE = re.compile(rf"^[\s,]*((?:(?:{_DW})\b|\d)[\s,]*){{4}}", re.I)
_TOK = re.compile(rf"(?:{_DW})\b|\d", re.I)
_HOL = None


def _holidays():
    global _HOL
    if _HOL is None:
        h = pd.read_csv(DATASET_DIR / "data" / "holidays.csv", dtype=str).fillna("")
        _HOL = {date.fromisoformat(d.strip()) for d in h["date"] if d.strip()}
    return _HOL


def is_business_day(d):
    return d.weekday() != 6 and d not in _holidays()


def after(start, n):
    d, count = start, 0
    while count < n:
        d += timedelta(days=1)
        if is_business_day(d):
            count += 1
    return d


def fmt(d):
    return f"{d.day} {d.strftime('%B')} {d.year} ({d.strftime('%A')})"


def _year(rest):
    m = _YEAR_RE.match(rest)
    if m:
        digits = "".join(WORD_DIGITS.get(t.lower(), t) for t in _TOK.findall(m.group(0)))
        if len(digits) == 4 and digits.startswith("20"):
            return int(digits)
    return 2026  # the dataset's year


def parse_dates(text):
    out = []
    for m in DATE_RE.finditer(text):
        try:
            d = date(_year(m.group("rest")), MONTHS[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            continue
        if d not in out:
            out.append(d)
    return out[:2]


def calendar_block(text):
    dates = parse_dates(text)
    if not dates:
        return ""
    lines = []
    for s in dates:
        lines.append(f"Counting from {fmt(s)} (the start day is not counted):")
        lines += [f"  after {n} business days = {fmt(after(s, n))}" for n in range(1, 11)]
    return ("[CALENDAR] Business-day dates computed from holidays.csv "
            "(Monday to Saturday, excluding holidays)\n" + "\n".join(lines))
