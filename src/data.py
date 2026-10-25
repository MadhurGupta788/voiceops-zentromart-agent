"""Facts from products.csv / pincodes.csv when the customer names a SKU, product or pincode."""
import difflib
import re
import pandas as pd
from .config import DATASET_DIR

_P = None
_PIN = None
_TOK = {}
SKU_RE = re.compile(r"\b([A-Za-z]{3})[- ]?(\d{4})\b")
PIN_RE = re.compile(r"\b(\d{6})\b")

WORD_DIGITS = {"zero": "0", "oh": "0", "one": "1", "won": "1", "two": "2", "to": "2", "too": "2",
               "three": "3", "four": "4", "for": "4", "fore": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "ate": "8", "nine": "9"}
_DW = "|".join(sorted(WORD_DIGITS, key=len, reverse=True))
SPOKEN_SKU_RE = re.compile(rf"\b([a-z]{{3}})[\s,.-]+((?:(?:{_DW})\b|\d)[\s,.-]*){{4}}", re.I)
_TOKEN_RE = re.compile(rf"(?:{_DW})\b|\d", re.I)
STOP = {"the", "a", "an", "of", "and", "for", "with", "in", "to", "pack", "set"}


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _load():
    global _P, _PIN
    if _P is None:
        _P = pd.read_csv(DATASET_DIR / "data" / "products.csv", dtype=str).fillna("")
        _PIN = pd.read_csv(DATASET_DIR / "data" / "pincodes.csv", dtype=str).fillna("")
        _P["_base"] = _P["title"].str.split(",").str[0].str.strip().str.lower()
        for base in _P["_base"].unique():
            _TOK[base] = [t for t in _words(base) if t not in STOP]


def _row_text(row):
    return "; ".join(f"{k}={v}" for k, v in row.items() if v and not k.startswith("_"))


def sku_candidates(text):
    """SKUs written as ELE-0012 / ELE 0012, or spoken as 'ele zero zero one to'."""
    cands = [f"{a.upper()}-{b}" for a, b in SKU_RE.findall(text)]
    for m in SPOKEN_SKU_RE.finditer(text):
        toks = _TOKEN_RE.findall(m.group(0)[len(m.group(1)):].lower())
        digits = "".join(WORD_DIGITS.get(t, t) for t in toks)
        if len(digits) >= 4:
            cands.append(f"{m.group(1).upper()}-{digits[:4]}")
    return cands


def _name_matches(text):
    """Products whose name words appear in the text, tolerating small speech-recognition spelling errors."""
    qtoks = set(_words(text))
    scored = []
    for base, ptoks in _TOK.items():
        if len(ptoks) < 2:
            continue
        hit = 0
        for t in ptoks:
            if t in qtoks or (len(t) >= 5 and difflib.get_close_matches(t, qtoks, n=1, cutoff=0.8)):
                hit += 1
        if hit >= 2 and hit / len(ptoks) >= 0.75:
            scored.append((hit / len(ptoks), len(ptoks), base))
    scored.sort(reverse=True)
    return [b for _, _, b in scored[:3]]


def lookup(text):
    """Return [(evidence_key, text)], e.g. ('products.csv:ELE-0012', 'sku=ELE-0012; ...')."""
    _load()
    out, seen = [], set()
    skus = set(_P["sku"])
    for sku in sku_candidates(text):
        if sku in skus and sku not in seen:
            seen.add(sku)
            out.append((f"products.csv:{sku}", _row_text(_P[_P["sku"] == sku].iloc[0])))

    for base in _name_matches(text):
        for _, row in _P[_P["_base"] == base].head(2).iterrows():
            if row["sku"] not in seen:
                seen.add(row["sku"])
                out.append((f"products.csv:{row['sku']}", _row_text(row)))

    for pin in PIN_RE.findall(text):
        m = _PIN[_PIN["pincode"] == pin]
        if len(m):
            out.append((f"pincodes.csv:{pin}", _row_text(m.iloc[0])))
    return out[:5]
