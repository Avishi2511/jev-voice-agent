"""Regex/parser candidates for free-form values (A-5).

Parsers propose candidates; Jev only picks among them and code does any comparison.
English number words are supported (zero to ninety-nine as separate words, plus
hundred / thousand / lakh / million multipliers); dates need digits for the day.
"""

import re
from dataclasses import dataclass, field

_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_DIGIT_WORDS = {w: v for w, v in _UNITS.items() if v <= 9}
_MULTIPLIERS = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 100_000,
    "lac": 100_000,
    "million": 1_000_000,
}
_SUFFIX = {"k": 1_000, "thousand": 1_000, "lakh": 100_000, "lac": 100_000, "million": 1_000_000}

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}  # fmt: skip
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))


@dataclass(frozen=True)
class Candidate:
    kind: str
    value: str  # normalised label offered to Jev as an option
    text: str  # span of the utterance it came from
    parts: dict[str, int | None] = field(default_factory=dict, hash=False, compare=False)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.lower())


# -- numbers ---------------------------------------------------------------


def number_from_words(words: list[str]) -> int | None:
    """Parse e.g. ['five', 'thousand'] -> 5000, ['two', 'lakh', 'fifty', 'thousand'] -> 250000."""
    total = current = 0
    seen = False
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _MULTIPLIERS:
            mult = _MULTIPLIERS[w]
            if mult == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * mult
                current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


def _number_word_runs(text: str) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    run: list[str] = []
    for tok in [*_tokens(text), ""]:
        if tok in _UNITS or tok in _MULTIPLIERS:
            run.append(tok)
            continue
        if run:
            n = number_from_words(run)
            if n is not None:
                runs.append((" ".join(run), n))
            run = []
    return runs


# -- amounts ---------------------------------------------------------------

_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹)?\s*(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?\s*(k|thousand|lakh|lac|million)?\b",
    re.IGNORECASE,
)


def extract_amounts(text: str) -> list[Candidate]:
    out: list[Candidate] = []
    for m in _AMOUNT_RE.finditer(text):
        n = int(m.group(1).replace(",", "")) * _SUFFIX.get((m.group(2) or "").lower(), 1)
        if n >= 100:  # skip bare small numbers such as days of the month
            out.append(Candidate("amount", str(n), m.group(0).strip(), {"amount": n}))
    if not out:
        for span, n in _number_word_runs(text):
            if n >= 100:
                out.append(Candidate("amount", str(n), span, {"amount": n}))
    return out


# -- phone numbers ---------------------------------------------------------

_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]*)?0?([6-9]\d{4}[\s-]?\d{5})(?!\d)")


def extract_phones(text: str) -> list[Candidate]:
    out = [
        Candidate("phone", re.sub(r"\D", "", m.group(1)), m.group(0).strip())
        for m in _PHONE_RE.finditer(text)
    ]
    if out:
        return out
    toks = _tokens(text)
    digits = [str(_DIGIT_WORDS[t]) for t in toks if t in _DIGIT_WORDS]
    if len(digits) == 10 and len(digits) == len(toks) and digits[0] in "6789":
        return [Candidate("phone", "".join(digits), " ".join(toks))]
    return []


# -- dates -----------------------------------------------------------------


def _year(y: str | None) -> int | None:
    if not y:
        return None
    return 2000 + int(y) if len(y) == 2 else int(y)


def _date(day: int, month: int | None, year: int | None, text: str) -> Candidate | None:
    if not 1 <= day <= 31 or (month is not None and not 1 <= month <= 12):
        return None
    label = f"{day:02d}-{month:02d}-{year or '????'}" if month else f"{day:02d}-??-{year or '????'}"
    return Candidate("date", label, text, {"day": day, "month": month, "year": year})


_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{4}|\d{2}))?(?!\d)")
_DAY_MONTH = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_RE})\b(?:\s*,?\s*(\d{{4}}))?",
    re.IGNORECASE,
)
_MONTH_DAY = re.compile(
    rf"\b({_MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s*(\d{{4}}))?", re.IGNORECASE
)
_DAY_ONLY = re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


def extract_dates(text: str) -> list[Candidate]:
    found: list[Candidate | None] = []
    for m in _NUMERIC_DATE.finditer(text):
        found.append(_date(int(m[1]), int(m[2]), _year(m[3]), m[0]))
    for m in _DAY_MONTH.finditer(text):
        found.append(_date(int(m[1]), _MONTHS[m[2].lower()], _year(m[3]), m[0]))
    for m in _MONTH_DAY.finditer(text):
        found.append(_date(int(m[2]), _MONTHS[m[1].lower()], _year(m[3]), m[0]))
    unique: dict[str, Candidate] = {}
    for c in found:
        if c is not None:
            unique.setdefault(c.value, c)
    if not unique:  # "on the 15th": day only, when nothing more specific was found
        for m in _DAY_ONLY.finditer(text):
            c = _date(int(m[1]), None, None, m[0])
            if c is not None:
                unique.setdefault(c.value, c)
    return list(unique.values())


def date_matches(a: Candidate, b: Candidate) -> bool:
    """Compare in code, part by part; unknown parts match anything."""
    return all(
        a.parts[p] is None or b.parts[p] is None or a.parts[p] == b.parts[p]
        for p in ("day", "month", "year")
    )
