"""Regex/parser candidates for free-form values (A-5).

Parsers propose candidates; Jev only picks among them and code does any comparison.
Hindi number words are supported for 0-31, round tens, and sau/hazaar/lakh multipliers.
"""

import re
from dataclasses import dataclass, field

_UNITS: dict[str, int] = {
    "zero": 0, "shunya": 0,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
    "chhe": 6, "chhah": 6, "che": 6, "saat": 7, "sat": 7, "aath": 8, "ath": 8,
    "nau": 9, "no": 9, "das": 10, "gyarah": 11, "barah": 12, "terah": 13,
    "chaudah": 14, "pandrah": 15, "solah": 16, "satrah": 17, "atharah": 18,
    "unnis": 19, "bees": 20, "ikkis": 21, "baees": 22, "baais": 22, "teis": 23,
    "chaubis": 24, "pachchis": 25, "pachis": 25, "chhabbis": 26, "sattais": 27,
    "atthais": 28, "untis": 29, "tees": 30, "tis": 30, "iktis": 31, "ikattis": 31,
    "chalis": 40, "pachas": 50, "saath": 60, "sattar": 70, "assi": 80, "nabbe": 90,
}  # fmt: skip
_DIGIT_WORDS = {w: v for w, v in _UNITS.items() if v <= 9}
_MULTIPLIERS = {"sau": 100, "hazaar": 1_000, "hazar": 1_000, "lakh": 100_000, "lac": 100_000}

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "janwari": 1, "feb": 2, "february": 2, "farwari": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5, "mai": 5, "jun": 6,
    "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "agast": 8,
    "sep": 9, "sept": 9, "september": 9, "sitambar": 9, "oct": 10, "october": 10,
    "aktubar": 10, "nov": 11, "november": 11, "navambar": 11, "dec": 12,
    "december": 12, "disambar": 12,
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


def hindi_number(words: list[str]) -> int | None:
    """Parse e.g. ['paanch', 'hazaar'] -> 5000, ['do', 'lakh', 'pachas', 'hazaar'] -> 250000."""
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
            n = hindi_number(run)
            if n is not None:
                runs.append((" ".join(run), n))
            run = []
    return runs


# -- amounts ---------------------------------------------------------------

_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹)?\s*(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?\s*(k|hazaar|hazar|lakh|lac)?",
    re.IGNORECASE,
)


def extract_amounts(text: str) -> list[Candidate]:
    out: list[Candidate] = []
    for m in _AMOUNT_RE.finditer(text):
        n = int(m.group(1).replace(",", ""))
        suffix = (m.group(2) or "").lower()
        n *= {"k": 1_000, "hazaar": 1_000, "hazar": 1_000, "lakh": 100_000, "lac": 100_000}.get(
            suffix, 1
        )
        if n >= 100:  # skip bare small numbers such as dates
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
_DAY_WORD = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s*(?:tareekh|tarikh|date)\b", re.IGNORECASE
)


def extract_dates(text: str) -> list[Candidate]:
    found: list[Candidate | None] = []
    for m in _NUMERIC_DATE.finditer(text):
        found.append(_date(int(m[1]), int(m[2]), _year(m[3]), m[0]))
    for m in _DAY_MONTH.finditer(text):
        found.append(_date(int(m[1]), _MONTHS[m[2].lower()], _year(m[3]), m[0]))
    for m in _MONTH_DAY.finditer(text):
        found.append(_date(int(m[2]), _MONTHS[m[1].lower()], _year(m[3]), m[0]))
    for m in _DAY_WORD.finditer(text):
        found.append(_date(int(m[1]), None, None, m[0]))
    # "pandrah tareekh": number word directly before the date marker
    toks = _tokens(text)
    for i, tok in enumerate(toks[:-1]):
        if tok in _UNITS and toks[i + 1] in {"tareekh", "tarikh", "date"}:
            found.append(_date(_UNITS[tok], None, None, f"{tok} {toks[i + 1]}"))
    unique: dict[str, Candidate] = {}
    for c in found:
        if c is not None:
            unique.setdefault(c.value, c)
    return list(unique.values())


def date_matches(a: Candidate, b: Candidate) -> bool:
    """Compare in code, part by part; unknown parts match anything."""
    return all(
        a.parts[p] is None or b.parts[p] is None or a.parts[p] == b.parts[p]
        for p in ("day", "month", "year")
    )
