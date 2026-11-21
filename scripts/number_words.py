"""Rewrites spoken amounts in Swahili and English as digits, for a voice client to run on an
utterance before it asks the model for tool calls: "kiasi laki tatu na elfu hamsini" becomes
"kiasi 350000", "two million shillings" "2000000 shillings". The model then copies a number
instead of working it out, which it got wrong (elfu ishirini as 2000).

Only phrases holding a scale word (mia, elfu, laki, milioni; hundred, thousand, million) are
rewritten, so a lone "moja" or "one" in ordinary speech stays a word. Digits inside a phrase
count as a number ("milioni 2", "elfu 50").

Swahili puts the scale first and its multiplier after it (elfu hamsini = 1000 x 50), and "na"
(and) joins the parts: milioni moja na laki tano = 1500000, elfu kumi na tano = 15000,
elfu mbili na mia tano = 2500.
"""
import re

SW_UNITS = {"sifuri": 0, "moja": 1, "mbili": 2, "tatu": 3, "nne": 4, "tano": 5, "sita": 6,
            "saba": 7, "nane": 8, "tisa": 9}
SW_TENS = {"kumi": 10, "ishirini": 20, "thelathini": 30, "arobaini": 40, "hamsini": 50,
           "sitini": 60, "sabini": 70, "themanini": 80, "tisini": 90}
SW_SCALES = {"milioni": 1_000_000, "laki": 100_000, "elfu": 1000}

EN_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
            "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_SCALES = {"thousand": 1000, "million": 1_000_000}

SW_WORDS = set(SW_UNITS) | set(SW_TENS) | set(SW_SCALES) | {"mia", "na"}
EN_WORDS = set(EN_SMALL) | set(EN_SCALES) | {"hundred", "and", "a"}


def _digits(w):
    return w.isdigit()


class _Sw:
    """Recursive descent over one span of Swahili number words."""

    def __init__(self, words):
        self.w, self.i = words, 0

    def peek(self, k=0):
        return self.w[self.i + k] if self.i + k < len(self.w) else None

    def small(self):
        """Up to 999: [mia N] [tens [na unit] | unit], or digits. None if nothing parsed."""
        v, got = 0, False
        if self.peek() == "mia":
            self.i += 1
            m = self.peek()
            if m in SW_UNITS:
                self.i += 1
                v = 100 * SW_UNITS[m]
            elif m in SW_TENS:  # "mia kumi" is rare but means 1000
                self.i += 1
                v = 100 * SW_TENS[m]
            else:
                v = 100
            got = True
            if self.peek() == "na" and (self.peek(1) in SW_TENS or self.peek(1) in SW_UNITS):
                self.i += 1
        w = self.peek()
        if w in SW_TENS:
            self.i += 1
            v += SW_TENS[w]
            got = True
            if self.peek() == "na" and self.peek(1) in SW_UNITS:
                v += SW_UNITS[self.peek(1)]
                self.i += 2
        elif w in SW_UNITS:
            self.i += 1
            v += SW_UNITS[w]
            got = True
        elif w is not None and _digits(w) and not got:
            self.i += 1
            v, got = int(w), True
        return v if got else None

    def number(self):
        total, got = 0, False
        for scale, mult in SW_SCALES.items():
            if self.peek() == scale:
                self.i += 1
                n = self.small()
                total += mult * (1 if n is None else n)
                got = True
                if self.peek() == "na" and self.peek(1) is not None:
                    self.i += 1
        rest = self.small()
        if rest is not None:
            total, got = total + rest, True
        return total if got else None


def _en(words):
    total, cur, got = 0, 0, False
    for w in words:
        if w in EN_SMALL:
            cur += EN_SMALL[w]
        elif _digits(w):
            cur += int(w)
        elif w == "a":
            cur += 1
        elif w == "hundred":
            cur = (cur or 1) * 100
        elif w in EN_SCALES:
            total += (cur or 1) * EN_SCALES[w]
            cur = 0
        elif w == "and":
            continue
        got = True
    return total + cur if got else None


def _spans(tokens, vocab, scales):
    """(start, end) of runs of number words (or digits) that hold a scale word."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i].lower() in vocab or _digits(tokens[i]):
            j = i
            while j < len(tokens) and (tokens[j].lower() in vocab or _digits(tokens[j])):
                j += 1
            # Connectors never start or end a number, nor does "a" end one.
            s, e = i, j
            while s < e and tokens[s].lower() in ("na", "and"):
                s += 1
            while e > s and tokens[e - 1].lower() in ("na", "and", "a"):
                e -= 1
            if any(t.lower() in scales for t in tokens[s:e]):
                out.append((s, e))
            i = j
        else:
            i += 1
    return out


def _sw(words):
    p = _Sw(words)
    v = p.number()
    return v if p.i == len(words) else None  # words left over: not one number


def to_digits(text):
    """`text` with each spoken amount holding a scale word written as digits."""
    for vocab, scales, parse in ((SW_WORDS, set(SW_SCALES) | {"mia"}, _sw),
                                 (EN_WORDS, set(EN_SCALES) | {"hundred"}, _en)):
        # Words at even indices, the separators between them at odd ones.
        parts = re.split(r"(\W+)", text)
        words = parts[0::2]
        for s, e in reversed(_spans(words, vocab, scales)):
            if any(sep.strip() for sep in parts[2 * s + 1:2 * e - 1:2]):
                continue  # a comma or full stop inside: not one spoken number
            value = parse([w.lower() for w in words[s:e]])
            if value is not None:
                parts[2 * s:2 * e - 1] = [str(value)]
        text = "".join(parts)
    return text
