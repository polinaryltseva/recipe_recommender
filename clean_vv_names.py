import csv
import os
import re
import sys
from typing import List, Tuple


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()


def remove_quoted_and_bracketed(text: str) -> str:
    text = re.sub(r"[\"«»„“”‟ʻʼ’‚‘`'’]{1}[^\"«»„“”‟ʻʼ’‚‘`']*[\"«»„“”‟ʻʼ’‚‘`'’]{1}", " ", text, flags=re.UNICODE)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    return text


def expand_abbreviations(text: str) -> str:
    text = re.sub(r"\bохл\b\.?", " охлажденный ", text)
    text = re.sub(r"\bзам\b\.?", " замороженный ", text)
    text = re.sub(r"\bж/б\b", " ", text)
    text = re.sub(r"\bс/б\b", " ", text)
    text = re.sub(r"\bст/б\b", " ", text)
    text = re.sub(r"\bн/к\b", " ", text)
    return text


def remove_units_numbers(text: str) -> str:
    patterns = [
        r"\b\d+[.,]\d+\s?%",
        r"\b\d+\s?%",
        r"\b№\s?\d+\b",
        r"\b\d+\s?(г|гр|грамм|кг|мл|л)\b",
        r"\b\d+[.,]?\d*\s?(г|гр|грамм|кг|мл|л)\b",
        r"\b\d+\s?(шт|уп|упаковка|пакетик(а|ов)?|х)\b",
        r"\b\d+\s?x\s?\d+\b",
        r"\b~\b",
        r"\b\d+[.,]?\d*\b",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.UNICODE)
    return text


def truncate_at_punctuation(text: str) -> str:
    for sep in [",", "/"]:
        if sep in text:
            text = text.split(sep, 1)[0]
    return text


def first_segment_before_preposition(tokens: List[str]) -> List[str]:
    prepositions = {
        "с", "со", "в", "во", "на", "из", "для", "по", "без", "при", "под", "над", "около", "у", "к", "от"
    }
    segment: List[str] = []
    for token in tokens:
        if token in prepositions:
            break
        segment.append(token)
    return segment


def remove_geo_like_adjectives(tokens: List[str]) -> List[str]:
    cleaned: List[str] = []
    for t in tokens:
        if re.search(r"(ский|ская|ское|ские|ского|ской|ских|ским|скими)$", t):
            continue
        cleaned.append(t)
    return cleaned


STATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"заморож", re.UNICODE), "замороженный"),
    (re.compile(r"охлажден", re.UNICODE), "охлажденный"),
    (re.compile(r"копчен", re.UNICODE), "копченый"),
    (re.compile(r"копчё", re.UNICODE), "копченый"),
    (re.compile(r"маринован", re.UNICODE), "маринованный"),
    (re.compile(r"солен", re.UNICODE), "соленый"),
    (re.compile(r"солё", re.UNICODE), "соленый"),
    (re.compile(r"сушен", re.UNICODE), "сушеный"),
    (re.compile(r"жарен", re.UNICODE), "жареный"),
    (re.compile(r"печен", re.UNICODE), "печеный"),
    (re.compile(r"варен", re.UNICODE), "вареный"),
    (re.compile(r"ультрапастер", re.UNICODE), "ультрапастеризованный"),
    (re.compile(r"пастеризован", re.UNICODE), "пастеризованный"),
    (re.compile(r"безлактоз", re.UNICODE), "безлактозный"),
    (re.compile(r"цельнозернов", re.UNICODE), "цельнозерновой"),
    (re.compile(r"газирован", re.UNICODE), "газированный"),
    (re.compile(r"негазирован", re.UNICODE), "негазированный"),
    (re.compile(r"консерв", re.UNICODE), "консервированный"),
]


MARKETING_TOKENS = {
    "полезный", "полезная", "полезное",
    "био", "organic", "органик",
    "здоровье", "фитнес", "детский", "детская", "детское",
    "веган", "веганский", "постный", "классический", "классическая", "классическое",
    "премиум", "premium", "высший", "сорт",
    "отборный", "отборная", "отборное", "отборные",
    "прямого", "отжима", "свежевыжатый", "свежевыжатая", "свежевыжатое",
}


def strip_marketing(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in MARKETING_TOKENS]


def normalize_synonyms(tokens: List[str]) -> List[str]:
    return [{"горький": "темный"}.get(t, t) for t in tokens]

def drop_latin_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if not re.search(r"[a-z]", t)]


def choose_core_tokens(all_tokens: List[str]) -> List[str]:
    ёcore = first_segment_before_preposition(all_tokens)
    core = remove_geo_like_adjectives(core)
    core = strip_marketing(core)
    core = drop_latin_tokens(core)
    core = normalize_synonyms(core)
    if len(core) >= 3:
        core = core[:3]
    return core


def detect_state(text_lower: str, core_tokens: List[str]) -> str:
    core_text = " ".join(core_tokens)
    for pattern, normalized in STATE_PATTERNS:
        if pattern.search(text_lower):
            if any(pattern.search(tok) for tok in core_tokens):
                return ""
            if normalized not in core_text:
                return normalized
            break
    return ""


def clean_name(original_name: str) -> str:
    s = original_name.strip()
    if not s:
        return s
    s = s.replace("ё", "е").replace("Ё", "Е")
    s_lower = s.lower()

    s_lower = remove_quoted_and_bracketed(s_lower)
    s_lower = expand_abbreviations(s_lower)
    s_lower = truncate_at_punctuation(s_lower)
    s_lower = remove_units_numbers(s_lower)

    s_lower = re.sub(r"[^\w\-а-яА-Я ]+", " ", s_lower, flags=re.UNICODE)
    s_lower = s_lower.replace("_", " ")
    s_lower = normalize_whitespace(s_lower)

    tokens = []
    for tok in s_lower.split():
        if "-" in tok and not re.match(r"^\-+$", tok):
            tokens.extend([p for p in tok.split("-") if p])
        else:
            tokens.append(tok)

    if not tokens:
        return original_name.strip()

    core_tokens = choose_core_tokens(tokens)
    if not core_tokens:
        core_tokens = tokens[:2]

    state_word = detect_state(s_lower, core_tokens)

    result_tokens = core_tokens[:]
    if state_word and state_word not in result_tokens:
        result_tokens.append(state_word)

    result = normalize_whitespace(" ".join(result_tokens))
    if not result:
        result = tokens[0]

    if result:
        result = result[0].upper() + result[1:]
    return result


def read_names_from_csv(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                cell = row[0].strip()
                if not cell:
                    continue
                if cell.isdigit() and not names:
                    continue
                names.append(cell)
            else:
                names.append(", ".join([c.strip() for c in row if c.strip()]))
    return names


def write_mapping_to_csv(pairs: List[Tuple[str, str]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_name", "cleaned_name"])
        for original, cleaned in pairs:
            writer.writerow([original, cleaned])