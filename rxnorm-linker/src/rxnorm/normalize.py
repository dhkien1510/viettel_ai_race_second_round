"""Turn a raw drug-mention span (English/Vietnamese, free text) into the
pieces we need for matching against RxNorm: ingredient/form tokens plus
any explicit strength(s).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Route-of-administration and frequency shorthand that shows up glued to the
# drug name in clinical notes but never appears inside an RxNorm STR, so it
# only adds noise to token-overlap scoring. Vietnamese equivalents included
# since NER spans in this dataset mix both languages.
_STOPWORDS = {
    "po", "iv", "im", "sc", "sq", "subq", "subcutaneous", "top", "topical",
    "pr", "gtt", "sl", "sublingual", "ud", "prn", "stat", "ac", "pc", "hs",
    "daily", "qd", "od", "bid", "tid", "qid", "qam", "qpm", "qhs", "qod",
    "weekly", "monthly", "hourly", "x1", "x2", "x3",
    "uong", "tiem", "truyen", "ngay", "lan", "sang", "toi", "truoc", "sau",
    "an", "moi", "hom", "nay", "duong", "mieng", "tai", "cho", "va", "voi",
    # bare unit words with no attached number (the number+unit combo is
    "dung", "lieu", "duy", "nhat", "gram", "vien", "ong", "goi", "chai", "lo", "tinh", "mach", "tiem",
    # already consumed by _STRENGTH_RE above; a leftover unit word by
    # itself -- e.g. "guaifenesin ml po q6h" giving dose-per-volume without
    # ever stating the dose -- carries no matchable information)
    "ml", "mg", "mcg", "g", "l", "tab", "tabs", "cap", "caps", "unt", "unit", "units",
}

# "q6h", "q4h", "q12h", "q8h", ... -- frequency shorthand that has digits
# baked in, so it must be stripped from the raw text *before* word-level
# tokenization: _WORD_RE only matches letter runs and would otherwise split
# "q6h:prn" into stray single-letter tokens "q" and "h" plus "prn".
_FREQ_INLINE_RE = re.compile(r"\bq\d+\s*(?:h|hr|hrs|d|day|days|min|mins)\b", re.IGNORECASE)

# unit spelling -> canonical RxNorm-style unit. Includes the compound
# concentration units (MG/ML, UNT/ML, ...) the regex below can match --
# without an entry here they'd silently be dropped as "unrecognized" and
# every liquid/concentration-dosed strength (e.g. "100000 UNT/ML") would
# never be extracted at all, not just for this query but for every RxNorm
# entry indexed in build_index.py too (it reuses this same parser).
_UNIT_MAP = {
    "mg": "MG", "milligram": "MG", "milligrams": "MG",
    "mcg": "MCG", "microgram": "MCG", "micrograms": "MCG", "ug": "MCG",
    "g": "G", "gram": "G", "grams": "G",
    "ml": "ML", "milliliter": "ML", "milliliters": "ML",
    "l": "L", "liter": "L", "liters": "L",
    "unt": "UNT", "unit": "UNT", "units": "UNT", "iu": "UNT",
    "meq": "MEQ", "mmol": "MMOL", "%": "%",
    "mg/ml": "MG/ML", "mcg/ml": "MCG/ML", "unt/ml": "UNT/ML", "meq/ml": "MEQ/ML",
}

# Bare volume, as opposed to a concentration (MG/ML, UNT/ML, ...) -- see the
# comment in parse_span for why these don't count as a real strength. "G"
# (grams) does NOT belong here -- unlike a volume to administer, a gram
# figure ("Ceftriaxone 1 gram") IS the dose itself, just in a different mass
# unit than RxNorm's own MG convention (see the G->MG conversion below).
_VOLUME_ONLY_UNITS = {"ML", "L"}

_STRENGTH_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)"
    r"(?:\s*-\s*(\d+(?:[.,]\d+)?))?"
    r"\s*(mg/ml|mcg/ml|unt/ml|meq/ml|mg|mcg|microgram|micrograms|ug|g|gram|"
    r"grams|ml|milliliter|milliliters|l|liter|liters|unt|units?|iu|%|meq|mmol)\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-zA-ZÀ-ỹ]+")

# Words that legitimately pad out an RxNorm STR beyond bare ingredient and
# must never count against a candidate just because the query span omitted
# them -- pure route/packaging filler, not a distinct real-world choice.
# (No bare unit words like "ml"/"mg" here -- those never survive to this
# point at all, since _STOPWORDS strips them out during tokenization.)
NEUTRAL_TOKENS = {
    "oral", "tablet", "tablets", "drug", "product", "per",
    "base", "usp", "as",
}

# Salt/ester qualifiers: same active ingredient, but a genuinely different
# formulation choice (doxycycline hyclate vs. monohydrate vs. anhydrous are
# not interchangeable prescriptions). Scored like FORM_VARIANT_TOKENS below
# -- a small penalty when the candidate has one the query never mentioned,
# so a bare ingredient query prefers the plain IN entry over tied PIN salt
# variants instead of returning all of them.
SALT_VARIANT_TOKENS = {
    "besylate", "hydrochloride", "hcl", "sodium", "potassium", "calcium",
    "magnesium", "sulfate", "succinate", "tartrate", "bitartrate", "maleate",
    "mesylate", "citrate", "acetate", "hydrobromide", "fumarate", "phosphate",
    "nitrate", "oxide", "chloride", "gluconate", "lactate", "pamoate",
    "palmitate", "stearate", "benzoate", "camsylate", "besilate", "dihydrate",
    "monohydrate", "anhydrous",
}

# Dose-form words that name a *specific alternative* to the plain default
# ("Oral Tablet"). When the query doesn't ask for one of these, an entry
# that has it anyway (e.g. "Chewable Tablet", "Disintegrating Oral Tablet")
# should rank slightly below the plain form -- these are real, mutually
# exclusive product choices, not filler. If the query *does* mention one
# (e.g. "...oral suspension..." lands in ingredient_tokens), it matches the
# query token directly and this list never even gets consulted for it.
FORM_VARIANT_TOKENS = {
    "capsule", "capsules", "solution", "suspension", "injectable", "injection",
    "extended", "release", "delayed", "chewable", "disintegrating",
    "effervescent", "topical", "cream", "ointment", "patch", "inhalant",
    "inhalation", "nasal", "ophthalmic", "otic", "rectal", "vaginal",
    "subcutaneous", "intramuscular", "intravenous", "prefilled", "syringe",
    "pen", "implant", "kit", "pack", "hr", "film", "coated", "gel", "foam",
    "spray", "lotion", "powder", "granule", "granules", "syrup", "elixir",
    "lozenge", "buccal", "transdermal",
}

# Colloquial/common ingredient names that don't literally appear in RxNorm's
# own STR vocabulary, so exact/fuzzy token matching would otherwise miss
# them entirely. Extend as real misses turn up -- keep it small and verified,
# not a guessed brand-name dump (RxNorm already carries real brand names as
# BN entries, which match token-for-token without needing an alias).
INGREDIENT_ALIASES = {
    "senna": "sennosides",
    # "Coumadin" (the brand) is verifiably absent from this RxNorm
    # Prescribable-content snapshot entirely -- confirmed via direct grep,
    # not a guess -- likely dropped after the branded product was
    # discontinued. It's common enough in real notes (appears in this
    # dataset's own input/27.txt) that falling through to embedding and
    # risking confusion with the unrelated chemical "coumarin" isn't worth
    # it when the correct generic mapping is unambiguous.
    "coumadin": "warfarin",
    # Singularize common plural forms that appear in clinical notes
    # "nitrates": "nitrate",  # DISABLED: causes EXPAND when GT=empty
    # "nsaids": "nsaid",  # DISABLED: causes EXPAND when GT=empty
    # N-acetylcysteine common abbreviation. NOTE: was previously aliased to
    # "neuac", which is WRONG -- Neuac (rxcui 1540242) is a benzoyl
    # peroxide/clindamycin acne gel brand, not N-acetylcysteine. Confirmed by
    # querying the built index directly; "acetylcysteine" itself is RxNorm IN
    # rxcui 197 and matches lexically without needing an alias detour.
    "nac": "acetylcysteine",
    "dilaudid": "hydromorphone",
    "bactrim": "sulfamethoxazole trimethoprim",
    "cotrimoxazol": "sulfamethoxazole trimethoprim",
    "cotrimoxazole": "sulfamethoxazole trimethoprim",
    "suboxone": "buprenorphine naloxone",
    "augmentin": "amoxicillin clavulanate",
    "simenic": "simethicone",
    "aquima": "aluminum hydroxide magnesium hydroxide",
    "pimperam": "metoclopramide",
    "vastarel": "trimetazidine",
    "berlthyrox": "levothyroxine",
    "t4": "levothyroxine",
    "nhom": "aluminum",
    "magie": "magnesium",
    "hydroxid": "hydroxide",
    "trimetazidin": "trimetazidine",
    "simethicon": "simethicone",
    "alverin": "alverine",
    "sat": "iron",
    "thiamin": "thiamine",
    "pyridoxin": "pyridoxine",
    "riboflavin": "riboflavin",
    "cyanocobalamin": "cyanocobalamin",
    "cobalamin": "cyanocobalamin",
    "axit": "acid",
    "acid": "acid",
    "b12": "vitamin b",
    "vitamin b12": "vitamin b",
    "vitamin c": "ascorbic acid",
    "ns": "sodium chloride",
    "normal saline": "sodium chloride",
    "asa": "aspirin",
    "ntg": "nitroglycerin",
    "vanco": "vancomycin",
    "omez": "omeprazole",
    "medrol": "methylprednisolone",
    "zestril": "lisinopril",
    "tylenol": "acetaminophen",
    "pimperan": "metoclopramide",
    "clavulanic": "clavulanate",
}

# Generic drug-CLASS terms (not a specific ingredient/product) that clinical
# notes mention without naming an actual drug ("pt on NSAIDs", "held
# nitrates"). These have zero direct token match in RxNorm's own vocabulary
# (RxNorm indexes specific ingredients/products, not class names), so they
# fall through every lexical tier straight to embedding fallback -- where the
# bi-encoder ends up matching surface/spelling overlap instead of real
# pharmacology. Confirmed on real submission output: "NSAID"/"NSAIDs" ->
# brand names "Ansaid"/"Pennsaid" (matched on the substring "nsaid", not
# clinical meaning) and unrelated "indigotindisulfonic acid"; "nitrates" ->
# "nitric acid"/"nitrate ion"/"Nutrilyte", none an actual nitrate drug. Ground
# truth expects empty for a bare class mention, so short-circuit before tier 3
# ever runs instead of guessing. Extend as more confirmed misses turn up --
# keep it small and verified, same policy as INGREDIENT_ALIASES above.
NON_LINKABLE_CLASS_TERMS = {
    "nsaid", "nsaids", "nitrates",
}

# Dose-form abbreviations that hint at TTY/form but aren't real ingredient
# tokens; kept separately so they can nudge form matching without diluting
# the ingredient-name comparison.
_FORM_HINTS = {
    "xl": "extended release", "er": "extended release", "sr": "extended release",
    "cr": "extended release", "la": "extended release",
}


def _strip_accents(s: str) -> str:
    import unicodedata

    # đ/Đ has no NFD decomposition (it's an atomic Latin letter, not a base
    # letter + combining mark), so plain NFD-strip leaves it untouched --
    # "đường" -> "đuong" instead of "duong", silently missing the _STOPWORDS
    # entries that assume the fully-stripped ASCII spelling.
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


@dataclass
class ParsedSpan:
    raw: str
    ingredient_tokens: tuple[str, ...] = field(default_factory=tuple)
    form_hints: tuple[str, ...] = field(default_factory=tuple)
    strengths: tuple[tuple[float, str], ...] = field(default_factory=tuple)  # (value, unit)

    @property
    def all_tokens(self) -> tuple[str, ...]:
        return self.ingredient_tokens + tuple(h for hint in self.form_hints for h in hint.split())


def extract_context_info(entity: str, context: str) -> list[str]:
    if not context:
        return []
    
    entity_normalized = _strip_accents(entity.lower().strip())
    context_normalized = _strip_accents(context.lower())
    
    entity_escaped = re.escape(entity_normalized)
    
    # 1. Colon pattern: e.g. "Aquima: nhôm hydroxid, magie hydroxid..."
    # 2. Parentheses pattern: e.g. "Vastarel (trimetazidin)"
    pattern_colon = re.compile(rf"\b{entity_escaped}\s*:\s*([^.\n]+)", re.IGNORECASE)
    pattern_paren = re.compile(rf"\b{entity_escaped}\s*\(\s*([^)\n]+)\)", re.IGNORECASE)
    
    raw_matches = []
    for m in pattern_colon.finditer(context_normalized):
        content = m.group(1).strip()
        # Clean up some common trailing noise like " - "
        if " - " in content:
            content = content.split(" - ")[0].strip()
        raw_matches.append(content)
        
    for m in pattern_paren.finditer(context_normalized):
        content = m.group(1).strip()
        raw_matches.append(content)
        
    # Split matches into individual component strings
    components = []
    for match in raw_matches:
        # Standardize separators to |
        clean = re.sub(r"\b(va|va|và|va)\b", "|", match)
        clean = clean.replace("+", "|").replace(",", "|")
        for part in clean.split("|"):
            part_clean = part.strip()
            # Ignore placeholder mask components
            if part_clean and "*" not in part_clean:
                components.append(part_clean)
    return components


def parse_span(text: str, context: str | None = None) -> ParsedSpan:
    lowered = text.lower()

    # Apply multi-word drug replacements before tokenization
    multi_word_replacements = {
        "mucinex d": "guaifenesin pseudoephedrine",
        "vitamin 3b": "thiamine pyridoxine cyanocobalamin",
        "3b": "thiamine pyridoxine cyanocobalamin",
        "albuterolipratropium": "albuterol ipratropium",
        "vitamin b12": "vitamin b",
        "vitamin b 12": "vitamin b",
        "b12": "vitamin b",
        "b 12": "vitamin b",
        "vitamin c": "ascorbic acid",
        "normal saline": "sodium chloride",
    }
    for old, new in multi_word_replacements.items():
        lowered = re.sub(rf"\b{re.escape(old)}\b", new, lowered)

    strengths: list[tuple[float, str]] = []
    for m in _STRENGTH_RE.finditer(lowered):
        unit = _UNIT_MAP.get(m.group(3).lower())
        if unit is None:
            continue
        if unit in _VOLUME_ONLY_UNITS:
            # RxNorm always pairs a bare volume with a concentration ("5 ML
            # digoxin 0.25 MG/ML Injection") -- it never stands alone as a
            # strength descriptor. In clinical text "5 ml" almost always
            # means the volume to administer, not the drug's concentration,
            # so treating it as a matchable strength sends the query down
            # the wrong tier (chasing a specific dosed SCD that doesn't
            # exist) instead of the ingredient-level fallback that's
            # actually right when no real concentration was stated. Still
            # stripped from the text below so it doesn't pollute ingredient
            # tokens -- just excluded from what counts as "has a strength".
            continue
        low = float(m.group(1).replace(",", "."))
        if unit == "G":
            # RxNorm strengths are conventionally stored in MG, not G ("1
            # gram" never appears as a native RxNorm strength for solid/
            # injectable doses) -- convert so "Ceftriaxone 1 gram" can match
            # the "ceftriaxone 1000 MG" entry instead of comparing "1 G" to
            # "1000 MG" and failing the unit check.
            low *= 1000
            unit = "MG"
        # A range ("325-650 mg") only contributes its lower bound as the
        # target strength -- confirmed against BTC's own official example:
        # "acetaminophen 325-650 mg po q6h:prn" -> candidates ["313782"]
        # (325 MG Oral Tablet only), even though RxNorm has a separate 650 MG
        # product too. Treating both ends as a required set was guessing a
        # 2-code answer where gold wants exactly 1.
        strengths.append((low, unit))

    # drop the strength substrings so leftover word-tokenizing doesn't choke
    # on stray numbers/units.
    stripped = _STRENGTH_RE.sub(" ", lowered)
    stripped = _FREQ_INLINE_RE.sub(" ", stripped)

    words = _WORD_RE.findall(_strip_accents(stripped))
    ingredient_tokens: list[str] = []
    form_hints: list[str] = []
    for w in words:
        if w in _STOPWORDS:
            continue
        if w in _FORM_HINTS:
            form_hints.append(_FORM_HINTS[w])
            continue
        expanded = INGREDIENT_ALIASES.get(w, w)
        ingredient_tokens.extend(expanded.split())

    # Deduplicate while preserving order
    ingredient_tokens = tuple(dict.fromkeys(ingredient_tokens))
    form_hints = tuple(dict.fromkeys(form_hints))
    strengths = tuple(dict.fromkeys(strengths))

    return ParsedSpan(
        raw=text,
        ingredient_tokens=ingredient_tokens,
        form_hints=form_hints,
        strengths=strengths,
    )
