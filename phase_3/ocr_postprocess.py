#!/usr/bin/env python3
"""Shared cleanup for Telugu OCR output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import unicodedata


TELUGU_START = "\u0c00"
TELUGU_END = "\u0c7f"
TELUGU_JOINERS = {"\u200c", "\u200d"}


@dataclass(frozen=True)
class OCRText:
    """Cleaned OCR text and the information needed to review weak results."""

    text: str
    suspicious_output: bool
    suspicious_reason: str
    telugu_character_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_text(text: str | None) -> str:
    """Normalize Unicode to NFC and collapse repeated whitespace."""
    return unicodedata.normalize("NFC", " ".join((text or "").split()))


def _is_telugu(character: str) -> bool:
    return TELUGU_START <= character <= TELUGU_END


def clean_telugu_text(text: str | None) -> str:
    """Keep Telugu text while dropping Latin text and model commentary markup.

    Hyphens are retained only when they occur between Telugu characters. This
    prevents punctuation left behind by removed English commentary from
    appearing in the final response.
    """
    normalized = normalize_text(text)
    kept = []
    for index, character in enumerate(normalized):
        if (
            _is_telugu(character)
            or character in TELUGU_JOINERS
            or character.isspace()
        ):
            kept.append(character)
        elif (
            character == "-"
            and index > 0
            and index + 1 < len(normalized)
            and _is_telugu(normalized[index - 1])
            and _is_telugu(normalized[index + 1])
        ):
            kept.append(character)
        else:
            kept.append(" ")
    return normalize_text("".join(kept))


def postprocess_ocr(text: str | None, min_telugu_chars: int = 2) -> OCRText:
    """Clean OCR output and flag empty or suspiciously short responses."""
    if min_telugu_chars < 1:
        raise ValueError("min_telugu_chars must be at least 1")

    cleaned = clean_telugu_text(text)
    count = sum(_is_telugu(character) for character in cleaned)
    if count == 0:
        reason = "no_telugu_text"
    elif count < min_telugu_chars:
        reason = f"fewer_than_{min_telugu_chars}_telugu_characters"
    else:
        reason = ""

    return OCRText(
        text=cleaned,
        suspicious_output=bool(reason),
        suspicious_reason=reason,
        telugu_character_count=count,
    )
