from __future__ import annotations

import re
from dataclasses import dataclass

GID_PATTERN = re.compile(r'^[A-Z]{4}\d{5}[А-ЯA-Z]$')

WHISPER_INITIAL_PROMPT = 'Ангарск, Билборд, Широта, Долгота, адрес, улица, дом.'


@dataclass
class TextFeatures:
    gid: str | None
    addresses: list[str]
    merged_text: str


def extract_priority_gid(tokens: list[str]) -> str | None:
    for token in tokens:
        normalized = token.strip().replace(' ', '')
        if GID_PATTERN.match(normalized):
            return normalized
    return None


def extract_addresses_with_natasha(text: str) -> list[str]:
    try:
        from natasha import AddrExtractor, MorphVocab

        extractor = AddrExtractor(MorphVocab())
        return [text[match.start:match.stop] for match in extractor(text)]
    except Exception:
        return []


def merge_and_clean_text(raw_text: str | None, asr_text: str | None) -> str:
    merged = ' '.join(part for part in [raw_text or '', asr_text or ''] if part).strip()
    merged = re.sub(r'\s+', ' ', merged)
    return merged


def extract_text_features(raw_text: str | None, asr_text: str | None, ocr_tokens: list[str]) -> TextFeatures:
    gid = extract_priority_gid(ocr_tokens)
    merged_text = merge_and_clean_text(raw_text, asr_text)
    addresses = extract_addresses_with_natasha(merged_text)
    return TextFeatures(gid=gid, addresses=addresses, merged_text=merged_text)
