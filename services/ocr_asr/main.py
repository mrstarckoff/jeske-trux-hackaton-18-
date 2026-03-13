import base64
import re
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from natasha import AddrExtractor, MorphVocab
except Exception:  # optional dependency runtime fallback
    AddrExtractor = None
    MorphVocab = None

STRICT_GID_REGEX = re.compile(r"^[A-Z]{4}\d{5}[A-ZА-Я]$")
GID_CANDIDATE_REGEX = re.compile(r"\b[\wА-Яа-я]{10}\b")
COORD_PAIR_REGEX = re.compile(r"(-?\d{1,2}[\.,]\d+)\s*[,;\s]\s*(-?\d{1,3}[\.,]\d+)")
FALLBACK_ADDRESS_REGEX = re.compile(
    r"\b(?:ул\.?|улица|проспект|пр-т|шоссе|наб\.?|переулок)\s+[А-ЯA-ZЁ][^,.;]{2,40}\b",
    flags=re.IGNORECASE,
)

INITIAL_WHISPER_PROMPT = (
    "Контекст: рекламные поверхности и геолокация. Слова: Ангарск, билборд, суперсайт, ситиборд, "
    "рекламная поверхность, GID, адрес, широта, долгота. "
    "Коды GID строго в формате ANBB00001А: 4 латинские буквы, 5 цифр, финальная буква. "
    "Не путай B/В, A/А, O/0 и цифры с буквами."
)

_CONFUSABLE_TO_LATIN = str.maketrans(
    {"А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X"}
)

app = FastAPI(title="OCR & ASR Service")


class ImageExtractRequest(BaseModel):
    image_b64: str


class AudioExtractRequest(BaseModel):
    audio_b64: str


class TextExtractRequest(BaseModel):
    text: str


class TextExtractResponse(BaseModel):
    gid: Optional[str] = None
    transcript: Optional[str] = None
    extracted_lat: Optional[float] = None
    extracted_lon: Optional[float] = None
    extracted_addresses: list[str] = []
    used_initial_prompt: Optional[str] = None


@lru_cache(maxsize=1)
def _natasha_addr_extractor():
    if AddrExtractor is None or MorphVocab is None:
        return None
    morph_vocab = MorphVocab()
    return AddrExtractor(morph_vocab)


def _extract_addresses(text: str) -> list[str]:
    values: list[str] = []
    extractor = _natasha_addr_extractor()
    if extractor is not None:
        for match in extractor(text):
            frag = text[match.start : match.stop].strip()
            if frag:
                values.append(frag)
    if not values:
        values = [m.group(0).strip() for m in FALLBACK_ADDRESS_REGEX.finditer(text)]
    seen = set()
    dedup = []
    for v in values:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(v)
    return dedup[:5]


def _normalize_gid_candidate(token: str) -> str:
    token = token.strip().upper().replace(" ", "")
    if len(token) != 10:
        return token
    return f"{token[:4].translate(_CONFUSABLE_TO_LATIN)}{token[4:9]}{token[9]}"


def _find_gid(text: str) -> Optional[str]:
    for raw in GID_CANDIDATE_REGEX.findall(text.upper()):
        candidate = _normalize_gid_candidate(raw)
        if STRICT_GID_REGEX.fullmatch(candidate):
            return candidate
    return None


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _extract_coords(text: str) -> tuple[Optional[float], Optional[float]]:
    lowered = text.lower()
    if "широта" not in lowered and "долгота" not in lowered and not COORD_PAIR_REGEX.search(text):
        return None, None
    match = COORD_PAIR_REGEX.search(text)
    if match:
        lat, lon = _to_float(match.group(1)), _to_float(match.group(2))
    else:
        floats = re.findall(r"-?\d{1,3}[\.,]\d+", text)
        if len(floats) < 2:
            return None, None
        lat, lon = _to_float(floats[0]), _to_float(floats[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None
    return lat, lon


@app.post("/extract/image", response_model=TextExtractResponse)
def extract_from_image(req: ImageExtractRequest) -> TextExtractResponse:
    raw = base64.b64decode(req.image_b64)
    as_text = raw.decode("utf-8", errors="ignore").upper()
    return TextExtractResponse(gid=_find_gid(as_text))


@app.post("/extract/audio", response_model=TextExtractResponse)
def extract_from_audio(req: AudioExtractRequest) -> TextExtractResponse:
    raw = base64.b64decode(req.audio_b64)
    transcript = raw.decode("utf-8", errors="ignore").strip() or ""
    gid = _find_gid(transcript)
    lat, lon = _extract_coords(transcript)
    return TextExtractResponse(
        gid=gid,
        transcript=transcript,
        extracted_lat=lat,
        extracted_lon=lon,
        extracted_addresses=_extract_addresses(transcript),
        used_initial_prompt=INITIAL_WHISPER_PROMPT,
    )


@app.post("/extract/text", response_model=TextExtractResponse)
def extract_from_text(req: TextExtractRequest) -> TextExtractResponse:
    gid = _find_gid(req.text)
    lat, lon = _extract_coords(req.text)
    return TextExtractResponse(
        gid=gid,
        transcript=req.text,
        extracted_lat=lat,
        extracted_lon=lon,
        extracted_addresses=_extract_addresses(req.text),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "natasha": _natasha_addr_extractor() is not None}
