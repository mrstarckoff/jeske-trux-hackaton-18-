import base64
import re
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

# Canonical GID format: ANBB00001А
# 4 latin letters + 5 digits + trailing RU/EN letter (OCR often confuses А/A).
STRICT_GID_REGEX = re.compile(r"^[A-Z]{4}\d{5}[A-ZА-Я]$")
GID_CANDIDATE_REGEX = re.compile(r"\b[\wА-Яа-я]{10}\b")
COORD_PAIR_REGEX = re.compile(r"(-?\d{1,2}[\.,]\d+)\s*[,;\s]\s*(-?\d{1,3}[\.,]\d+)")

# Whisper/faster-whisper prompt tuning for ad-tech domain.
INITIAL_WHISPER_PROMPT = (
    "Контекст: распознавание описания рекламной поверхности. Термины: билборд, суперсайт, "
    "ситиборд, digital-экран, GID, адрес, широта, долгота, RussOutdoor, RussTech. "
    "Коды GID распознавай строго в шаблоне ANBB00001А (4 латинские буквы + 5 цифр + буква). "
    "Не заменяй цифры на буквы и наоборот."
)

# OCR confusion map for visual-similar letters
_CONFUSABLE_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)

app = FastAPI(title="OCR & ASR Service")


class ImageExtractRequest(BaseModel):
    image_b64: str


class AudioExtractRequest(BaseModel):
    audio_b64: str


class TextExtractResponse(BaseModel):
    gid: Optional[str] = None
    transcript: Optional[str] = None
    extracted_lat: Optional[float] = None
    extracted_lon: Optional[float] = None
    used_initial_prompt: Optional[str] = None


def _normalize_gid_candidate(token: str) -> str:
    token = token.strip().upper().replace(" ", "")
    if len(token) != 10:
        return token

    # Normalize first 4 letters to Latin when OCR returns Cyrillic look-alikes.
    prefix = token[:4].translate(_CONFUSABLE_TO_LATIN)
    core = token[4:9]
    suffix = token[9]
    return f"{prefix}{core}{suffix}"


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
    if not match:
        floats = re.findall(r"-?\d{1,3}[\.,]\d+", text)
        if len(floats) < 2:
            return None, None
        lat, lon = _to_float(floats[0]), _to_float(floats[1])
    else:
        lat, lon = _to_float(match.group(1)), _to_float(match.group(2))

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
        used_initial_prompt=INITIAL_WHISPER_PROMPT,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
