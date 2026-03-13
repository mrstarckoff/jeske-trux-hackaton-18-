import base64
import re
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

GID_REGEX = re.compile(r"^[A-ZА-Я]{4}\d{5}[A-ZА-Я]$")
COORD_REGEX = re.compile(r"(-?\d{1,2}\.\d+)[,\s]+(-?\d{1,3}\.\d+)")
INITIAL_WHISPER_PROMPT = (
    "Реклама, билборд, GID, адрес, широта, долгота, поверхность, RussTech. "
    "Точно распознавай коды формата ANBB00001А и географические координаты."
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


def _find_gid(text: str) -> Optional[str]:
    for token in re.split(r"\s+|[,;:.]", text.upper()):
        if GID_REGEX.match(token):
            return token
    return None


def _extract_coords(text: str) -> tuple[Optional[float], Optional[float]]:
    lowered = text.lower()
    if "широта" in lowered or "долгота" in lowered or COORD_REGEX.search(text):
        match = COORD_REGEX.search(text)
        if match:
            return float(match.group(1)), float(match.group(2))
        floats = re.findall(r"-?\d{1,3}\.\d+", text)
        if len(floats) >= 2:
            return float(floats[0]), float(floats[1])
    return None, None


@app.post("/extract/image", response_model=TextExtractResponse)
def extract_from_image(req: ImageExtractRequest) -> TextExtractResponse:
    raw = base64.b64decode(req.image_b64)
    # Mock OCR: try to find ASCII text patterns in bytes.
    as_text = raw.decode("utf-8", errors="ignore").upper()
    gid = _find_gid(as_text)
    return TextExtractResponse(gid=gid)


@app.post("/extract/audio", response_model=TextExtractResponse)
def extract_from_audio(req: AudioExtractRequest) -> TextExtractResponse:
    raw = base64.b64decode(req.audio_b64)
    # Mock ASR: for demo use utf-8 payload as transcript if possible.
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
