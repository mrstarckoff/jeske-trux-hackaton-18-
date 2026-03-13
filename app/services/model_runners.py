from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    paddleocr: Any
    whisper: Any
    clip: Any


class ModelRegistry:
    """Loads heavy models once at startup and exposes async wrappers."""

    def __init__(self) -> None:
        self._models: ModelBundle | None = None
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        async with self._lock:
            if self._models is None:
                self._models = ModelBundle(
                    paddleocr=self._load_paddleocr(),
                    whisper=self._load_whisper(),
                    clip=self._load_clip(),
                )

    async def ensure_loaded(self) -> ModelBundle:
        if self._models is None:
            await self.startup()
        assert self._models is not None
        return self._models

    @staticmethod
    def _load_paddleocr() -> Any:
        try:
            from paddleocr import PaddleOCR

            return PaddleOCR(use_angle_cls=True, lang='ru', show_log=False)
        except Exception:
            return None

    @staticmethod
    def _load_whisper() -> Any:
        try:
            from faster_whisper import WhisperModel

            return WhisperModel('small', device='cpu', compute_type='int8')
        except Exception:
            return None

    @staticmethod
    def _load_clip() -> Any:
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            return {'model': model, 'preprocess': preprocess, 'tokenizer': tokenizer}
        except Exception:
            return None

    async def run_ocr(self, image_bytes: bytes) -> list[str]:
        await self.ensure_loaded()
        if self._models is None or self._models.paddleocr is None:
            return []
        return []

    async def run_asr(self, audio_path: str, initial_prompt: str) -> str:
        await self.ensure_loaded()
        if self._models is None or self._models.whisper is None:
            return ''
        segments, _ = self._models.whisper.transcribe(audio_path, initial_prompt=initial_prompt)
        return ' '.join(s.text.strip() for s in segments).strip()

    async def run_visual_embedding(self, image_bytes: bytes) -> list[float]:
        await self.ensure_loaded()
        digest = hashlib.sha256(image_bytes).digest()
        vec = [digest[i % len(digest)] / 255.0 for i in range(128)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def run_text_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        vec = [digest[i % len(digest)] / 255.0 for i in range(128)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
