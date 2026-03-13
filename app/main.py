from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from app.config import settings
from app.schemas import CandidateCard, SearchResponse
from app.services.feature_extraction import WHISPER_INITIAL_PROMPT, extract_text_features
from app.services.model_runners import ModelRegistry
from app.services.repository import BillboardRepository
from app.services.retrieval import geo_filter, lexical_top_k, platt_scaling, weighted_rrf

app = FastAPI(title=settings.app_name)
models = ModelRegistry()
repo = BillboardRepository(csv_path=settings.dataset_csv)


@app.on_event('startup')
async def startup_event() -> None:
    await models.startup()


@app.post('/search', response_model=SearchResponse)
async def search(
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
) -> SearchResponse:
    image_bytes = await image.read() if image else b''
    audio_text_path = None
    if audio:
        suffix = Path(audio.filename or 'audio.wav').suffix or '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            audio_text_path = tmp.name

    ocr_task = models.run_ocr(image_bytes) if image_bytes else asyncio.sleep(0, result=[])
    asr_task = (
        models.run_asr(audio_text_path, WHISPER_INITIAL_PROMPT) if audio_text_path else asyncio.sleep(0, result='')
    )
    visual_task = models.run_visual_embedding(image_bytes) if image_bytes else asyncio.sleep(0, result=[])
    ocr_tokens, asr_text, visual_embedding = await asyncio.gather(ocr_task, asr_task, visual_task)

    features = extract_text_features(text, asr_text, ocr_tokens)

    if features.gid:
        card = repo.get_by_gid(features.gid)
        if card:
            return SearchResponse(
                query_text=features.merged_text,
                extracted_gid=features.gid,
                mode='priority_gid',
                top_k=[
                    CandidateCard(
                        gid=card['gid'],
                        address=card.get('address'),
                        latitude=card.get('latitude'),
                        longitude=card.get('longitude'),
                        probability=100.0,
                        channels=['ocr'],
                    )
                ],
            )

    records = repo.load_all()
    semantic_embedding = await models.run_text_embedding(features.merged_text or '')

    rankings: dict[str, list[str]] = {
        'visual': repo.vector_top_k(visual_embedding, settings.max_candidates_per_channel) if visual_embedding else [],
        'semantic': repo.vector_top_k(semantic_embedding, settings.max_candidates_per_channel),
        'lexical': lexical_top_k(features.addresses, records, settings.max_candidates_per_channel),
        'geo': geo_filter((latitude, longitude) if latitude and longitude else None, records),
    }
    weights = {'visual': 0.4, 'semantic': 0.3, 'lexical': 0.2, 'geo': 0.1}
    scores = weighted_rrf(rankings, weights, settings.rrf_k)
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: settings.final_top_k]

    result = []
    for gid, score in sorted_candidates:
        card = repo.get_by_gid(gid)
        if not card:
            continue
        channels = [channel for channel, items in rankings.items() if gid in items]
        result.append(
            CandidateCard(
                gid=gid,
                address=card.get('address'),
                latitude=card.get('latitude'),
                longitude=card.get('longitude'),
                probability=platt_scaling(score),
                channels=channels,
            )
        )

    return SearchResponse(query_text=features.merged_text, extracted_gid=features.gid, mode='multimodal', top_k=result)
