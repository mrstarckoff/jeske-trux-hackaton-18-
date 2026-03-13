import asyncio
import base64
import math
import os
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from rapidfuzz import fuzz

OCR_ASR_URL = os.getenv("OCR_ASR_URL", "http://ocr_asr:8001")
VECTOR_URL = os.getenv("VECTOR_URL", "http://vector_search:8002")
GEO_URL = os.getenv("GEO_URL", "http://geo:8003")
MOCK_DB_URL = os.getenv("MOCK_DB_URL", "http://mock_db:8004")

RRF_K = 60

app = FastAPI(title="Billboard Smart Search Orchestrator")


class RankedItem(BaseModel):
    gid: str
    confidence: float
    score: float
    card: dict


class SearchResponse(BaseModel):
    exact_match: Optional[dict] = None
    top5: list[RankedItem]
    debug: dict


def _weights(has_image: bool, has_text: bool, has_ocr_gid: bool) -> dict[str, float]:
    if has_ocr_gid:
        return {"ocr": 1.0, "visual": 0.0, "semantic": 0.0, "lexical": 0.0, "geo": 0.0}
    if has_image and has_text:
        return {"ocr": 0.0, "visual": 0.4, "semantic": 0.3, "lexical": 0.15, "geo": 0.15}
    if has_image:
        return {"ocr": 0.0, "visual": 0.7, "semantic": 0.0, "lexical": 0.0, "geo": 0.3}
    return {"ocr": 0.0, "visual": 0.0, "semantic": 0.5, "lexical": 0.3, "geo": 0.2}


def _rrf(lists: dict[str, list[dict]], weights: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for source, items in lists.items():
        for rank, item in enumerate(items, start=1):
            gid = item["gid"]
            scores[gid] = scores.get(gid, 0.0) + weights.get(source, 0.0) * (1.0 / (RRF_K + rank))
    return scores


def _to_confidence(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    x = (score / max_score) * 8 - 4
    return round(100 / (1 + math.exp(-x)), 2)


async def _get_card(client: httpx.AsyncClient, gid: str) -> dict:
    resp = await client.get(f"{MOCK_DB_URL}/objects/{gid}")
    resp.raise_for_status()
    return resp.json()


async def _lexical_search(client: httpx.AsyncClient, text: str, top_k: int = 10) -> list[dict]:
    if not text.strip():
        return []
    resp = await client.get(f"{MOCK_DB_URL}/objects", params={"limit": 1000})
    resp.raise_for_status()
    items = resp.json().get("items", [])
    scored = []
    for item in items:
        hay = f"{item.get('address', '')} {item.get('description', '')} {item.get('gid', '')}"
        score = fuzz.token_set_ratio(text.lower(), hay.lower()) / 100.0
        scored.append({"gid": item["gid"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@app.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile | None = File(default=None),
    audio: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None),
):
    if image is None and audio is None and not text and (lat is None or lon is None):
        raise HTTPException(status_code=422, detail="Provide at least one modality: image/audio/text/lat+lon")

    debug = {"modalities": []}
    image_b64 = None
    if image is not None:
        image_b64 = base64.b64encode(await image.read()).decode("utf-8")
        debug["modalities"].append("image")

    audio_b64 = None
    if audio is not None:
        audio_b64 = base64.b64encode(await audio.read()).decode("utf-8")
        debug["modalities"].append("audio")

    if text:
        debug["modalities"].append("text")
    if lat is not None and lon is not None:
        debug["modalities"].append("coords")

    async with httpx.AsyncClient(timeout=25.0) as client:
        req_map = {}
        if image_b64:
            req_map["ocr_image"] = client.post(f"{OCR_ASR_URL}/extract/image", json={"image_b64": image_b64})
            req_map["visual"] = client.post(f"{VECTOR_URL}/search/image", json={"image_b64": image_b64, "top_k": 10})
        if audio_b64:
            req_map["audio"] = client.post(f"{OCR_ASR_URL}/extract/audio", json={"audio_b64": audio_b64})
        if text:
            req_map["text_nlp"] = client.post(f"{OCR_ASR_URL}/extract/text", json={"text": text})
            req_map["semantic"] = client.post(f"{VECTOR_URL}/search/text", json={"text": text, "top_k": 10})
        if lat is not None and lon is not None:
            req_map["geo"] = client.post(f"{GEO_URL}/filter", json={"lat": lat, "lon": lon, "radius_m": 1000.0, "top_k": 10})

        names = list(req_map)
        responses = await asyncio.gather(*req_map.values())
        payloads = {}
        for name, resp in zip(names, responses):
            resp.raise_for_status()
            payloads[name] = resp.json()

        extracted_gid = payloads.get("ocr_image", {}).get("gid")
        combined_text_parts = []
        extracted_addresses = []

        if payloads.get("text_nlp"):
            extracted_gid = extracted_gid or payloads["text_nlp"].get("gid")
            combined_text_parts.append(payloads["text_nlp"].get("transcript") or "")
            extracted_addresses.extend(payloads["text_nlp"].get("extracted_addresses", []))
            if (lat is None or lon is None) and payloads["text_nlp"].get("extracted_lat") is not None:
                lat, lon = payloads["text_nlp"]["extracted_lat"], payloads["text_nlp"]["extracted_lon"]

        if payloads.get("audio"):
            extracted_gid = extracted_gid or payloads["audio"].get("gid")
            combined_text_parts.append(payloads["audio"].get("transcript") or "")
            extracted_addresses.extend(payloads["audio"].get("extracted_addresses", []))
            if (lat is None or lon is None) and payloads["audio"].get("extracted_lat") is not None:
                lat, lon = payloads["audio"]["extracted_lat"], payloads["audio"]["extracted_lon"]

        merged_text = " ".join(x.strip() for x in combined_text_parts if x and x.strip()) or (text or "")

        if lat is not None and lon is not None and "geo" not in payloads:
            geo_resp = await client.post(f"{GEO_URL}/filter", json={"lat": lat, "lon": lon, "radius_m": 1000.0, "top_k": 10})
            geo_resp.raise_for_status()
            payloads["geo"] = geo_resp.json()

        if merged_text and "semantic" not in payloads:
            sem_resp = await client.post(f"{VECTOR_URL}/search/text", json={"text": merged_text, "top_k": 10})
            sem_resp.raise_for_status()
            payloads["semantic"] = sem_resp.json()

        lexical = await _lexical_search(client, (merged_text + " " + " ".join(extracted_addresses)).strip(), top_k=10)
        payloads["lexical"] = {"results": lexical}

        if extracted_gid:
            card = await _get_card(client, extracted_gid)
            return SearchResponse(
                exact_match=card,
                top5=[],
                debug={**debug, "reason": "exact_gid", "gid": extracted_gid, "addresses": extracted_addresses},
            )

        lists = {
            "visual": payloads.get("visual", {}).get("results", []),
            "semantic": payloads.get("semantic", {}).get("results", []),
            "lexical": payloads.get("lexical", {}).get("results", []),
            "geo": payloads.get("geo", {}).get("results", []),
        }
        weights = _weights(has_image=image_b64 is not None, has_text=bool(merged_text), has_ocr_gid=False)
        rrf_scores = _rrf(lists, weights)
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        max_score = ranked[0][1] if ranked else 1.0

        top5 = []
        for gid, score in ranked:
            card = await _get_card(client, gid)
            top5.append(RankedItem(gid=gid, score=round(score, 6), confidence=_to_confidence(score, max_score), card=card))

        return SearchResponse(
            exact_match=None,
            top5=top5,
            debug={**debug, "weights": weights, "signals": list(lists), "addresses": list(dict.fromkeys(extracted_addresses))},
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
