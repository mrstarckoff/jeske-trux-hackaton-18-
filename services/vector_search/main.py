import base64
import csv
import os
import hashlib
import math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

DATA_PATH = Path(os.getenv("BILLBOARD_CSV", Path(__file__).resolve().parents[2] / "data" / "billboards.csv"))
DIM = 32

app = FastAPI(title="Vector Search Service")


class SearchImageRequest(BaseModel):
    image_b64: str
    top_k: int = 10


class SearchTextRequest(BaseModel):
    text: str
    top_k: int = 10


def _hash_embedding(payload: bytes) -> list[float]:
    digest = hashlib.sha256(payload).digest()
    out = []
    while len(out) < DIM:
        for b in digest:
            out.append((b / 255.0) * 2.0 - 1.0)
            if len(out) == DIM:
                break
        digest = hashlib.sha256(digest).digest()
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _load_db() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _search(query_vec: list[float], top_k: int) -> list[dict]:
    rows = _load_db()
    scored = []
    for row in rows:
        basis = f"{row['gid']}|{row['address']}|{row['description']}".encode("utf-8")
        score = _cos(query_vec, _hash_embedding(basis))
        scored.append({"gid": row["gid"], "score": float((score + 1) / 2)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@app.post("/search/image")
def search_image(req: SearchImageRequest) -> dict:
    raw = base64.b64decode(req.image_b64)
    q = _hash_embedding(raw)
    return {"results": _search(q, req.top_k)}


@app.post("/search/text")
def search_text(req: SearchTextRequest) -> dict:
    q = _hash_embedding(req.text.encode("utf-8"))
    return {"results": _search(q, req.top_k)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
