import csv
import os
import math
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

DATA_PATH = Path(os.getenv("BILLBOARD_CSV", Path(__file__).resolve().parents[2] / "data" / "billboards.csv"))

app = FastAPI(title="Geo Service")


class GeoFilterRequest(BaseModel):
    lat: float
    lon: float
    radius_m: float = 200.0
    top_k: int = 10


def _load_db() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.post("/filter")
def geo_filter(req: GeoFilterRequest) -> dict:
    rows = _load_db()
    filtered = []
    for row in rows:
        dist = _haversine(req.lat, req.lon, float(row["lat"]), float(row["lon"]))
        if dist <= req.radius_m:
            score = max(0.0, 1.0 - dist / req.radius_m)
            filtered.append({"gid": row["gid"], "distance_m": dist, "score": score})
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return {"results": filtered[: req.top_k]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
