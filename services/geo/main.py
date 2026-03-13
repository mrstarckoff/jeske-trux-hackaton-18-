import csv
import os
from pathlib import Path

from fastapi import FastAPI
from geopy.distance import geodesic
from pydantic import BaseModel

DATA_PATH = Path(os.getenv("BILLBOARD_CSV", Path(__file__).resolve().parents[2] / "data" / "billboards.csv"))

app = FastAPI(title="Geo Service")


class GeoFilterRequest(BaseModel):
    lat: float
    lon: float
    radius_m: float = 1000.0
    top_k: int = 10


def _load_db() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.post("/filter")
def geo_filter(req: GeoFilterRequest) -> dict:
    rows = _load_db()
    filtered = []
    for row in rows:
        dist = geodesic((req.lat, req.lon), (float(row["lat"]), float(row["lon"]))).meters
        if dist <= req.radius_m:
            score = max(0.0, 1.0 - dist / req.radius_m)
            filtered.append({"gid": row["gid"], "distance_m": dist, "score": score})
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return {"results": filtered[: req.top_k]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
