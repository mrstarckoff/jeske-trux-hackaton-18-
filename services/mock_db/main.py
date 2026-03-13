import csv
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DATA_PATH = Path(os.getenv("BILLBOARD_CSV", Path(__file__).resolve().parents[2] / "data" / "billboards.csv"))

app = FastAPI(title="Mock DB Service")


class Coordinates(BaseModel):
    lat: float
    lon: float


class BillboardCard(BaseModel):
    gid: str
    address: str
    surface_type: str
    coordinates: Coordinates
    photo_url: str
    description: str


def _load_db() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_card(row: dict) -> BillboardCard:
    return BillboardCard(
        gid=row["gid"],
        address=row["address"],
        surface_type=row["type"],
        coordinates=Coordinates(lat=float(row["lat"]), lon=float(row["lon"])),
        photo_url=row["photo_url"],
        description=row["description"],
    )


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.get("/objects")
def list_objects(
    limit: int = Query(default=100, ge=1, le=1000),
    q: str | None = None,
) -> dict:
    items = [_to_card(row).model_dump() for row in _load_db()]
    if q:
        ql = q.lower()
        items = [x for x in items if ql in x["address"].lower() or ql in x["description"].lower() or ql in x["gid"].lower()]
    return {"items": items[:limit]}


@app.get("/objects/{gid}")
def get_object(gid: str) -> dict:
    for row in _load_db():
        if row["gid"].upper() == gid.upper():
            return _to_card(row).model_dump()
    raise HTTPException(status_code=404, detail="GID not found")


@app.get("/nearest")
def nearest(lat: float, lon: float, radius_m: float = 200.0, top_k: int = 5) -> dict:
    matches = []
    for row in _load_db():
        dist = _haversine(lat, lon, float(row["lat"]), float(row["lon"]))
        if dist <= radius_m:
            card = _to_card(row).model_dump()
            card["distance_m"] = round(dist, 2)
            matches.append(card)
    matches.sort(key=lambda x: x["distance_m"])
    return {"items": matches[:top_k]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
