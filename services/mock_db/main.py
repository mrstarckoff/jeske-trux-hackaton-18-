import csv
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

DATA_PATH = Path(os.getenv("BILLBOARD_CSV", Path(__file__).resolve().parents[2] / "data" / "billboards.csv"))

app = FastAPI(title="Mock DB Service")


def _load_db() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.get("/objects")
def list_objects() -> dict:
    return {"items": _load_db()}


@app.get("/objects/{gid}")
def get_object(gid: str) -> dict:
    for row in _load_db():
        if row["gid"] == gid:
            return row
    raise HTTPException(status_code=404, detail="GID not found")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
