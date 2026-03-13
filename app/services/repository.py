from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BillboardRepository:
    csv_path: str

    def load_all(self) -> list[dict]:
        path = Path(self.csv_path)
        if not path.exists():
            return []
        with path.open('r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        records = []
        for row in rows:
            records.append(
                {
                    'gid': row.get('gid', ''),
                    'address': row.get('address', ''),
                    'latitude': float(row['latitude']) if row.get('latitude') else None,
                    'longitude': float(row['longitude']) if row.get('longitude') else None,
                    'embedding': _parse_embedding(row.get('embedding', '')),
                }
            )
        return records

    def get_by_gid(self, gid: str) -> dict | None:
        gid = gid.strip()
        for rec in self.load_all():
            if rec['gid'] == gid:
                return rec
        return None

    def vector_top_k(self, embedding: list[float], top_k: int = 10) -> list[str]:
        records = self.load_all()
        scored = []
        for rec in records:
            emb = rec.get('embedding')
            if emb is None:
                continue
            score = _cosine_similarity(embedding, emb)
            scored.append((score, rec['gid']))
        scored.sort(reverse=True)
        return [gid for _, gid in scored[:top_k]]


def _parse_embedding(raw: str) -> list[float] | None:
    if not raw:
        return None
    try:
        return [float(x) for x in raw.split()]
    except ValueError:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
