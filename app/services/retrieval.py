from __future__ import annotations

import math
from collections import defaultdict

from geopy.distance import geodesic
from rapidfuzz import fuzz


def weighted_rrf(rankings: dict[str, list[str]], weights: dict[str, float], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for channel, docs in rankings.items():
        weight = weights.get(channel, 0.0)
        if weight <= 0:
            continue
        for idx, gid in enumerate(docs, start=1):
            scores[gid] += weight * (1.0 / (k + idx))
    return dict(scores)


def platt_scaling(score: float, a: float = 12.0, b: float = -5.0) -> float:
    p = 1.0 / (1.0 + math.exp(-(a * score + b)))
    return round(p * 100.0, 2)


def lexical_top_k(query_addresses: list[str], records: list[dict], top_k: int) -> list[str]:
    if not query_addresses:
        return []
    query = ' '.join(query_addresses)
    scored = []
    for item in records:
        addr = item.get('address', '')
        score = fuzz.token_set_ratio(query, addr)
        scored.append((score, item['gid']))
    scored.sort(reverse=True)
    return [gid for _, gid in scored[:top_k]]


def geo_filter(center: tuple[float, float] | None, records: list[dict], radius_km: float = 1.0) -> list[str]:
    if center is None:
        return []
    result = []
    for item in records:
        lat, lon = item.get('latitude'), item.get('longitude')
        if lat is None or lon is None:
            continue
        if geodesic(center, (lat, lon)).km <= radius_km:
            result.append(item['gid'])
    return result
