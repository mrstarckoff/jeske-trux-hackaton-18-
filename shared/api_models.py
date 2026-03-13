from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.enums import SourceType


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured error details",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service: str
    version: str


class Coordinates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(..., description="Dataset item identifier")
    score: float = Field(..., description="Raw candidate score")
    source: SourceType = Field(..., description="Service that produced the candidate")
    title: str | None = Field(default=None, description="Optional short title")
    gid: str | None = Field(default=None, description="Detected or matched GID")
    distance_meters: float | None = Field(
        default=None,
        description="Distance in meters for geo candidates",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured metadata",
    )


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    rank: int = Field(..., ge=1)
    final_score: float = Field(..., description="Fused score after aggregation")
    title: str | None = None
    gid: str | None = None
    sources: list[SourceType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Card(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    title: str | None = None
    description: str | None = None
    address: str | None = None
    coordinates: Coordinates | None = None
    image_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Top5Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    rank: int = Field(..., ge=1, le=5)
    final_score: float
    title: str | None = None
    gid: str | None = None
    sources: list[SourceType] = Field(default_factory=list)
    card: Card | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)