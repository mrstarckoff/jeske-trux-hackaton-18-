from pydantic import BaseModel, Field


class CandidateCard(BaseModel):
    gid: str
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    probability: float = Field(..., ge=0.0, le=100.0)
    channels: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query_text: str | None = None
    extracted_gid: str | None = None
    mode: str
    top_k: list[CandidateCard]
