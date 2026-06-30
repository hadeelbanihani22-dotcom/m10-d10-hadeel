"""Drill 10 — Typed Retrieval Endpoint.

Build a FastAPI service with a single `/retrieve` POST endpoint that
returns a top-k token-overlap retrieval against an in-memory fixture,
plus a `/healthz` liveness probe.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrieval import retrieve_top_k  # pre-implemented; do not modify retrieval.py

app = FastAPI(title="Drill 10 — Typed Retrieval Endpoint")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(3, ge=1, le=10)


class Chunk(BaseModel):
    chunk_id: int
    text: str
    score: float


class RetrieveResponse(BaseModel):
    retrieved: list[Chunk]


class HealthResponse(BaseModel):
    status: str


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    raw = retrieve_top_k(req.query, req.k)
    return RetrieveResponse(retrieved=[Chunk(**r) for r in raw])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")