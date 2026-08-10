"""Dual REST API: the LLM both *uses* (infer) and *teaches* (evolve) a
SoftModel model, plus version management. Thin layer over SoftModelFactory.

Run:  python scripts/run_api.py   (or: uvicorn generator.api:app)
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Config
from .factory import SoftModelFactory
from .spec import ModelSpec


class Example(BaseModel):
    # Model input: a feature object (e.g. {"amount": 800, "night": 0}) extracted
    # by the calling LLM; plain strings are accepted by the mock backend only.
    # target may be a label or a number — the MODEL shapes itself from its data.
    input: Any
    target: str


class CreateModelBody(BaseModel):
    # Deliberately minimal: no types, no schemas. The factory manufactures;
    # the model learns its own shape from the data it is taught.
    model_id: str
    description: str = ""
    holdout: list[Example] = []


class InferBody(BaseModel):
    input: Any
    version: Optional[str] = None


class TeachBody(BaseModel):
    examples: list[Example]
    mode: str = "sft"
    # M2: train on only the most recent N examples of the accumulated store
    # (sheds old labels the new reality contradicts). None = full replay.
    window: Optional[int] = None
    # M2: judge the promotion gate on only the most recent N held-out
    # examples (post-drift, a mixed-era holdout can tie and block adaptation).
    recent_n: Optional[int] = None


class HoldoutBody(BaseModel):
    examples: list[Example]


class DriftBody(BaseModel):
    recent_n: Optional[int] = None


class RollbackBody(BaseModel):
    to: str


def create_app(config: Optional[Config] = None) -> FastAPI:
    factory = SoftModelFactory(config or Config.from_env())
    app = FastAPI(title="SoftModel Model", version="0.6.1")

    def _require(model_id: str) -> None:
        if not factory.exists(model_id):
            raise HTTPException(404, f"model '{model_id}' not found")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "backend": factory.config.backend}

    @app.post("/v1/models")
    def create_model(body: CreateModelBody) -> dict[str, Any]:
        return factory.create(ModelSpec(
            model_id=body.model_id, description=body.description,
            holdout=[e.model_dump() for e in body.holdout]))

    @app.get("/v1/models")
    def list_models() -> list[dict[str, Any]]:
        return factory.list_models()

    @app.post("/v1/models/{model_id}/infer")
    def infer(model_id: str, body: InferBody) -> dict[str, Any]:
        _require(model_id)
        return factory.infer(model_id, body.input, version=body.version)

    @app.post("/v1/models/{model_id}/teach")
    def teach(model_id: str, body: TeachBody) -> dict[str, Any]:
        _require(model_id)
        return factory.teach(model_id, [e.model_dump() for e in body.examples],
                             body.mode, window=body.window,
                             recent_n=body.recent_n)

    @app.post("/v1/models/{model_id}/holdout")
    def add_holdout(model_id: str, body: HoldoutBody) -> dict[str, Any]:
        _require(model_id)
        return factory.add_holdout(model_id, [e.model_dump() for e in body.examples])

    @app.post("/v1/models/{model_id}/drift")
    def drift(model_id: str, body: DriftBody = DriftBody()) -> dict[str, Any]:
        _require(model_id)
        return factory.check_drift(model_id, recent_n=body.recent_n)

    @app.post("/v1/models/{model_id}/evaluate")
    def evaluate(model_id: str, version: Optional[str] = None,
                 recent_n: Optional[int] = None) -> dict[str, Any]:
        _require(model_id)
        return factory.evaluate(model_id, version=version, recent_n=recent_n)

    @app.get("/v1/models/{model_id}/versions")
    def versions(model_id: str) -> dict[str, Any]:
        _require(model_id)
        return factory.versions(model_id)

    @app.post("/v1/models/{model_id}/rollback")
    def rollback(model_id: str, body: RollbackBody) -> dict[str, Any]:
        _require(model_id)
        return factory.rollback(model_id, body.to)

    @app.get("/v1/models/{model_id}/discoveries")
    def discoveries(model_id: str, version: Optional[str] = None) -> dict[str, Any]:
        _require(model_id)
        try:
            return factory.discoveries(model_id, version=version)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/v1/models/{model_id}/card")
    def card(model_id: str) -> dict[str, Any]:
        _require(model_id)
        return factory.card(model_id)

    return app


app = create_app()
