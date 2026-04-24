"""FastAPI server for dynamic self-pruned model inference."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.network import CIFAR10PrunableNet


logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
	"""Request payload for selecting an inference mode."""

	mode: Literal["fast", "balanced", "accurate"]


class PredictResponse(BaseModel):
	"""Response payload containing prediction and runtime metrics."""

	prediction: int
	model_used: str
	latency_ms: float
	sparsity: float


@dataclass
class ModelEntry:
	"""Container for model metadata and module reference."""

	name: str
	model: CIFAR10PrunableNet


class ModelService:
	"""Manages model loading and retrieval for inference requests."""

	def __init__(self) -> None:
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._models: Dict[str, ModelEntry] = {}

	def load_models(self) -> None:
		"""Load and register all inference model variants."""
		# Placeholder model loading; swap with real checkpoints later.
		self._models = {
			"fast": ModelEntry(name="fast_model", model=CIFAR10PrunableNet()),
			"balanced": ModelEntry(name="balanced_model", model=CIFAR10PrunableNet()),
			"accurate": ModelEntry(name="accurate_model", model=CIFAR10PrunableNet()),
		}

		for entry in self._models.values():
			entry.model.to(self.device)
			entry.model.eval()

	def get(self, mode: str) -> ModelEntry:
		"""Return registered model entry for an inference mode."""
		if mode not in self._models:
			raise KeyError(mode)
		return self._models[mode]


app = FastAPI(title="Self-Pruning Inference API")


@app.get("/health")
async def health() -> Dict[str, str]:
	"""Return service health status."""
	return {"status": "ok"}


@app.on_event("startup")
async def startup_event() -> None:
	"""Initialize model service and load model registry on startup."""
	service = ModelService()
	service.load_models()
	app.state.model_service = service


def _get_model_service() -> ModelService:
	"""Retrieve initialized model service from application state."""
	service = getattr(app.state, "model_service", None)
	if service is None:
		raise RuntimeError("Model service is not initialized.")
	return service


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
	"""Run inference using dynamically selected model mode."""
	service = _get_model_service()
	logger.info("Inference request received for mode=%s", request.mode)

	try:
		entry = service.get(request.mode)
	except KeyError as exc:
		raise HTTPException(status_code=400, detail="Invalid mode") from exc

	model = entry.model
	input_tensor = torch.randn(1, 3, 32, 32).to(service.device)

	start_time = time.perf_counter()
	with torch.no_grad():
		logits = model(input_tensor)
		prediction = int(logits.argmax(dim=1).item())
	latency_ms = (time.perf_counter() - start_time) * 1000.0

	return PredictResponse(
		prediction=prediction,
		model_used=entry.name,
		latency_ms=float(latency_ms),
		sparsity=float(model.get_total_sparsity()),
	)
