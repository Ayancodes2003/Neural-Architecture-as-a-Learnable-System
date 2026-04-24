"""Evaluation and analysis utilities for self-pruning models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from core.layer import PrunableLinear


@dataclass
class RunResult:
	"""Container for a single run's tradeoff point."""

	lambda_value: float
	accuracy: float
	sparsity: float


def evaluate(model: nn.Module, dataloader: Iterable, device: torch.device | str) -> float:
	"""Evaluate a model and return classification accuracy."""
	device = torch.device(device)
	model.to(device)
	model.eval()

	total_correct = 0
	total_samples = 0

	with torch.no_grad():
		for inputs, targets in dataloader:
			inputs = inputs.to(device)
			targets = targets.to(device)

			logits = model(inputs)
			predictions = logits.argmax(dim=1)

			total_correct += int((predictions == targets).sum().item())
			total_samples += int(targets.size(0))

	return total_correct / max(total_samples, 1)


def extract_gates_and_snrs(model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
	"""Extract flattened gate and SNR values from all prunable layers."""
	gate_tensors: List[Tensor] = []

	for module in model.modules():
		if isinstance(module, PrunableLinear):
			gate_tensors.append(module.get_gates().detach().reshape(-1).cpu())

	snr_tensors = [snr.detach().reshape(-1).cpu() for snr in model.get_all_snrs()]

	if gate_tensors:
		all_gates = torch.cat(gate_tensors).numpy()
	else:
		all_gates = np.array([], dtype=np.float32)

	if snr_tensors:
		all_snrs = torch.cat(snr_tensors).numpy()
	else:
		all_snrs = np.array([], dtype=np.float32)

	return all_gates, all_snrs


def plot_gate_distribution(gates: np.ndarray, output_path: str | Path = "gate_distribution.png") -> None:
	"""Plot and save histogram of gate values."""
	plt.figure(figsize=(8, 5))
	plt.hist(gates, bins=50)
	plt.title("Gate Distribution")
	plt.xlabel("Gate Value")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def plot_snr_vs_gate(
	snrs: np.ndarray,
	gates: np.ndarray,
	output_path: str | Path = "snr_vs_gate.png",
) -> None:
	"""Plot and save SNR-to-gate scatter relationship."""
	count = min(snrs.size, gates.size)
	snr_values = snrs[:count]
	gate_values = gates[:count]

	if count > 5000:
		indices = np.random.choice(count, size=5000, replace=False)
		snr_values = snr_values[indices]
		gate_values = gate_values[indices]

	plt.figure(figsize=(8, 5))
	plt.scatter(np.clip(snr_values, 1e-12, None), gate_values, s=6, alpha=0.5)
	plt.xscale("log")
	plt.title("SNR vs Gate Value")
	plt.xlabel("SNR")
	plt.ylabel("Gate Value")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def add_run_result(
	results: List[RunResult],
	lambda_value: float,
	accuracy: float,
	sparsity: float,
) -> None:
	"""Append a run result for Pareto analysis."""
	results.append(
		RunResult(
			lambda_value=lambda_value,
			accuracy=accuracy,
			sparsity=sparsity,
		)
	)


def plot_pareto(
	results: Sequence[RunResult],
	output_path: str | Path = "pareto.png",
) -> None:
	"""Plot and save Pareto curve using sparsity (x) and accuracy (y)."""
	sparsity_values = [result.sparsity for result in results]
	accuracy_values = [result.accuracy for result in results]

	plt.figure(figsize=(8, 5))
	plt.plot(sparsity_values, accuracy_values, marker="o")
	for result in results:
		plt.annotate(
			f"{result.lambda_value:g}",
			(result.sparsity, result.accuracy),
			textcoords="offset points",
			xytext=(4, 4),
		)
	plt.title("Pareto Curve")
	plt.xlabel("Sparsity")
	plt.ylabel("Accuracy")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()
