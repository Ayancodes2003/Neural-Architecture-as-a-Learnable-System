"""Training engine for self-pruning models with SNR-aware sparsity regularization."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

from core.layer import PrunableLinear


class Trainer:
	"""Minimal training engine for prunable neural networks."""

	def __init__(
		self,
		model: nn.Module,
		optimizer: torch.optim.Optimizer,
		device: torch.device | str,
		lambda_max: float,
		epochs: int,
	) -> None:
		self.model = model
		self.optimizer = optimizer
		self.device = torch.device(device)
		self.lambda_max = float(lambda_max)
		self.epochs = int(epochs)

		self.model.to(self.device)

	def get_lambda(self, epoch: int) -> float:
		"""Return lambda with linear warmup over the first 20% of epochs."""
		warmup_epochs = max(1, int(0.2 * self.epochs))
		if epoch < warmup_epochs:
			return self.lambda_max * (epoch / warmup_epochs)
		return self.lambda_max

	def compute_sparsity_loss(self) -> torch.Tensor:
		"""Compute SNR-weighted sparsity loss across all prunable layers."""
		total_loss = torch.zeros(1, device=self.device)

		for module in self.model.modules():
			if isinstance(module, PrunableLinear):
				gates = module.get_gates()
				snr = module.get_snr()

				weight = 1.0 / (snr + 1e-8)
				weight = weight.clamp(max=10.0)
				total_loss = total_loss + (gates * weight).mean()

		return total_loss

	def train_epoch(self, dataloader: Any, epoch: int) -> Dict[str, float]:
		"""Run one training epoch and return aggregate metrics."""
		self.model.train()

		lambda_value = self.get_lambda(epoch)
		total_loss = 0.0
		total_correct = 0
		total_samples = 0
		total_batches = 0

		for inputs, targets in dataloader:
			inputs = inputs.to(self.device)
			targets = targets.to(self.device)

			self.optimizer.zero_grad(set_to_none=True)

			logits = self.model(inputs)
			ce_loss = F.cross_entropy(logits, targets)
			sparsity_loss = self.compute_sparsity_loss()
			loss = ce_loss + lambda_value * sparsity_loss

			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			self.optimizer.step()

			total_loss += float(loss.item())
			with torch.no_grad():
				predictions = logits.argmax(dim=1)
			total_correct += int((predictions == targets).sum().item())
			total_samples += int(targets.size(0))
			total_batches += 1

		avg_loss = total_loss / max(total_batches, 1)
		accuracy = total_correct / max(total_samples, 1)

		if hasattr(self.model, "get_total_sparsity"):
			sparsity = float(self.model.get_total_sparsity())
		else:
			sparsity_values = [
				module.get_sparsity()
				for module in self.model.modules()
				if isinstance(module, PrunableLinear)
			]
			sparsity = (
				float(sum(sparsity_values) / len(sparsity_values))
				if sparsity_values
				else 0.0
			)

		return {
			"loss": avg_loss,
			"accuracy": accuracy,
			"sparsity": sparsity,
		}
