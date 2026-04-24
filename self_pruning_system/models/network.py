"""CIFAR-10 classifier architecture using prunable linear layers."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from core.layer import PrunableLinear


class CIFAR10PrunableNet(nn.Module):
	"""A compact CNN-MLP hybrid with a prunable classifier for CIFAR-10.

	The classifier uses learnable-gate linear layers to support dynamic
	sparsification while keeping the overall architecture simple and modular.
	"""

	def __init__(self) -> None:
		super().__init__()
		self._feature_dim = 64 * 16 * 16

		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)

		self.classifier = nn.Sequential(
			PrunableLinear(self._feature_dim, 512),
			nn.ReLU(),
			PrunableLinear(512, 10),
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Run a forward pass for CIFAR-10 classification logits."""
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

	def get_total_sparsity(self) -> float:
		"""Return average sparsity (%) across all PrunableLinear layers."""
		sparsities = [
			module.get_sparsity()
			for module in self.modules()
			if isinstance(module, PrunableLinear)
		]
		if not sparsities:
			return 0.0
		return float(sum(sparsities) / len(sparsities))

	def get_all_snrs(self) -> List[Tensor]:
		"""Return SNR tensors collected from all PrunableLinear layers."""
		return [
			module.get_snr()
			for module in self.modules()
			if isinstance(module, PrunableLinear)
		]

	def get_mean_snr(self) -> float:
		"""Return the mean SNR across all prunable connections."""
		snrs = self.get_all_snrs()
		if not snrs:
			return 0.0

		flattened = torch.cat([snr.reshape(-1) for snr in snrs], dim=0)
		return float(flattened.mean().item())
