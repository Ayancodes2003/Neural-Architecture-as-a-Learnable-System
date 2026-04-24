"""Prunable linear layer with learnable gates and gradient SNR tracking."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
	"""
	A linear layer with learnable multiplicative gates on each weight.

	This layer behaves like nn.Linear but applies element-wise gating:
		W_pruned = W * sigmoid(gate_scores)

	Additionally, it tracks gradient statistics (mean and variance)
	to estimate signal-to-noise ratio (SNR) for each connection.
	"""

	def __init__(
		self,
		in_features: int,
		out_features: int,
		bias: bool = True,
		beta: float = 0.9,
	) -> None:
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.beta = beta

		self.weight = nn.Parameter(torch.empty(out_features, in_features))
		self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

		if bias:
			self.bias = nn.Parameter(torch.empty(out_features))
		else:
			self.register_parameter("bias", None)

		# Running gradient statistics for gate scores.
		self.register_buffer("grad_mean", torch.zeros(out_features, in_features))
		self.register_buffer("grad_var", torch.zeros(out_features, in_features))
		self.register_buffer("step_count", torch.zeros((), dtype=torch.long))

		self.reset_parameters()
		self.gate_scores.register_hook(self._gate_gradient_hook)

	def reset_parameters(self) -> None:
		"""Initialize learnable parameters with stable defaults."""
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
			nn.init.uniform_(self.bias, -bound, bound)

		# logit(0.9): sigmoid(gate_scores) starts near 0.9.
		with torch.no_grad():
			self.gate_scores.fill_(math.log(0.9 / 0.1))

	def _gate_gradient_hook(self, grad: Tensor) -> Tensor:
		"""Update EMA statistics for gate-score gradients and pass grad through."""
		with torch.no_grad():
			prev_mean = self.grad_mean

			new_mean = self.beta * prev_mean + (1.0 - self.beta) * grad
			new_var = self.beta * self.grad_var + (1.0 - self.beta) * (grad - prev_mean).pow(2)

			self.grad_mean.copy_(new_mean)
			self.grad_var.copy_(new_var)
			self.step_count.add_(1)

		return grad

	def forward(self, input: Tensor) -> Tensor:
		"""Apply gated linear transformation."""
		gates = torch.sigmoid(self.gate_scores)
		pruned_weight = self.weight * gates
		return F.linear(input, pruned_weight, self.bias)

	def get_snr(self) -> Tensor:
		"""Return per-weight gradient SNR estimate."""
		var = self.grad_var.clamp(min=1e-12)
		return self.grad_mean.abs() / (var.sqrt() + 1e-8)

	def get_gates(self) -> Tensor:
		"""Return current gate values in [0, 1]."""
		return torch.sigmoid(self.gate_scores)

	def get_sparsity(self, threshold: float = 1e-2) -> float:
		"""Return percentage of gates below a threshold."""
		with torch.no_grad():
			gates = self.get_gates()
			sparse_fraction = (gates < threshold).float().mean()
			return float(sparse_fraction.item() * 100.0)

	def extra_repr(self) -> str:
		"""Return concise module representation."""
		return (
			f"in_features={self.in_features}, out_features={self.out_features}, "
			f"bias={self.bias is not None}, beta={self.beta}"
		)
