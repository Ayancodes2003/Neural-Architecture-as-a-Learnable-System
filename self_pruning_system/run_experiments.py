"""Run end-to-end training and evaluation experiments for multiple sparsity settings."""

from __future__ import annotations

from pathlib import Path

import torch

from data.datamodule import CIFAR10DataModule
from engine.trainer import Trainer
from evaluate import (
    RunResult,
    add_run_result,
    evaluate,
    extract_gates_and_snrs,
    plot_gate_distribution,
    plot_pareto,
    plot_snr_vs_gate,
)
from models.network import CIFAR10PrunableNet


def main() -> None:
    """Execute training, evaluation, checkpointing, and analysis for all modes."""
    lambda_values = [1e-5, 1e-4, 1e-3]
    lambda_to_mode = {
        1e-5: "accurate",
        1e-4: "balanced",
        1e-3: "fast",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10

    checkpoints_dir = Path("checkpoints")
    outputs_dir = Path("outputs")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []

    for lambda_value in lambda_values:
        mode = lambda_to_mode[lambda_value]

        model = CIFAR10PrunableNet().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            lambda_max=lambda_value,
            epochs=epochs,
        )

        data = CIFAR10DataModule(
            batch_size=batch_size,
            num_workers=2,
            total_epochs=epochs,
        )

        for epoch in range(epochs):
            train_loader = data.get_train_loader(epoch)
            trainer.train_epoch(train_loader, epoch)

        checkpoint_path = checkpoints_dir / f"{mode}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        test_loader = data.get_test_loader()
        accuracy = evaluate(model, test_loader, device)
        sparsity = model.get_total_sparsity()

        add_run_result(results, lambda_value=lambda_value, accuracy=accuracy, sparsity=sparsity)

        gates, snrs = extract_gates_and_snrs(model)
        plot_gate_distribution(gates, outputs_dir / f"gate_distribution_{mode}.png")
        plot_snr_vs_gate(snrs, gates, outputs_dir / f"snr_vs_gate_{mode}.png")

    plot_pareto(results, outputs_dir / "pareto.png")

    for result in results:
        print(
            f"lambda={result.lambda_value:.0e}, "
            f"accuracy={result.accuracy:.4f}, "
            f"sparsity={result.sparsity:.2f}"
        )


if __name__ == "__main__":
    main()
