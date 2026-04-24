"""CIFAR-10 data module with progressive epoch-based augmentation."""

from __future__ import annotations

from typing import List, Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10DataModule:
	"""Reusable CIFAR-10 data pipeline with progressive augmentation stages."""

	def __init__(
		self,
		batch_size: int,
		num_workers: int,
		total_epochs: int,
		data_dir: str = "./data",
	) -> None:
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.data_dir = data_dir
		self.total_epochs = total_epochs

		self.mean = (0.4914, 0.4822, 0.4465)
		self.std = (0.2470, 0.2430, 0.2610)

		self._train_dataset: Optional[datasets.CIFAR10] = None
		self._test_dataset: Optional[datasets.CIFAR10] = None

	def get_train_transform(self, epoch: int) -> transforms.Compose:
		"""Return a progressive train transform based on the current epoch."""
		aug_transforms: List[transforms.Transform] = [
			transforms.RandomCrop(32, padding=4),
		]

		if epoch < 0.3 * self.total_epochs:
			pass
		elif epoch < 0.7 * self.total_epochs:
			aug_transforms.append(transforms.RandomHorizontalFlip())
		else:
			aug_transforms.extend(
				[
					transforms.RandomHorizontalFlip(),
					transforms.ColorJitter(
						brightness=0.2,
						contrast=0.2,
						saturation=0.2,
					),
				]
			)

		aug_transforms.extend(
			[
				transforms.ToTensor(),
				transforms.Normalize(self.mean, self.std),
			]
		)
		return transforms.Compose(aug_transforms)

	def get_train_loader(self, epoch: int) -> DataLoader:
		"""Build and return the training dataloader for a given epoch."""
		if self._train_dataset is None:
			self._train_dataset = datasets.CIFAR10(
				root=self.data_dir,
				train=True,
				transform=self.get_train_transform(epoch),
				download=True,
			)
		else:
			self._train_dataset.transform = self.get_train_transform(epoch)

		return DataLoader(
			self._train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			pin_memory=True,
			persistent_workers=self.num_workers > 0,
		)

	def get_test_loader(self) -> DataLoader:
		"""Build and return the test dataloader without augmentations."""
		test_transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(self.mean, self.std),
			]
		)

		if self._test_dataset is None:
			self._test_dataset = datasets.CIFAR10(
				root=self.data_dir,
				train=False,
				transform=test_transform,
				download=True,
			)
		else:
			self._test_dataset.transform = test_transform

		return DataLoader(
			self._test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=True,
			persistent_workers=self.num_workers > 0,
		)
