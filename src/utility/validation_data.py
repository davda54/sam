import argparse
import random
from itertools import compress

import torch
import torchvision.transforms as transforms
from cifar_utils import cifar100_stats
from misc_utils import get_project_root
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

dataset_path = get_project_root() / "datasets"
dataset_path.mkdir(parents=True, exist_ok=True)


class CIFAR100Indexed(Dataset):
    def __init__(self, root, download, train, transform):
        self.cifar100 = CIFAR100(
            root=root, download=download, train=train, transform=transform
        )

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)


def make_validation_dataset():
    """
    from the training set, get 100 images for each superclass
    export as validation set
    """

    mean, std = cifar100_stats(root=str(dataset_path))

    validation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    validation_dataset = CIFAR100Indexed(
        root=str(dataset_path),
        train=True,
        download=True,
        transform=validation_transform,
    )

    class_numbers = range(100)
    sampled_indices = []
    for cls in class_numbers:
        mask = [
            t == cls for t in validation_dataset.cifar100.targets
        ]  # get the indices for targets matching our superclass arg
        indices = [
            i for i, e in enumerate(mask) if e
        ]  # get the indices for all the data from superclass
        sampled_indices += random.sample(indices, 20)

    dataset_mask = [False] * 50_000
    for i in sampled_indices:
        dataset_mask[i] = True

    validation_dataset.cifar100.targets = list(
        compress(validation_dataset.cifar100.targets, dataset_mask)
    )  # subset targets using mask
    validation_dataset.cifar100.data = list(
        compress(validation_dataset.cifar100.data, dataset_mask)
    )  # subset data using mask
    validation_dataset.cifar100.meta["type"] = "validation"
    return validation_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=6, type=int, help="Index of GPU to use",
    )
    args = parser.parse_args()

    random.seed(42)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = make_validation_dataset()
    output_path = (
        get_project_root() / "datasets" / "validation" / "validation_dataset.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving validation dataset to file path: {output_path}")
    torch.save(dataset, output_path)
