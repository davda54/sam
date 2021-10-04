import random
import argparse
import torch
import torchvision
from itertools import compress
from misc_utils import get_project_root
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

dataset_fp = get_project_root() / "datasets"
dataset_fp.mkdir(parents=True, exist_ok=True)


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {"__getitem__": __getitem__,})


def cifar100_stats():
    _data_set = torchvision.datasets.CIFAR100(
        root=str(dataset_fp),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    _data_tensors = torch.cat([d[0] for d in DataLoader(_data_set)])
    mean, std = _data_tensors.mean(dim=[0, 2, 3]), _data_tensors.std(dim=[0, 2, 3])
    return mean, std


def make_validation_dataset():
    """
    from the training set, get 100 images for each superclass
    export as validation set
    """

    mean, std = cifar100_stats()

    validation_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    validation_dataset = CIFAR100(
        root=str(dataset_fp), train=True, download=True, transform=validation_transform,
    )

    class_numbers = range(100)
    sampled_indices = []
    for cls in class_numbers:
        mask = [
            t == cls for t in validation_dataset.targets
        ]  # get the indices for targets matching our superclass arg
        indices = [
            i for i, e in enumerate(mask) if e
        ]  # get the indices for all the data from superclass
        sampled_indices += random.sample(indices, 20)

    dataset_mask = [False] * 50_000
    for i in sampled_indices:
        dataset_mask[i] = True

    validation_dataset.targets = list(
        compress(validation_dataset.targets, dataset_mask)
    )  # subset targets using mask
    validation_dataset.data = list(
        compress(validation_dataset.data, dataset_mask)
    )  # subset data using mask

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
    fp = get_project_root() / "datasets" / "validation" / "validation_dataset.pt"
    fp.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving validation dataset to file path: {fp}")
    torch.save(dataset, fp)
