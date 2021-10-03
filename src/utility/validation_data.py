import random
import argparse
import torch
import torchvision
from itertools import compress
from misc_utils import get_project_root


def make_validation_dataset():
    """
    from the training set, get 100 images for each superclass
    export as validation set
    """
    cifar100 = get_project_root() / "datasets"
    cifar100.mkdir(parents=True, exist_ok=True)
    dataset = torchvision.datasets.CIFAR100(
        root=str(cifar100), train=True, download=True
    )

    class_numbers = range(100)
    sampled_indices = []
    for cls in class_numbers:
        mask = [
            t == cls for t in dataset.targets
        ]  # get the indices for targets matching our superclass arg
        indices = [
            i for i, e in enumerate(mask) if e
        ]  # get the indices for all the data from superclass
        sampled_indices += random.sample(indices, 20)

    dataset_mask = [False] * 50_000
    for i in sampled_indices:
        dataset_mask[i] = True

    dataset.targets = list(
        compress(dataset.targets, dataset_mask)
    )  # subset targets using mask
    dataset.data = list(compress(dataset.data, dataset_mask))  # subset data using mask

    return dataset


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
