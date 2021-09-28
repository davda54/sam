import random
import pickle
import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.utility.cutout import Cutout
from src.utility.cifar_utils import (
    fine_classes,
    coarse_classes,
    coarse_idxs,
    coarse_classes_map,
)


class CifarHundred:
    def __init__(self, use_fine_classes, crop_size, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose(
            [
                torchvision.transforms.RandomCrop(
                    size=(crop_size, crop_size), padding=4
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout(),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        train_set = torchvision.datasets.CIFAR100(
            root=str(Path.cwd()/"datasets"/"cifar100"),
            train=True,
            download=True,
            transform=train_transform,
        )
        test_set = torchvision.datasets.CIFAR100(
            root=str(Path.cwd()/"datasets"/"cifar100"),
            train=False,
            download=True,
            transform=test_transform,
        )

        if use_fine_classes:
            self.classes = fine_classes
        else:
            self.classes = coarse_classes
            train_set.classes, test_set.classes = coarse_classes, coarse_classes
            train_set.class_to_idx, test_set.class_to_idx = coarse_idxs, coarse_idxs
            train_set.targets = list(map(coarse_classes_map.get, train_set.targets))
            test_set.targets = list(map(coarse_classes_map.get, test_set.targets))

        self.train = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=threads
        )
        self.test = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=threads
        )

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR100(
            root=str(Path.cwd()/"datasets"/"cifar100"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


def export_dataset(obj: CifarHundred, split: str):
    if split not in ["train", "test"]:
        raise ValueError("split must be 'train' or 'test'")
    filename = str(
        Path.cwd()
        / "datasets"
        / "cifar100"
        / f"cifar100_{split}_fine{args.use_fine_classes}_crop{args.crop_size}_batch{args.batch_size}_threads{args.threads}.pkl"
    )
    file = open(filename, "wb")
    pickle.dump(getattr(obj, split), file)
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_classes", dest="use_fine_classes", action="store_true")
    parser.add_argument(
        "--coarse_classes", dest="use_fine_classes", action="store_false",
    )
    parser.set_defaults(use_fine_classes=True)
    parser.add_argument(
        "--crop_size",
        default=32,
        type=int,
        help="Crop size used in data transformations.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size used in the training and validation loop.",
    )
    parser.add_argument(
        "--threads", default=2, type=int, help="Number of CPU threads for dataloaders."
    )
    args = parser.parse_args()

    print(args)

    random.seed(42)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    dataset = CifarHundred(
        args.use_fine_classes, args.crop_size, args.batch_size, args.threads
    )
    export_dataset(dataset, "train")
    export_dataset(dataset, "test")
