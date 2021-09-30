import random
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout
from utility.cifar_utils import (
    coarse_classes,
    coarse_idxs,
    coarse_classes_map,
    load_dataset,
    save_dataset,
)

from src.utility.misc_utils import get_project_root
import sys

sys.path.append("")


def make_cifar100(parsed_args):
    use_fine_classes, crop_size = parsed_args.use_fine_classes, parsed_args.crop_size
    cifar100 = get_project_root() / "datasets"
    cifar100.mkdir(parents=True, exist_ok=True)

    _train_set = torchvision.datasets.CIFAR100(
        root=str(cifar100), train=True, download=True, transform=transforms.ToTensor(),
    )

    _train_data = torch.cat([d[0] for d in DataLoader(_train_set)])
    mean, std = _train_data.mean(dim=[0, 2, 3]), _train_data.std(dim=[0, 2, 3])

    train_transform = transforms.Compose(
        [
            torchvision.transforms.RandomCrop(size=(crop_size, crop_size), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    _train_set = torchvision.datasets.CIFAR100(
        root=str(cifar100), train=True, download=True, transform=train_transform,
    )

    _test_set = torchvision.datasets.CIFAR100(
        root=str(cifar100), train=False, download=True, transform=test_transform,
    )

    if use_fine_classes:
        pass
        # TODO: Make a function to split into 20 datasets by the coarse class
        # train_set # dataset has 100 fine_classes
        # for i in range(20): # there are 20 classes
        #     pass

    else:
        _train_set.classes, _test_set.classes = coarse_classes, coarse_classes
        _train_set.class_to_idx, _test_set.class_to_idx = coarse_idxs, coarse_idxs
        _train_set.targets = list(map(coarse_classes_map.get, _train_set.targets))
        _test_set.targets = list(map(coarse_classes_map.get, _test_set.targets))

    return _train_set, _test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_classes", dest="use_fine_classes", action="store_true")
    parser.add_argument(
        "--coarse_classes", dest="use_fine_classes", action="store_false",
    )
    parser.set_defaults(use_fine_classes=True)
    parser.add_argument(
        "--superclass", default="all", type=str, help="Superclass we want to use",
    )
    parser.add_argument(
        "--crop_size",
        default=32,
        type=int,
        help="Crop size used in data transformations.",
    )
    args = parser.parse_args()
    print(args)

    if args.use_fine_classes:
        args.granularity = "fine"
        if not args.superclass:
            ValueError(
                "Must provide superclass when building datasets with fine labels"
            )
        superclass = str(args.superclass)
    else:
        args.granularity = "coarse"
        if not args.superclass:
            superclass = "all"

    # TODO need to use environmental variables instead of this method to distribute w/ shell commands
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    random.seed(42)
    train_set, test_set = make_cifar100(args)

    save_dataset(train_set, "train", args)
    save_dataset(test_set, "test", args)

    _train_set_ = load_dataset("train", args)
    assert _train_set_
    _test_set_ = load_dataset("test", args)
    assert _test_set_
